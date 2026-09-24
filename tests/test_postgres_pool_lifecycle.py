"""A partially initialized asyncpg pool must never strand server sessions."""

import asyncio
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from engram.storage.postgres import PostgresBackend


@pytest.mark.parametrize(
    "failure", ["exception", "cancellation"], ids=["postgres-exception", "postgres-cancellation"]
)
async def test_partial_pool_initialization_releases_connections_and_retries(monkeypatch, failure):
    admin_dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
    if not admin_dsn:
        pytest.skip("requires a disposable PostgreSQL admin DSN")
    admin = await asyncpg.connect(admin_dsn)
    database = "engram_pool_" + uuid.uuid4().hex
    dsn = urlunsplit(urlsplit(admin_dsn)._replace(path="/" + database))
    backend = PostgresBackend(dsn)
    original = backend._init_connection
    second_started = asyncio.Event()
    first_pid = None
    calls = 0

    async def initialize(conn):
        nonlocal calls, first_pid
        calls += 1
        await original(conn)
        if calls == 1:
            first_pid = conn.get_server_pid()
        else:
            second_started.set()
            if failure == "exception":
                raise RuntimeError("second connection failed")
            await asyncio.Event().wait()

    task = None
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
        monkeypatch.setattr(backend, "_init_connection", initialize)
        task = asyncio.create_task(backend._get_pool())
        await asyncio.wait_for(second_started.wait(), timeout=10)
        if failure == "cancellation":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="second connection failed"):
                await task
        assert backend._pool is None
        assert first_pid is not None
        for _ in range(50):
            remaining = await admin.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE datname=$1", database
            )
            if remaining == 0:
                break
            await asyncio.sleep(0.02)
        assert remaining == 0, "partially initialized pool leaked database sessions"
        monkeypatch.setattr(backend, "_init_connection", original)
        assert await backend._get_pool() is backend._pool
        await backend.close()
        assert backend._pool is None
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await backend.close()
        await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        await admin.close()
