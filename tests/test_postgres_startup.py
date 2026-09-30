"""Fresh-database startup regressions for GitHub issue #3.

Set ENGRAM_TEST_POSTGRES_DSN to a disposable PostgreSQL admin database to
also run the integration test. It creates and removes its own temporary DB.
"""

import os
import uuid
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from engram.core.models import Bullet, Context, IntentAnchor
from engram.storage.postgres import PostgresBackend


class AwaitablePool:
    """Match asyncpg's synchronous factory returning an awaitable Pool."""

    def __init__(self):
        self.initialize = None
        self.close = AsyncMock()
        self.terminate = Mock()

    def __await__(self):
        async def initialize():
            if self.initialize is not None:
                await self.initialize()
            return self

        return initialize().__await__()


async def test_extension_exists_before_pool_registers_codecs(monkeypatch):
    events = []
    connection = AsyncMock()
    connection.execute.side_effect = lambda sql: events.append(sql)
    connection.close.side_effect = lambda: events.append("closed")
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=connection))
    pool = AwaitablePool()

    async def register(conn):
        assert events == ["CREATE EXTENSION IF NOT EXISTS vector", "closed"]
        events.append("registered")

    def create_pool(*args, **kwargs):
        pool.initialize = lambda: kwargs["init"](AsyncMock())
        return pool

    monkeypatch.setattr("engram.storage.postgres.register_vector", register)
    create = Mock(side_effect=create_pool)
    monkeypatch.setattr(asyncpg, "create_pool", create)
    backend = PostgresBackend("postgresql://test")
    assert await backend._get_pool() is pool
    assert await backend._get_pool() is pool
    create.assert_called_once()
    connection.close.assert_awaited_once()
    await backend.close()


async def test_failed_extension_setup_closes_connection_and_can_retry(monkeypatch):
    connection = AsyncMock()
    connection.execute.side_effect = asyncpg.InsufficientPrivilegeError("install vector first")
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=connection))
    create = Mock(return_value=AwaitablePool())
    monkeypatch.setattr(asyncpg, "create_pool", create)
    backend = PostgresBackend("postgresql://test")
    with pytest.raises(asyncpg.InsufficientPrivilegeError, match="install vector first"):
        await backend.initialize()
    connection.close.assert_awaited_once()
    create.assert_not_called()
    assert backend._pool is None

    connection.execute.side_effect = None
    await backend._get_pool()
    create.assert_called_once()
    assert connection.close.await_count == 2
    await backend.close()


async def test_failed_pool_creation_is_not_cached(monkeypatch):
    connection = AsyncMock()
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=connection))
    create = Mock(side_effect=RuntimeError("pool failed"))
    monkeypatch.setattr(asyncpg, "create_pool", create)
    backend = PostgresBackend("postgresql://test")
    with pytest.raises(RuntimeError, match="pool failed"):
        await backend._get_pool()
    connection.close.assert_awaited_once()
    assert backend._pool is None


@pytest.mark.skipif(
    not os.environ.get("ENGRAM_TEST_POSTGRES_DSN"),
    reason="requires a disposable PostgreSQL admin DSN",
)
async def test_fresh_postgres_vector_roundtrip_and_restart():
    admin_dsn = os.environ["ENGRAM_TEST_POSTGRES_DSN"]
    admin = await asyncpg.connect(admin_dsn)
    db_name = "engram_test_" + uuid.uuid4().hex
    role_name = "engram_test_" + uuid.uuid4().hex
    parts = urlsplit(admin_dsn)
    dsn = urlunsplit(parts._replace(path="/" + db_name))
    backend = PostgresBackend(dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
        await backend.initialize()
        ctx = await backend.create_context(
            Context(name="Startup test", intent=IntentAnchor(objective="Vector roundtrip"))
        )
        embedding = [1.0] + [0.0] * 1535
        bullet = Bullet(content="Persist across restart", embedding=embedding)
        await backend.add_bullet(str(ctx.id), bullet)
        await backend.close()

        await backend.initialize()
        stored = await backend.get_bullet(bullet.id)
        assert stored.embedding == pytest.approx(embedding)
        matches = await backend.find_similar_bullets(str(ctx.id), embedding, threshold=0.99)
        assert matches[0][0].id == bullet.id

        # Existing extension does not require the app role to be a superuser.
        await admin.execute(f'CREATE ROLE "{role_name}"')
        async with backend._pool.acquire() as conn:
            await conn.execute(f'SET ROLE "{role_name}"')
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await conn.execute("RESET ROLE")
    finally:
        await backend.close()
        await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        await admin.execute(f'DROP ROLE IF EXISTS "{role_name}"')
        await admin.close()
