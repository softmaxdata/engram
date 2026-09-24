"""Archive expiry, explicit purge and REST restores preserve concurrent state."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import aiosqlite
import asyncpg
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from test_storage_transactions import storage as transaction_storage

from engram.core.models import Bullet, ConceptEdge, EdgeType
from engram.server.routes.lifecycle import restore_bullet
from engram.storage.postgres import PostgresBackend
from engram.storage.sqlite import SQLiteBackend

storage = transaction_storage


async def archived_with_edge(backend, context_id):
    first, second = uuid.uuid4(), uuid.uuid4()
    await backend.add_bullet(context_id, Bullet(id=str(first), content="archived"))
    await backend.add_bullet(context_id, Bullet(id=str(second), content="other"))
    await backend.archive_bullet(context_id, str(first))
    bullet = await backend.get_bullet(str(first))
    bullet.archived_at = datetime.now(UTC) - timedelta(days=200)
    await backend.update_bullet(bullet)
    edge = await backend.add_edge(
        uuid.UUID(context_id),
        ConceptEdge(from_node=first, to_node=second, type=EdgeType.RELATED_TO),
    )
    return str(first), edge.id


def sql_connection_type(backend):
    return aiosqlite.Connection if isinstance(backend, SQLiteBackend) else asyncpg.Connection


@pytest.mark.parametrize("mode", ["expired", "direct"])
async def test_purge_failure_preserves_bullet_and_edges(storage, monkeypatch, mode):
    backend, context_id = storage
    bullet_id, edge_id = await archived_with_edge(backend, context_id)
    cls = sql_connection_type(backend)
    execute = cls.execute

    async def fail(self, query, *args, **kwargs):
        if query.startswith("DELETE FROM bullets"):
            raise RuntimeError("injected bullet deletion failure")
        return await execute(self, query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cls, "execute", fail)
        with pytest.raises(RuntimeError, match="injected bullet deletion"):
            if mode == "expired":
                await backend.purge_expired_archives(context_id, 180)
            else:
                await backend.purge_bullet(context_id, bullet_id)
    assert await backend.get_bullet(bullet_id) is not None
    assert [edge.id for edge in await backend.get_edges(uuid.UUID(context_id))] == [edge_id]
    assert await backend.purge_bullet(context_id, bullet_id)
    assert await backend.get_bullet(bullet_id) is None
    assert await backend.get_edges(uuid.UUID(context_id)) == []


@pytest.mark.parametrize("storage", ["file", "postgres"], indirect=True)
async def test_successful_concurrent_restore_is_not_purged(storage, monkeypatch):
    backend, context_id = storage
    bullet_id, edge_id = await archived_with_edge(backend, context_id)
    peer = (
        SQLiteBackend(backend.db_path, existing_only=True)
        if isinstance(backend, SQLiteBackend)
        else PostgresBackend(backend.dsn, existing_only=True)
    )
    at_delete, release = asyncio.Event(), asyncio.Event()
    cls = sql_connection_type(backend)
    execute = cls.execute
    purge_task = None

    async def pause(self, query, *args, **kwargs):
        if asyncio.current_task() is purge_task and query.startswith("DELETE FROM edges"):
            at_delete.set()
            await release.wait()
        return await execute(self, query, *args, **kwargs)

    monkeypatch.setattr(cls, "execute", pause)
    purge_task = asyncio.create_task(backend.purge_expired_archives(context_id, 180))
    restore_task = None
    try:
        await asyncio.wait_for(at_delete.wait(), timeout=5)
        restore_task = asyncio.create_task(peer.restore_bullet(context_id, bullet_id))
        await asyncio.wait({restore_task}, timeout=0.25)
        release.set()
        purged, restored = await asyncio.wait_for(
            asyncio.gather(purge_task, restore_task), timeout=10
        )
        if restored is not None:
            assert purged == 0
            current = await backend.get_bullet(bullet_id)
            assert current is not None and not current.is_archived
            assert [e.id for e in await backend.get_edges(uuid.UUID(context_id))] == [edge_id]
        else:
            assert purged == 1
            assert await backend.get_bullet(bullet_id) is None
    finally:
        release.set()
        await asyncio.gather(
            *[t for t in (purge_task, restore_task) if t is not None], return_exceptions=True
        )
        await peer.close()


@pytest.mark.parametrize("storage", ["file", "postgres"], indirect=True)
async def test_concurrent_restores_respect_context_capacity(storage, monkeypatch):
    backend, context_id = storage
    context = await backend.get_context(uuid.UUID(context_id))
    context.lifecycle_config.max_active_bullets = 1
    await backend.update_context(context)
    for bullet_id in ("one", "two"):
        await backend.add_bullet(context_id, Bullet(id=bullet_id, content=bullet_id))
        await backend.archive_bullet(context_id, bullet_id)
    peer = (
        SQLiteBackend(backend.db_path, existing_only=True)
        if isinstance(backend, SQLiteBackend)
        else PostgresBackend(backend.dsn, existing_only=True)
    )
    capacity = type(backend).get_capacity_status
    both_read = asyncio.Barrier(2)

    async def synchronized_read(self, *args, **kwargs):
        value = await capacity(self, *args, **kwargs)
        if self._scope is None:
            await both_read.wait()
        return value

    monkeypatch.setattr(type(backend), "get_capacity_status", synchronized_read)

    async def restore(store, bullet_id):
        request = Request({"type": "http", "app": SimpleNamespace(
            state=SimpleNamespace(storage=store)), "state": {"user_id": context.owner}})
        try:
            return await restore_bullet(context_id, bullet_id, request)
        except HTTPException as exc:
            return exc.status_code

    try:
        results = await asyncio.wait_for(asyncio.gather(
            restore(backend, "one"), restore(peer, "two")), timeout=10)
        assert sum(not isinstance(result, int) for result in results) == 1
        assert 409 in results
        assert len(await backend.list_bullets(context_id)) == 1
    finally:
        await peer.close()
