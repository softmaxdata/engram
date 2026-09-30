"""Storage transactions isolate tasks and retain rollback boundaries."""

import asyncio
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from engram.core.config import Settings
from engram.core.models import Bullet, Context, IntentAnchor
from engram.storage.postgres import PostgresBackend
from engram.storage.sqlite import SQLiteBackend


@pytest.fixture(params=[":memory:", "file", "postgres"])
async def storage(request, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "engram.core.config.get_settings", lambda: Settings(_env_file=None, auth_enabled=False)
    )
    if request.param == "postgres":
        admin_dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
        if not admin_dsn:
            pytest.skip("requires disposable PostgreSQL admin DSN")
        admin = await asyncpg.connect(admin_dsn)
        database = "engram_tx_" + uuid.uuid4().hex
        dsn = urlunsplit(urlsplit(admin_dsn)._replace(path="/" + database))
        backend = PostgresBackend(dsn)
        try:
            await admin.execute(f'CREATE DATABASE "{database}"')
            await backend.initialize()
            context = await backend.create_context(
                Context(name="tx", intent=IntentAnchor(objective="test"))
            )
            yield backend, str(context.id)
        finally:
            await backend.close()
            await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
            await admin.close()
        return
    path = ":memory:" if request.param == ":memory:" else str(tmp_path / "transactions.db")
    backend = SQLiteBackend(path)
    await backend.initialize()
    context = await backend.create_context(
        Context(name="tx", intent=IntentAnchor(objective="test"))
    )
    yield backend, str(context.id)
    await backend.close()


async def test_commit_and_rollback(storage):
    backend, context_id = storage
    async with backend.transaction(context_id) as tx:
        await tx.add_bullet(context_id, Bullet(id="committed", content="first"))
    with pytest.raises(ValueError):
        async with backend.transaction(context_id) as tx:
            await tx.add_bullet(context_id, Bullet(id="rolledback", content="second"))
            raise ValueError("abort")
    assert [b.id for b in await backend.list_bullets(context_id)] == ["committed"]


async def test_scoped_handles_reject_parent_wrong_task_and_expired_use(storage):
    backend, context_id = storage
    async with backend.transaction(context_id) as tx:
        with pytest.raises(RuntimeError):
            await backend.list_bullets(context_id)
        with pytest.raises(RuntimeError):
            async with backend.transaction(context_id):
                pass
        with pytest.raises(RuntimeError):
            await asyncio.create_task(tx.list_bullets(context_id))
        with pytest.raises(RuntimeError):
            await tx.close()
    with pytest.raises(RuntimeError):
        await tx.list_bullets(context_id)


async def test_nested_savepoint_rollback(storage):
    backend, context_id = storage
    async with backend.transaction(context_id) as tx:
        await tx.add_bullet(context_id, Bullet(id="outer", content="kept"))
        with pytest.raises(ValueError):
            async with tx.transaction(context_id) as nested:
                await nested.add_bullet(context_id, Bullet(id="inner", content="discarded"))
                raise ValueError("rollback nested only")
        assert [b.id for b in await tx.list_bullets(context_id)] == ["outer"]
        with pytest.raises(ValueError):
            async with tx.transaction("different"):
                pass
    assert [b.id for b in await backend.list_bullets(context_id)] == ["outer"]


async def test_other_task_cannot_read_or_commit_uncommitted_state(storage):
    backend, context_id = storage
    entered = asyncio.Event()
    release = asyncio.Event()

    async def writer():
        with pytest.raises(ValueError):
            async with backend.transaction(context_id) as tx:
                await tx.add_bullet(context_id, Bullet(id="uncommitted", content="discarded"))
                entered.set()
                await release.wait()
                raise ValueError("abort")

    task = asyncio.create_task(writer())
    await entered.wait()
    outside = asyncio.create_task(
        backend.add_bullet(context_id, Bullet(id="outside", content="kept"))
    )
    await asyncio.sleep(0)
    if isinstance(backend, SQLiteBackend):
        assert not outside.done()
    release.set()
    await asyncio.gather(task, outside)
    assert [b.id for b in await backend.list_bullets(context_id)] == ["outside"]


async def test_cancellation_rolls_back_and_connection_is_reusable(storage):
    backend, context_id = storage
    entered = asyncio.Event()

    async def writer():
        async with backend.transaction(context_id) as tx:
            await tx.add_bullet(context_id, Bullet(id="cancelled", content="discarded"))
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(writer())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await backend.list_bullets(context_id) == []
    async with backend.transaction(context_id) as tx:
        await tx.add_bullet(context_id, Bullet(id="after", content="usable"))
    assert (await backend.get_bullet("after")).content == "usable"


@pytest.mark.parametrize("during", ["begin", "ordinary_write", "savepoint"])
async def test_sqlite_cancellation_during_queued_sql_does_not_leak_transaction(
    storage, monkeypatch, during
):
    backend, context_id = storage
    if not isinstance(backend, SQLiteBackend):
        pytest.skip("SQLite worker queue regression")
    db = await backend._get_db()
    original = db.execute
    entered = asyncio.Event()

    async def delayed(sql, *args, **kwargs):
        result = await original(sql, *args, **kwargs)
        matches = (
            (during == "begin" and sql == "BEGIN IMMEDIATE")
            or (during == "ordinary_write" and sql.startswith("INSERT INTO bullets"))
            or (during == "savepoint" and sql.startswith("SAVEPOINT"))
        )
        if matches:
            entered.set()
            await asyncio.Event().wait()
        return result

    monkeypatch.setattr(db, "execute", delayed)

    async def operation():
        if during == "ordinary_write":
            await backend.add_bullet(context_id, Bullet(id="cancelled", content="discarded"))
        else:
            async with backend.transaction(context_id) as tx:
                if during == "savepoint":
                    async with tx.transaction(context_id):
                        pass

    task = asyncio.create_task(operation())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(db, "execute", original)
    assert not db.in_transaction
    assert await backend.list_bullets(context_id) == []
    async with backend.transaction(context_id) as tx:
        await tx.add_bullet(context_id, Bullet(id="next", content="usable"))


async def test_read_modify_write_serializes_concurrent_archive(storage):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="active", content="fact"))
    if isinstance(backend, PostgresBackend):
        other = PostgresBackend(backend.dsn, existing_only=True)
    elif backend.db_path != ":memory:":
        other = SQLiteBackend(backend.db_path, existing_only=True)
    else:
        other = backend
    fetched = asyncio.Event()
    release = asyncio.Event()

    async def reinforce():
        async with backend.transaction(context_id) as tx:
            bullet = await tx.get_bullet("active")
            fetched.set()
            await release.wait()
            bullet.hit_count += 1
            await tx.update_bullet(bullet)

    writer = asyncio.create_task(reinforce())
    await asyncio.wait_for(fetched.wait(), timeout=5)
    archive = asyncio.create_task(other.archive_bullet(context_id, "active"))
    try:
        # An archive must wait for the locked read, otherwise the stale full-row
        # stats update would silently resurrect the archived bullet.
        await asyncio.sleep(0.05)
        assert not archive.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(writer, archive), timeout=5)
        after = await backend.get_bullet("active")
        assert after.is_archived
        assert after.hit_count == 1
    finally:
        release.set()
        for task in (writer, archive):
            if not task.done():
                task.cancel()
        await asyncio.gather(writer, archive, return_exceptions=True)
        if other is not backend:
            await other.close()
