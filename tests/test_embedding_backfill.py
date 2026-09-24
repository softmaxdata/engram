"""Exercise repair only against disposable databases and fake providers."""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from engram.core.models import Bullet, Context, IntentAnchor
from engram.maintenance import backfill_embeddings as repair_module
from engram.maintenance.backfill_embeddings import EMBEDDING_DIMENSIONS, backfill_embeddings
from engram.storage.postgres import PostgresBackend
from engram.storage.sqlite import SQLiteBackend

VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


@pytest.fixture(params=["sqlite", "postgres"])
async def repair_storage(request, tmp_path):
    if request.param == "sqlite":
        backend = SQLiteBackend(str(tmp_path / "repair.db"))
        await backend.initialize()
        try:
            yield backend
        finally:
            await backend.close()
        return
    dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("requires a disposable PostgreSQL admin DSN")
    admin = await asyncpg.connect(dsn)
    name = "engram_backfill_" + uuid.uuid4().hex
    target = urlunsplit(urlsplit(dsn)._replace(path="/" + name))
    backend = PostgresBackend(target)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        await backend.initialize()
        yield backend
    finally:
        await backend.close()
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()


async def seed(storage):
    context = await storage.create_context(
        Context(name="repair", intent=IntentAnchor(objective="preserve existing memories"))
    )
    bullet = await storage.add_bullet(
        str(context.id), Bullet(id="a-missing", content="Use dark mode", hit_count=3, salience=0.7)
    )
    return context, bullet


async def test_preview_never_calls_provider_or_transaction(repair_storage, monkeypatch):
    context, bullet = await seed(repair_storage)
    llm = AsyncMock()
    transaction = AsyncMock(side_effect=AssertionError("preview attempted a transaction"))
    monkeypatch.setattr(repair_storage, "transaction", transaction)
    result = await backfill_embeddings(repair_storage, context.id, llm=llm)
    assert result["dry_run"] and result["eligible"] == result["selected"] == 1
    assert result["attempted"] == result["updated"] == result["failed"] == 0
    llm.embed.assert_not_called()
    transaction.assert_not_called()
    assert (await repair_storage.get_bullet(bullet.id)).embedding is None


async def test_repairs_only_embedding_and_rerun_is_idempotent(repair_storage):
    context, bullet = await seed(repair_storage)
    before = (await repair_storage.get_bullet(bullet.id)).model_dump()
    llm = AsyncMock()
    llm.embed.return_value = VECTOR
    first = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert first["updated"] == 1 and first["failed"] == 0
    after = (await repair_storage.get_bullet(bullet.id)).model_dump()
    assert after.pop("embedding") == pytest.approx(VECTOR)
    before.pop("embedding")
    assert after == before  # Includes timestamps, lifecycle, counters, sources.
    assert await repair_storage.list_delta_batches(str(context.id)) == []
    again = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert again["eligible"] == again["attempted"] == 0
    llm.embed.assert_awaited_once_with(bullet.content)


async def test_limit_context_and_lifecycle_are_respected(repair_storage):
    context, _ = await seed(repair_storage)
    for bid in ("b-missing", "c-archived", "d-inactive"):
        await repair_storage.add_bullet(str(context.id), Bullet(id=bid, content=bid))
    await repair_storage.archive_bullet(str(context.id), "c-archived")
    await repair_storage.remove_bullet("d-inactive")
    foreign = await repair_storage.create_context(
        Context(name="foreign", intent=IntentAnchor(objective="untouched"))
    )
    await repair_storage.add_bullet(str(foreign.id), Bullet(id="foreign", content="Private"))
    llm = AsyncMock()
    llm.embed.return_value = VECTOR
    result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True, limit=1)
    assert result["eligible"] == 2 and result["updated"] == result["selected"] == 1
    for bid in ("b-missing", "c-archived", "d-inactive", "foreign"):
        assert (await repair_storage.get_bullet(bid)).embedding is None


@pytest.mark.parametrize("race", ["text", "archive", "already_repaired"])
async def test_provider_result_is_skipped_after_concurrent_change(repair_storage, race):
    context, bullet = await seed(repair_storage)

    async def embed(_):
        if race == "archive":
            await repair_storage.archive_bullet(str(context.id), bullet.id)
        else:
            current = await repair_storage.get_bullet(bullet.id)
            if race == "text":
                current.content = "A newer instruction"
            else:
                current.embedding = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)
            await repair_storage.update_bullet(current)
        return VECTOR

    llm = AsyncMock()
    llm.embed.side_effect = embed
    result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert result["skipped"] == 1 and result["updated"] == result["failed"] == 0
    after = await repair_storage.get_bullet(bullet.id)
    if race == "text":
        assert after.content == "A newer instruction" and after.embedding is None
    elif race == "archive":
        assert after.is_archived and after.embedding is None
    else:
        assert after.embedding[1] == 1.0


@pytest.mark.parametrize("invalid", [
    [], [1.0], [float("nan")] * 1536, [0.0] * 1536,
    [1e-300] + [0.0] * 1535, [10**1000] + [0.0] * 1535,
    [1e100] + [0.0] * 1535, [1e-30] + [0.0] * 1535, [1e20] + [0.0] * 1535,
])
async def test_invalid_provider_vectors_are_not_written(repair_storage, invalid):
    context, bullet = await seed(repair_storage)
    llm = AsyncMock()
    llm.embed.return_value = invalid
    result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert result["failed"] == 1 and result["errors"][0]["reason"] == "invalid_vector"
    assert (await repair_storage.get_bullet(bullet.id)).embedding is None


async def test_every_saved_repair_is_searchable_by_its_own_vector(repair_storage):
    context, _ = await seed(repair_storage)
    for bid in ("b-underflow", "c-overflow"):
        await repair_storage.add_bullet(str(context.id), Bullet(id=bid, content=bid))
    llm = AsyncMock()
    llm.embed.side_effect = [VECTOR, [1e-30] + [0.0] * 1535, [1e20] + [0.0] * 1535]
    result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert (result["updated"], result["failed"]) == (1, 2)
    for bullet in await repair_storage.list_bullets(str(context.id)):
        if bullet.embedding is not None:
            matches = await repair_storage.find_similar_bullets(
                str(context.id), bullet.embedding, threshold=0.99,
            )
            assert any(found.id == bullet.id for found, _ in matches)
    retry = await backfill_embeddings(repair_storage, context.id)
    assert retry["eligible"] == 2  # Rejected vectors remain repairable.


async def test_provider_failure_preserves_progress_and_redacts_message(repair_storage):
    context, bullet = await seed(repair_storage)
    await repair_storage.add_bullet(str(context.id), Bullet(id="b-second", content="Second"))
    llm = AsyncMock()
    llm.embed.side_effect = [RuntimeError("credential-should-never-be-printed"), VECTOR]
    result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert (result["updated"], result["failed"]) == (1, 1)
    assert "credential-should-never-be-printed" not in json.dumps(result)
    assert (await repair_storage.get_bullet(bullet.id)).embedding is None
    llm.embed.side_effect = None
    llm.embed.return_value = VECTOR
    retry = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert retry["updated"] == 1 and retry["failed"] == 0


async def test_failed_vector_write_rolls_back_and_can_be_retried(repair_storage, monkeypatch):
    context, bullet = await seed(repair_storage)
    cls = type(repair_storage)
    original = cls.update_bullet_embedding_if_missing

    async def fail_after_write(self, *args):
        await original(self, *args)
        raise RuntimeError("injected after SQL")

    llm = AsyncMock()
    llm.embed.return_value = VECTOR
    with monkeypatch.context() as patch:
        patch.setattr(cls, "update_bullet_embedding_if_missing", fail_after_write)
        result = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert result["failed"] == 1
    assert (await repair_storage.get_bullet(bullet.id)).embedding is None
    retry = await backfill_embeddings(repair_storage, context.id, llm=llm, apply=True)
    assert retry["updated"] == 1


async def test_read_only_backend_previews_existing_database(repair_storage):
    context, _ = await seed(repair_storage)
    if isinstance(repair_storage, SQLiteBackend):
        reader = SQLiteBackend(repair_storage.db_path, read_only=True, existing_only=True)
    else:
        reader = PostgresBackend(repair_storage.dsn, read_only=True, existing_only=True)
    try:
        assert (await backfill_embeddings(reader, context.id))["eligible"] == 1
        with pytest.raises(Exception):
            await reader.update_bullet_embedding_if_missing(
                str(context.id), "a-missing", "Use dark mode", VECTOR,
            )
    finally:
        await reader.close()


async def test_independent_repair_workers_save_only_one_vector(repair_storage):
    context, bullet = await seed(repair_storage)
    if isinstance(repair_storage, SQLiteBackend):
        other = SQLiteBackend(repair_storage.db_path, existing_only=True)
    else:
        other = PostgresBackend(repair_storage.dsn, existing_only=True)
    arrived = asyncio.Event()
    calls = 0

    async def embed(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            arrived.set()
        await arrived.wait()
        return VECTOR

    llm = AsyncMock()
    llm.embed.side_effect = embed
    try:
        results = await asyncio.wait_for(asyncio.gather(
            backfill_embeddings(repair_storage, context.id, llm=llm, apply=True),
            backfill_embeddings(other, context.id, llm=llm, apply=True),
        ), timeout=15)
    finally:
        await other.close()
    assert sorted(result["updated"] for result in results) == [0, 1]
    assert sum(result["skipped"] for result in results) == 1
    assert all(result["failed"] == 0 for result in results)
    assert (await repair_storage.get_bullet(bullet.id)).embedding == pytest.approx(VECTOR)


@pytest.mark.parametrize("limit", [0, -1, 1001])
async def test_invalid_limits_fail_before_opening_storage(limit):
    storage = AsyncMock()
    with pytest.raises(ValueError, match="limit"):
        await backfill_embeddings(storage, uuid.uuid4(), limit=limit)
    storage.get_context.assert_not_called()


async def run_cli(*args, env=None):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "engram.maintenance.backfill_embeddings", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    return process.returncode, stdout.decode(), stderr.decode()


async def test_cli_preview_does_not_change_sqlite_file_or_journal_mode(tmp_path):
    path = tmp_path / "existing.db"
    store = SQLiteBackend(str(path))
    await store.initialize()
    context, _ = await seed(store)
    await store.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    code, stdout, _ = await run_cli("--sqlite", str(path), "--context-id", str(context.id))
    assert code == 0 and json.loads(stdout)["dry_run"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not Path(str(path) + "-wal").exists()


async def test_cli_missing_database_is_not_created_and_apply_requires_explicit_provider(tmp_path):
    target = tmp_path / "must-not-exist.db"
    args = ("--sqlite", str(target), "--context-id", str(uuid.uuid4()))
    code, stdout, _ = await run_cli(*args)
    assert code == 1 and json.loads(stdout)["error"] == "database_operation_failed"
    assert not target.exists()
    code, _, stderr = await run_cli(*args, "--apply")
    assert code == 2 and "requires --embedding-model" in stderr
    assert not target.exists()


async def test_cli_requires_a_database_target():
    code, _, stderr = await run_cli("--context-id", str(uuid.uuid4()))
    assert code == 2 and "required" in stderr


async def test_cli_apply_failure_returns_nonzero_without_raw_provider_log(
    tmp_path, monkeypatch, capsys, caplog,
):
    path = tmp_path / "repair.db"
    storage = SQLiteBackend(str(path))
    await storage.initialize()
    context, _ = await seed(storage)
    await storage.close()
    secret = "a-provider-key-that-must-not-be-logged"

    async def fail(_):
        logging.getLogger("engram.llm.adapter").error("provider said %s", secret)
        raise RuntimeError(secret)

    llm = AsyncMock()
    llm.embed.side_effect = fail
    monkeypatch.setattr(repair_module, "LiteLLMAdapter", lambda **kwargs: llm)
    monkeypatch.setenv("ENGRAM_REPAIR_TEST_KEY", secret)
    args = repair_module._parser().parse_args([
        "--sqlite", str(path), "--context-id", str(context.id), "--apply",
        "--embedding-model", "test-only", "--embedding-key-env", "ENGRAM_REPAIR_TEST_KEY",
    ])
    assert await repair_module._run(args) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["failed"] == 1
    assert secret not in captured.out + captured.err + caplog.text


async def test_real_adapter_invalid_model_does_not_load_dotenv_or_corrupt_json(tmp_path):
    path = tmp_path / "repair.db"
    storage = SQLiteBackend(str(path))
    await storage.initialize()
    context, _ = await seed(storage)
    await storage.close()
    (tmp_path / ".env").write_text("ENGRAM_DOTENV_CANARY=must-not-load\n")
    script = '''
import os, socket, sys
import dotenv
calls = []
def forbid_dotenv(*args, **kwargs):
    calls.append("dotenv")
    raise AssertionError("implicit dotenv access")
def forbid_network(*args, **kwargs):
    calls.append("network")
    raise AssertionError("network access")
dotenv.load_dotenv = forbid_dotenv
socket.socket.connect = forbid_network
from engram.maintenance.backfill_embeddings import main
code = main(["--sqlite", sys.argv[1], "--context-id", sys.argv[2], "--apply",
    "--embedding-model", "intentionally-unsupported-test-model",
    "--embedding-key-env", "ENGRAM_REPAIR_TEST_KEY"])
assert calls == [], calls
assert "ENGRAM_DOTENV_CANARY" not in os.environ
assert os.environ["LITELLM_MODE"] == "DEV"
raise SystemExit(code)
'''
    env = {
        **os.environ, "ENGRAM_REPAIR_TEST_KEY": "fake-provider-key-canary",
        "LITELLM_MODE": "DEV",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    env.pop("ENGRAM_DOTENV_CANARY", None)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, str(path), str(context.id),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path, env=env,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 1, stderr.decode()
    result = json.loads(stdout)
    assert result["failed"] == 1 and result["errors"][0]["reason"] == "provider_error"
    assert b"fake-provider-key-canary" not in stdout + stderr
    assert b"Provider List" not in stdout + stderr
