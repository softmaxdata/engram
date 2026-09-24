"""Fault injection checks persisted atomicity, not just operation counters."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.delta import DeltaEngine
from engram.core.ingestion import IngestionEngine
from engram.core.models import (
    Bullet,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    ExecutionFeedback,
    FeedbackOutcome,
    MaterializationRecord,
)
from engram.storage.postgres import PostgresBackend
from engram.storage.sqlite import SQLiteBackend

storage = transaction_storage


@pytest.mark.parametrize("failure", ["second_write", "audit"])
async def test_failed_batch_leaves_no_mutations_or_audit(storage, monkeypatch, failure):
    backend, context_id = storage
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="a", content="first"),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="b", content="second"),
        ],
    )
    name = "add_bullet" if failure == "second_write" else "save_delta_batch"
    original = getattr(type(backend), name)

    async def fail(self, *args, **kwargs):
        if failure == "audit" or args[1].id == "b":
            raise RuntimeError("injected failure")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(type(backend), name, fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        await DeltaEngine(backend).apply_batch(batch)
    assert await backend.list_bullets(context_id) == []
    assert await backend.get_delta_batch(batch.id) is None
    assert batch.bullets_added == 0
    monkeypatch.setattr(type(backend), name, original)
    result = await DeltaEngine(backend).apply_batch(batch)
    assert result.bullets_added == 2


async def test_failed_retry_uses_immediate_pre_retry_snapshot(storage, monkeypatch):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="target", content="initial"))
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(
                op_type=DeltaOpType.UPDATE_BULLET, target_id="target", content="applied"
            ),
        ],
    )
    original = type(backend).save_delta_batch

    async def fail(self, batch):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(type(backend), "save_delta_batch", fail)
    with pytest.raises(RuntimeError):
        await DeltaEngine(backend).apply_batch(batch)
    assert batch.operations[0].previous_state is None
    if hasattr(batch.operations[0], "rollback_state"):
        assert batch.operations[0].rollback_state is None
    monkeypatch.setattr(type(backend), "save_delta_batch", original)
    independent = await backend.get_bullet("target")
    independent.content = "independent update"
    await backend.update_bullet(independent)
    await DeltaEngine(backend).apply_batch(batch)
    await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("target")).content == "independent update"


@pytest.mark.parametrize("failure", ["marker", "second_bullet"])
async def test_feedback_failure_rolls_back_stats_and_audit(storage, monkeypatch, failure):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="a", content="first"))
    await backend.add_bullet(context_id, Bullet(id="b", content="second"))
    record = MaterializationRecord(context_id=context_id, bullets_included=["a", "b"])
    await backend.save_materialization(record)
    name = "mark_materialization_reconsolidated" if failure == "marker" else "update_bullet"
    original = getattr(type(backend), name)

    async def fail(self, *args):
        if failure == "marker" or args[0].id == "b":
            raise RuntimeError("receipt unavailable")
        return await original(self, *args)

    monkeypatch.setattr(type(backend), name, fail)
    engine = IngestionEngine(backend, None)
    feedback = ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS)
    with pytest.raises(RuntimeError, match="receipt unavailable"):
        await engine._reconsolidate(record.id, feedback, expected_context_id=context_id)
    assert [(b.hit_count, b.recall_count) for b in await backend.list_bullets(context_id)] == [
        (0, 0)
    ] * 2
    assert (await backend.get_materialization(record.id)).reconsolidated_at is None
    assert await backend.list_delta_batches(context_id) == []
    monkeypatch.setattr(type(backend), name, original)
    await engine._reconsolidate(record.id, feedback, expected_context_id=context_id)
    await engine._reconsolidate(record.id, feedback, expected_context_id=context_id)
    assert [(b.hit_count, b.recall_count) for b in await backend.list_bullets(context_id)] == [
        (1, 1)
    ] * 2


async def test_failed_rollback_is_atomic(storage, monkeypatch):
    backend, context_id = storage
    for bid in ("a", "b"):
        await backend.add_bullet(context_id, Bullet(id=bid, content="before"))
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id=bid, content="after")
            for bid in ("a", "b")
        ],
    )
    await DeltaEngine(backend).apply_batch(batch)
    original = type(backend).update_bullet

    async def fail(self, bullet):
        if bullet.id == "a":
            raise RuntimeError("rollback interrupted")
        return await original(self, bullet)

    monkeypatch.setattr(type(backend), "update_bullet", fail)
    with pytest.raises(RuntimeError):
        await DeltaEngine(backend).rollback_batch(batch.id)
    assert {b.content for b in await backend.list_bullets(context_id)} == {"after"}


async def test_independent_engines_preserve_distinct_receipt_increments(storage):
    backend, context_id = storage
    if isinstance(backend, SQLiteBackend) and backend.db_path == ":memory:":
        pytest.skip("independent in-memory instances intentionally have separate databases")
    second = (
        SQLiteBackend(backend.db_path, existing_only=True)
        if isinstance(backend, SQLiteBackend)
        else PostgresBackend(backend.dsn, existing_only=True)
    )
    try:
        await backend.add_bullet(context_id, Bullet(id="shared", content="fact"))
        receipts = [
            MaterializationRecord(context_id=context_id, bullets_included=["shared"])
            for _ in range(2)
        ]
        for record in receipts:
            await backend.save_materialization(record)
        feedback = ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS)
        await asyncio.gather(
            IngestionEngine(backend, None)._reconsolidate(
                receipts[0].id, feedback, expected_context_id=context_id
            ),
            IngestionEngine(second, None)._reconsolidate(
                receipts[1].id, feedback, expected_context_id=context_id
            ),
        )
        assert (await backend.get_bullet("shared")).hit_count == 2
    finally:
        await second.close()


WORKER = r"""
import asyncio, sys
from engram.core.config import Settings
import engram.core.config as config
from engram.core.ingestion import IngestionEngine
from engram.core.models import ExecutionFeedback, FeedbackOutcome
from engram.storage.sqlite import SQLiteBackend
from engram.storage.postgres import PostgresBackend
config.get_settings = lambda: Settings(_env_file=None, auth_enabled=False)
async def main():
    backend = (SQLiteBackend(sys.argv[2], existing_only=True) if sys.argv[1] == "sqlite"
               else PostgresBackend(sys.argv[2], existing_only=True))
    try:
        print("ready", flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        await IngestionEngine(backend, None)._reconsolidate(
            sys.argv[3], ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
            expected_context_id=sys.argv[4])
    finally:
        await backend.close()
asyncio.run(main())
"""


async def test_two_processes_consume_one_receipt_once(storage):
    backend, context_id = storage
    if isinstance(backend, SQLiteBackend) and backend.db_path == ":memory:":
        pytest.skip("process race requires persistent shared storage")
    await backend.add_bullet(context_id, Bullet(id="shared", content="fact", salience=0.5))
    record = MaterializationRecord(context_id=context_id, bullets_included=["shared"])
    await backend.save_materialization(record)
    kind = "sqlite" if isinstance(backend, SQLiteBackend) else "postgres"
    target = backend.db_path if kind == "sqlite" else backend.dsn
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    children = []
    try:
        for _ in range(2):
            children.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    WORKER,
                    kind,
                    target,
                    record.id,
                    context_id,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                )
            )
        for child in children:
            assert await asyncio.wait_for(child.stdout.readline(), 10) == b"ready\n"
        for child in children:
            child.stdin.write(b"go\n")
            await child.stdin.drain()
        for child in children:
            _, stderr = await asyncio.wait_for(child.communicate(), 30)
            assert child.returncode == 0, stderr.decode()
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
                await child.wait()
    await backend.close()
    await backend.initialize()
    bullet = await backend.get_bullet("shared")
    assert (bullet.recall_count, bullet.hit_count, bullet.miss_count) == (1, 1, 0)
    assert bullet.salience == pytest.approx(0.525)
    assert (await backend.get_materialization(record.id)).reconsolidated_at is not None


async def test_schema_and_bullet_rollback_together_when_audit_fails(storage, monkeypatch):
    backend, context_id = storage
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=DeltaOpType.ADD_SCHEMA, target_id="schema", content="group"),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="fact", content="fact"),
        ],
    )

    async def fail(self, batch):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(type(backend), "save_delta_batch", fail)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        await DeltaEngine(backend).apply_batch(batch)
    assert await backend.list_schemas(context_id) == []
    assert await backend.list_bullets(context_id) == []
    assert await backend.get_delta_batch(batch.id) is None


async def test_public_commit_retry_finishes_feedback_without_reextracting(storage, monkeypatch):
    import uuid

    from engram.core.config import IngestionConfig
    from engram.core.models import Context, IntentAnchor, Reflection, ReflectionInsight

    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="recalled", content="Previously recalled fact"))
    receipt = MaterializationRecord(context_id=context_id, bullets_included=["recalled"])
    await backend.save_materialization(receipt)

    class CountingLLM:
        completions = 0

        async def complete(self, *args, **kwargs):
            self.completions += 1
            return Reflection(
                new_insights=[ReflectionInsight(content="Newly extracted fact")], confidence=0.99
            ).model_dump_json()

        async def embed(self, text):
            return [1.0] + [0.0] * 1535

    llm = CountingLLM()
    engine = IngestionEngine(
        backend, llm, ingestion_config=IngestionConfig(max_reflection_rounds=1)
    )
    original = type(backend).mark_materialization_reconsolidated

    async def fail(self, *args):
        raise RuntimeError("receipt unavailable")

    monkeypatch.setattr(type(backend), "mark_materialization_reconsolidated", fail)
    arguments = dict(
        context_id=uuid.UUID(context_id),
        agent_id="retry-agent",
        session_id=uuid.uuid4(),
        content="Identical request with feedback",
        materialization_id=receipt.id,
        feedback=ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
    )
    with pytest.raises(RuntimeError, match="receipt unavailable"):
        await engine.commit(**arguments)
    ingestion_batches = await backend.list_delta_batches(context_id)
    assert len(ingestion_batches) == 1
    assert (await backend.get_bullet("recalled")).hit_count == 0
    assert (await backend.get_materialization(receipt.id)).reconsolidated_at is None
    assert {bullet.content for bullet in await backend.list_bullets(context_id)} == {
        "Previously recalled fact",
        "Newly extracted fact",
    }

    monkeypatch.setattr(type(backend), "mark_materialization_reconsolidated", original)
    for _ in range(3):
        retried = await engine.commit(**arguments)
        assert retried.id == ingestion_batches[0].id
    assert llm.completions == 1
    recalled = await backend.get_bullet("recalled")
    assert (recalled.hit_count, recalled.recall_count) == (1, 1)
    assert (await backend.get_materialization(receipt.id)).reconsolidated_at is not None
    assert len(await backend.list_bullets(context_id)) == 2
    assert len(await backend.list_activities(uuid.UUID(context_id))) == 1
    batches = await backend.list_delta_batches(context_id)
    if hasattr(DeltaOpType, "RECONSOLIDATE_BULLET"):
        reconsolidations = [batch for batch in batches if batch.trigger == "reconsolidation"]
        assert len(reconsolidations) == 1
        assert len(batches) == 2
        for operation in reconsolidations[0].operations:
            assert operation.agent_id == arguments["agent_id"]
            assert operation.session_id == str(arguments["session_id"])
    else:
        assert len(batches) == 1

    foreign_context = await backend.create_context(
        Context(name="foreign", intent=IntentAnchor(objective="isolated"))
    )
    await backend.add_bullet(str(foreign_context.id), Bullet(id="foreign", content="Foreign fact"))
    foreign_receipt = MaterializationRecord(
        context_id=str(foreign_context.id), bullets_included=["foreign"]
    )
    await backend.save_materialization(foreign_receipt)
    arguments["materialization_id"] = foreign_receipt.id
    await engine.commit(**arguments)
    assert (await backend.get_materialization(foreign_receipt.id)).reconsolidated_at is None
    assert (await backend.get_bullet("foreign")).hit_count == 0
    assert llm.completions == 1


@pytest.mark.parametrize("operation", ["feedback", "batch"])
async def test_context_delete_waits_for_atomic_child_mutations(storage, monkeypatch, operation):
    import uuid

    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="child", content="fact"))
    receipt = MaterializationRecord(context_id=context_id, bullets_included=["child"])
    await backend.save_materialization(receipt)
    fetched = asyncio.Event()
    release = asyncio.Event()
    original = type(backend).get_bullet

    async def pause_after_child_lock(self, bullet_id):
        bullet = await original(self, bullet_id)
        if self._scope is not None and bullet_id == "child":
            fetched.set()
            await release.wait()
        return bullet

    monkeypatch.setattr(type(backend), "get_bullet", pause_after_child_lock)
    if operation == "feedback":
        mutation = IngestionEngine(backend, None)._reconsolidate(
            receipt.id,
            ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS),
            expected_context_id=context_id,
        )
    else:
        mutation = DeltaEngine(backend).apply_batch(
            DeltaBatch(
                context_id=context_id,
                operations=[
                    DeltaOperation(
                        op_type=DeltaOpType.UPDATE_BULLET,
                        target_id="child",
                        content="updated",
                    )
                ],
            )
        )
    writer = asyncio.create_task(mutation)
    deletion = None
    try:
        await asyncio.wait_for(fetched.wait(), timeout=5)
        deletion = asyncio.create_task(backend.delete_context(uuid.UUID(context_id)))
        if isinstance(backend, PostgresBackend):
            pool = await backend._get_pool()
            for _ in range(100):
                waiting = await pool.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() "
                    "AND query LIKE 'DELETE FROM contexts%' AND wait_event_type='Lock')"
                )
                if waiting:
                    break
                await asyncio.sleep(0.01)
            assert waiting, "delete did not reach its database lock"
        else:
            await asyncio.sleep(0.02)
        assert not deletion.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(writer, deletion), timeout=10)
        assert await backend.get_context(uuid.UUID(context_id)) is None
        assert await backend.list_bullets(context_id) == []
        assert await backend.list_delta_batches(context_id) == []
        assert await backend.get_materialization(receipt.id) is None
    finally:
        release.set()
        tasks = [task for task in (writer, deletion) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
