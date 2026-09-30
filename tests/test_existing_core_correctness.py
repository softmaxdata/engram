"""Existing mutation paths preserve context boundaries and historical state."""

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.consolidation import ConsolidationEngine
from engram.core.delta import DeltaEngine
from engram.core.models import (
    Bullet,
    ConsolidationConfig,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    IntentAnchor,
    SchemaNode,
)

storage = transaction_storage
VECTOR = [1.0] + [0.0] * 1535


@pytest.mark.parametrize(
    "kind", ["update_bullet", "remove_bullet", "merge_bullets", "update_schema"]
)
async def test_foreign_delta_target_rejects_whole_batch(storage, kind):
    backend, context_id = storage
    foreign = await backend.create_context(
        Context(name="foreign", intent=IntentAnchor(objective="private"))
    )
    await backend.add_bullet(str(foreign.id), Bullet(id="foreign", content="private"))
    await backend.add_schema(
        str(foreign.id), SchemaNode(id="foreign-schema", name="private", description="private")
    )
    await backend.add_bullet(context_id, Bullet(id="local", content="local"))
    op = DeltaOperation(op_type=DeltaOpType(kind), target_id="foreign", content="changed")
    if kind == "merge_bullets":
        op.target_ids = ["local", "foreign"]
    elif kind == "update_schema":
        op.target_id = "foreign-schema"
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(
                op_type=DeltaOpType.ADD_BULLET, target_id="before-invalid", content="must roll back"
            ),
            op,
        ],
    )
    with pytest.raises(ValueError, match="context"):
        await DeltaEngine(backend).apply_batch(batch)
    assert await backend.get_bullet("before-invalid") is None
    assert (await backend.get_bullet("foreign")).content == "private"
    assert (await backend.get_bullet("foreign")).is_active
    assert (await backend.get_schema("foreign-schema")).description == "private"
    assert await backend.get_delta_batch(batch.id) is None


async def test_remove_rollback_preserves_already_inactive_bullet(storage):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="inactive", content="old", is_active=False))
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.REMOVE_BULLET, target_id="inactive"),
            ],
        )
    )
    assert await engine.rollback_batch(batch.id)
    assert not (await backend.get_bullet("inactive")).is_active


async def test_merge_rollback_restores_content_vectors_and_usage(storage):
    backend, context_id = storage
    before = [
        Bullet(
            id="first", content="first", embedding=VECTOR, salience=0.8, recall_count=4, hit_count=2
        ),
        Bullet(
            id="second",
            content="second",
            embedding=VECTOR,
            salience=0.4,
            recall_count=3,
            miss_count=2,
        ),
    ]
    for bullet in before:
        await backend.add_bullet(context_id, bullet)
    before = [await backend.get_bullet(bullet.id) for bullet in before]
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.MERGE_BULLETS,
                    target_ids=["first", "second"],
                    content="combined",
                ),
            ],
        )
    )
    assert len(await backend.list_bullets(context_id)) == 1
    assert await engine.rollback_batch(batch.id)
    for expected in before:
        loaded = await backend.get_bullet(expected.id)
        assert loaded.is_active
        for field in (
            "content",
            "embedding",
            "salience",
            "recall_count",
            "hit_count",
            "miss_count",
        ):
            assert getattr(loaded, field) == getattr(expected, field)


async def test_duplicate_merge_ids_do_not_double_usage(storage):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="one", content="one", recall_count=2))
    await backend.add_bullet(context_id, Bullet(id="two", content="two", recall_count=3))
    await DeltaEngine(backend).apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.MERGE_BULLETS, target_ids=["one", "one", "two"]),
            ],
        )
    )
    assert sum(b.recall_count for b in await backend.list_bullets(context_id)) == 5


async def test_legacy_rollback_cannot_mutate_foreign_context(storage):
    backend, context_id = storage
    foreign = await backend.create_context(
        Context(name="foreign", intent=IntentAnchor(objective="private"))
    )
    await backend.add_bullet(str(foreign.id), Bullet(id="foreign", content="private"))
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(
                op_type=DeltaOpType.UPDATE_BULLET,
                target_id="foreign",
                previous_state={"content": "injected"},
            ),
        ],
    )
    await backend.save_delta_batch(batch)
    with pytest.raises(ValueError, match="context"):
        await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("foreign")).content == "private"


async def test_successful_apply_preserves_caller_batch_and_operation_identity(storage):
    backend, context_id = storage
    op = DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="legacy caller")
    batch = DeltaBatch(context_id=context_id, operations=[op])
    result = await DeltaEngine(backend).apply_batch(batch)
    assert result is batch
    assert batch.operations[0] is op
    assert op.target_id is not None
    assert batch.bullets_added == 1
    assert (await backend.get_bullet(op.target_id)).content == "legacy caller"


async def test_dedup_failure_does_not_duplicate_statistics(storage, monkeypatch):
    backend, context_id = storage
    await backend.add_bullet(
        context_id,
        Bullet(id="first", content="first", embedding=VECTOR, salience=0.8, recall_count=4),
    )
    await backend.add_bullet(
        context_id,
        Bullet(id="second", content="second", embedding=VECTOR, salience=0.4, recall_count=3),
    )

    async def fail(self, bullet_id):
        raise RuntimeError("donor removal unavailable")

    monkeypatch.setattr(type(backend), "remove_bullet", fail)
    with pytest.raises(RuntimeError, match="donor removal"):
        await ConsolidationEngine(backend, None)._semantic_dedup(context_id, ConsolidationConfig())
    assert [(b.id, b.recall_count) for b in await backend.list_bullets(context_id)] == [
        ("first", 4),
        ("second", 3),
    ]
    assert await backend.list_delta_batches(context_id) == []


async def test_dedup_persists_reversible_audit(storage):
    backend, context_id = storage
    for i, salience in enumerate((0.8, 0.6, 0.4)):
        await backend.add_bullet(
            context_id,
            Bullet(
                id=str(i), content=str(i), embedding=VECTOR, salience=salience, recall_count=i + 1
            ),
        )
    count = await ConsolidationEngine(backend, None)._semantic_dedup(
        context_id, ConsolidationConfig()
    )
    assert count == 2
    assert [(b.id, b.recall_count) for b in await backend.list_bullets(context_id)] == [("0", 6)]
    batches = await backend.list_delta_batches(context_id)
    assert len(batches) == 1
    assert await DeltaEngine(backend).rollback_batch(batches[0].id)
    assert [(b.id, b.recall_count) for b in await backend.list_bullets(context_id)] == [
        ("0", 1),
        ("1", 2),
        ("2", 3),
    ]


async def test_existing_archive_lifecycle_survives_old_schema_upgrade(storage):
    backend, context_id = storage
    from engram.storage.sqlite import SQLiteBackend

    await backend.add_bullet(context_id, Bullet(id="archive", content="historical"))
    assert await backend.archive_bullet(context_id, "archive")
    # Simulate a populated pre-lifecycle database, retaining its old archive flag.
    if isinstance(backend, SQLiteBackend):
        db = await backend._get_db()
        await db.execute("DROP INDEX idx_bullets_lifecycle")
        await db.execute("ALTER TABLE bullets DROP COLUMN lifecycle_state")
        await db.execute("ALTER TABLE bullets DROP COLUMN archive_reason")
        await db.commit()
    else:
        pool = await backend._get_pool()
        await pool.execute("ALTER TABLE bullets DROP COLUMN lifecycle_state")
        await pool.execute("ALTER TABLE bullets DROP COLUMN archive_reason")
    await backend.initialize()
    archived = await backend.get_archived_bullets(context_id)
    assert [b.id for b in archived] == ["archive"]
    restored = await backend.restore_bullet(context_id, "archive")
    assert restored is not None
    assert restored.content == "historical"
    assert restored.is_active and not restored.is_archived
    # Idempotent initialization must not reclassify an existing lifecycle.
    await backend.initialize()
    assert (await backend.get_bullet("archive")).lifecycle_state.value == "active"


def _offline_ingestion(backend, reflect=None):
    from unittest.mock import AsyncMock

    from engram.core.ingestion import IngestionEngine
    from engram.core.models import Reflection

    engine = IngestionEngine(backend, AsyncMock())
    engine.llm.embed.return_value = VECTOR
    engine.reflector.reflect = AsyncMock(side_effect=reflect, return_value=Reflection())

    async def curate(context_id, **kwargs):
        return DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="extracted"),
            ],
        )

    engine.curator.curate = AsyncMock(side_effect=curate)
    return engine


async def test_failed_activity_write_does_not_orphan_ingestion_delta(storage, monkeypatch):
    import uuid

    backend, context_id = storage
    engine = _offline_ingestion(backend)
    original = type(backend).add_activity

    async def fail(self, *args):
        raise RuntimeError("activity ledger unavailable")

    monkeypatch.setattr(type(backend), "add_activity", fail)
    with pytest.raises(RuntimeError, match="activity ledger"):
        await engine.commit(uuid.UUID(context_id), "agent", "raw input")
    assert await backend.list_bullets(context_id) == []
    assert await backend.list_delta_batches(context_id) == []
    monkeypatch.setattr(type(backend), "add_activity", original)
    await engine.commit(uuid.UUID(context_id), "agent", "raw input")
    assert len(await backend.list_bullets(context_id)) == 1


async def test_concurrent_identical_commits_share_one_delta_and_activity(storage):
    import asyncio
    import uuid

    from engram.core.models import Reflection

    backend, context_id = storage
    arrived = 0
    both_reflecting = asyncio.Event()

    async def reflect(**kwargs):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_reflecting.set()
        await asyncio.wait_for(both_reflecting.wait(), 5)
        return Reflection()

    from unittest.mock import Mock

    engines = [_offline_ingestion(backend, reflect) for _ in range(2)]
    events = Mock()
    for engine in engines:
        engine.event_bus = events
    batches = await asyncio.gather(
        *[
            engine.commit(uuid.UUID(context_id), "agent", "identical raw input")
            for engine in engines
        ]
    )
    events.emit.assert_called_once()
    assert batches[0].id == batches[1].id
    assert len(await backend.list_bullets(context_id)) == 1
    assert len(await backend.get_activities_with_raw_input(context_id)) == 1


@pytest.mark.parametrize("pathological", [float("nan"), 1e308])
async def test_consolidation_does_not_merge_nonfinite_historical_similarity(
    storage, monkeypatch, pathological
):
    backend, context_id = storage
    for bullet_id in ("first", "second"):
        await backend.add_bullet(
            context_id, Bullet(id=bullet_id, content=bullet_id, embedding=VECTOR)
        )
    original = type(backend).list_bullets

    async def historical_vectors(self, *args, **kwargs):
        bullets = await original(self, *args, **kwargs)
        for bullet in bullets:
            # Historical/custom storage can contain values PostgreSQL now
            # rejects; an invalid similarity must never authorize deletion.
            bullet.embedding = [pathological] + [0.0] * 1535
        return bullets

    monkeypatch.setattr(type(backend), "list_bullets", historical_vectors)
    original_get = type(backend).get_bullet

    async def historical_vector(self, bullet_id):
        bullet = await original_get(self, bullet_id)
        if bullet is not None:
            bullet.embedding = [pathological] + [0.0] * 1535
        return bullet

    monkeypatch.setattr(type(backend), "get_bullet", historical_vector)
    assert (
        await ConsolidationEngine(backend, None)._semantic_dedup(context_id, ConsolidationConfig())
        == 0
    )
    assert (await backend.get_bullet("first")).is_active
    assert (await backend.get_bullet("second")).is_active
    assert await backend.list_delta_batches(context_id) == []


@pytest.mark.parametrize("missing_remove", [False, True])
async def test_commit_rejects_actual_capacity_overflow_atomically(storage, missing_remove):
    import uuid
    from unittest.mock import AsyncMock

    from engram.core.exceptions import CapacityExceededError

    backend, context_id = storage
    context = await backend.get_context(uuid.UUID(context_id))
    context.lifecycle_config.max_active_bullets = 1
    await backend.update_context(context)
    engine = _offline_ingestion(backend)
    operations = [
        DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id=name, content=name)
        for name in ("one", "two")
    ]
    if missing_remove:
        operations.append(DeltaOperation(op_type=DeltaOpType.REMOVE_BULLET, target_id="absent"))
    engine.curator.curate = AsyncMock(
        return_value=DeltaBatch(context_id=context_id, operations=operations)
    )
    with pytest.raises(CapacityExceededError):
        await engine.commit(uuid.UUID(context_id), "agent", "over capacity")
    assert await backend.list_bullets(context_id) == []
    assert await backend.list_delta_batches(context_id) == []
    assert await backend.get_activities_with_raw_input(context_id) == []


@pytest.mark.parametrize("mode", ["replace-full", "reduce-overfull", "edit-overfull"])
async def test_commit_accepts_capacity_preserving_or_reducing_changes(storage, mode):
    import uuid
    from unittest.mock import AsyncMock

    backend, context_id = storage
    context = await backend.get_context(uuid.UUID(context_id))
    context.lifecycle_config.max_active_bullets = 1
    await backend.update_context(context)
    await backend.add_bullet(context_id, Bullet(id="one", content="one"))
    if mode != "replace-full":
        await backend.add_bullet(context_id, Bullet(id="two", content="two"))
    operations = [DeltaOperation(op_type=DeltaOpType.REMOVE_BULLET, target_id="one")]
    if mode == "replace-full":
        operations.append(DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content="replacement"))
    elif mode == "edit-overfull":
        operations = [
            DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id="one", content="corrected")
        ]
    engine = _offline_ingestion(backend)
    engine.curator.curate = AsyncMock(
        return_value=DeltaBatch(context_id=context_id, operations=operations)
    )
    await engine.commit(uuid.UUID(context_id), "agent", "capacity preserving change")
    assert len(await backend.list_bullets(context_id)) == (2 if mode == "edit-overfull" else 1)
    assert len(await backend.get_activities_with_raw_input(context_id)) == 1
