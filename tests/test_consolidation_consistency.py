"""Consolidation and historical rollback never invent state from stale evidence."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.consolidation import ConsolidationEngine
from engram.core.delta import DeltaEngine
from engram.core.ingestion import CuratorEngine
from engram.core.models import (
    Bullet,
    ConsolidationConfig,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    Reflection,
    ReflectionInsight,
)
from engram.storage.sqlite import SQLiteBackend

storage = transaction_storage
VECTOR = [1.0] + [0.0] * 1535


@pytest.mark.parametrize("embedding", [None, [float("nan"), 1.0], [0.0, 0.0]])
@pytest.mark.parametrize("same_text", [True, False])
async def test_reextract_preserves_identity_or_retains_uncertain_comparison(embedding, same_text):
    old = Bullet(
        id="old", content="Historical fact", embedding=embedding, salience=0.1, miss_count=5
    )
    llm = AsyncMock()
    llm.embed.return_value = [1.0, 0.0]
    result = await CuratorEngine(None, llm).curate_re_extraction(
        Reflection(
            new_insights=[
                ReflectionInsight(content=" historical FACT " if same_text else "new fact")
            ]
        ),
        [old],
        "context",
    )
    assert all(op.op_type != DeltaOpType.REMOVE_BULLET for op in result.operations)
    if same_text:
        assert result.operations == []


async def test_reextract_failed_provider_cannot_authorize_removal():
    old = Bullet(
        id="old", content="Historical fact", embedding=[1.0, 0.0], salience=0.1, miss_count=5
    )
    llm = AsyncMock()
    llm.embed.side_effect = RuntimeError("provider unavailable")
    result = await CuratorEngine(None, llm).curate_re_extraction(
        Reflection(new_insights=[ReflectionInsight(content="new fact")]),
        [old],
        "context",
    )
    assert all(op.op_type != DeltaOpType.REMOVE_BULLET for op in result.operations)


@pytest.mark.parametrize(
    "kind", [DeltaOpType.ADD_BULLET, DeltaOpType.UPDATE_BULLET, DeltaOpType.REMOVE_BULLET]
)
async def test_legacy_missing_identity_or_inverse_declines_whole_rollback(storage, kind):
    backend, context_id = storage
    for bid in ("unknown", "known"):
        await backend.add_bullet(context_id, Bullet(id=bid, content=bid))
    unknown = DeltaOperation(
        op_type=kind,
        target_id=None if kind == DeltaOpType.ADD_BULLET else "unknown",
        content="unknown",
    )
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            unknown,
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="known", content="known"),
        ],
    )
    await backend.save_delta_batch(batch)
    assert not await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("known")).is_active
    assert (await backend.get_bullet("unknown")).is_active


async def test_new_explicit_noops_remain_rollbackable(storage):
    backend, context_id = storage
    operations = [
        DeltaOperation(op_type=kind, target_id="missing")
        for kind in (
            DeltaOpType.UPDATE_BULLET,
            DeltaOpType.REMOVE_BULLET,
            DeltaOpType.UPDATE_SCHEMA,
        )
    ]
    operations.extend(
        [
            DeltaOperation(op_type=DeltaOpType.ADD_SCHEMA),
            DeltaOperation(
                op_type=DeltaOpType.MERGE_BULLETS, target_ids=["missing", "also-missing"]
            ),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="new", content="new"),
        ]
    )
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(DeltaBatch(context_id=context_id, operations=operations))
    assert await engine.rollback_batch(batch.id)
    assert not (await backend.get_bullet("new")).is_active


async def _facts(backend, context_id, count=3):
    for i in range(count):
        await backend.add_bullet(
            context_id,
            Bullet(
                id=f"fact{i}", content=f"Fact {i}", section="facts", embedding=VECTOR, hit_count=2
            ),
        )


async def test_schema_induction_preserves_changes_during_provider_call(storage):
    backend, context_id = storage
    await _facts(backend, context_id, 4)

    async def complete(**kwargs):
        await backend.archive_bullet(context_id, "fact0")
        current = await backend.get_bullet("fact1")
        current.content = "newer content"
        current.hit_count = 9
        await backend.update_bullet(current)
        return "Old pattern"

    llm = AsyncMock()
    llm.complete.side_effect = complete
    count = await ConsolidationEngine(backend, llm)._induce_schemas(
        context_id, ConsolidationConfig()
    )
    assert count == 0
    assert (await backend.get_bullet("fact0")).is_archived
    assert (await backend.get_bullet("fact1")).content == "newer content"
    assert (await backend.get_bullet("fact1")).hit_count == 9
    assert await backend.list_schemas(context_id) == []


async def test_schema_and_links_rollback_together(storage, monkeypatch):
    backend, context_id = storage
    await _facts(backend, context_id)
    llm = AsyncMock()
    llm.complete.return_value = "pattern"
    original = type(backend).update_bullet

    async def fail(self, bullet):
        if bullet.id == "fact0":
            raise RuntimeError("link unavailable")
        return await original(self, bullet)

    monkeypatch.setattr(type(backend), "update_bullet", fail)
    with pytest.raises(RuntimeError, match="link unavailable"):
        await ConsolidationEngine(backend, llm)._induce_schemas(context_id, ConsolidationConfig())
    assert await backend.list_schemas(context_id) == []
    assert all(b.schema_id is None for b in await backend.list_bullets(context_id))
    assert await backend.list_delta_batches(context_id) == []


async def test_concurrent_schema_induction_creates_one_schema(storage):
    import asyncio

    backend, context_id = storage
    await _facts(backend, context_id)
    arrived = 0
    ready = asyncio.Event()

    async def complete(**kwargs):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
        return "pattern"

    llm = AsyncMock()
    llm.complete.side_effect = complete
    engines = [ConsolidationEngine(backend, llm) for _ in range(2)]
    counts = await asyncio.gather(
        *(engine._induce_schemas(context_id, ConsolidationConfig()) for engine in engines)
    )
    assert sorted(counts) == [0, 1]
    assert len(await backend.list_schemas(context_id)) == 1


@pytest.mark.parametrize("phase", ["forgetting", "archive", "dedup"])
async def test_consolidation_rechecks_locked_current_rows(storage, monkeypatch, phase):
    backend, context_id = storage
    old = datetime.now(UTC) - timedelta(days=90)
    await backend.add_bullet(
        context_id,
        Bullet(
            id="high",
            content="old",
            embedding=VECTOR,
            salience=0.1 if phase == "archive" else 0.9,
            created_at=old,
        ),
    )
    await backend.add_bullet(
        context_id, Bullet(id="low", content="other", embedding=VECTOR, salience=0.5)
    )
    stale = await backend.list_bullets(context_id)
    if phase == "archive":
        current = await backend.get_bullet("high")
        current.salience = 0.95
        current.hit_count = 10
        current.last_recalled_at = datetime.now(UTC)
        await backend.update_bullet(current)
    else:
        await backend.archive_bullet(context_id, "high")
    original = type(backend).list_bullets
    first = True

    async def stale_once(self, *args, **kwargs):
        nonlocal first
        if first:
            first = False
            return [b.model_copy(deep=True) for b in stale]
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(type(backend), "list_bullets", stale_once)
    engine = ConsolidationEngine(backend, None)
    method = {
        "forgetting": engine._apply_forgetting_curve,
        "archive": engine._archive_stale,
        "dedup": engine._semantic_dedup,
    }[phase]
    await method(context_id, ConsolidationConfig())
    high = await backend.get_bullet("high")
    if phase == "archive":
        assert not high.is_archived
        assert high.hit_count == 10
        assert high.salience == pytest.approx(0.95)
    else:
        assert high.is_archived
    assert (await backend.get_bullet("low")).is_active


@pytest.mark.parametrize("change", ["archive", "capacity", "existing"])
async def test_promotion_rechecks_current_qualification_duplicates_and_capacity(storage, change):
    backend, context_id = storage
    await _facts(backend, context_id)
    if change == "capacity":
        context = await backend.get_context(uuid.UUID(context_id))
        context.lifecycle_config.max_active_bullets = 3
        await backend.update_context(context)

    async def complete(**kwargs):
        if change == "archive":
            await backend.archive_bullet(context_id, "fact0")
        elif change == "existing":
            await backend.add_bullet(
                context_id,
                Bullet(
                    id="existing-principle",
                    content="existing",
                    section="facts",
                    bullet_type="principle",
                ),
            )
        return "New principle"

    llm = AsyncMock()
    llm.complete.side_effect = complete
    assert (
        await ConsolidationEngine(backend, llm)._promote_to_principles(
            context_id, ConsolidationConfig()
        )
        == 0
    )
    principles = await backend.list_bullets(context_id, bullet_type="principle")
    assert len(principles) == (1 if change == "existing" else 0)


async def test_lifecycle_migration_backfill_failure_is_retry_safe(storage, monkeypatch):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="archived", content="preserve"))
    await backend.archive_bullet(context_id, "archived")
    if isinstance(backend, SQLiteBackend):
        import aiosqlite

        db = await backend._get_db()
        await db.execute("DROP INDEX idx_bullets_lifecycle")
        await db.execute("ALTER TABLE bullets DROP COLUMN lifecycle_state")
        await db.execute("ALTER TABLE bullets DROP COLUMN archive_reason")
        await db.commit()
        connection_type = aiosqlite.Connection
    else:
        import asyncpg

        pool = await backend._get_pool()
        await pool.execute("ALTER TABLE bullets DROP COLUMN lifecycle_state")
        await pool.execute("ALTER TABLE bullets DROP COLUMN archive_reason")
        connection_type = asyncpg.Connection
    original = connection_type.execute

    async def fail(self, query, *args, **kwargs):
        if "UPDATE bullets SET lifecycle_state='archived'" in query:
            raise RuntimeError("backfill interrupted")
        return await original(self, query, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(connection_type, "execute", fail)
        with pytest.raises(RuntimeError, match="backfill interrupted"):
            await backend.initialize()
    await backend.initialize()
    assert [b.id for b in await backend.get_archived_bullets(context_id)] == ["archived"]
    assert (await backend.restore_bullet(context_id, "archived")).content == "preserve"


async def test_explicit_zero_confidence_update_and_rollback(storage):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="zero", content="same", confidence=0.8))
    before = await backend.get_bullet("zero")
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id="zero", confidence=0.0),
            ],
        )
    )
    assert (await backend.get_bullet("zero")).confidence == 0.0
    assert await engine.rollback_batch(batch.id)
    assert (await backend.get_bullet("zero")).confidence == before.confidence


@pytest.mark.parametrize("invalid", [[0.0, 0.0], [float("nan"), 1.0], [1.79e308, 1.79e308], [1.0]])
async def test_destructive_comparison_rejects_unknown_vectors_at_zero_threshold(
    storage, monkeypatch, invalid
):
    backend, context_id = storage
    for bid in ("one", "two"):
        await backend.add_bullet(context_id, Bullet(id=bid, content=bid, embedding=VECTOR))
    original = type(backend).get_bullet

    async def malformed(self, bullet_id):
        bullet = await original(self, bullet_id)
        bullet.embedding = invalid if bullet_id == "one" else [1.0, 0.0]
        return bullet

    monkeypatch.setattr(type(backend), "get_bullet", malformed)
    assert (
        await ConsolidationEngine(backend, None)._semantic_dedup(
            context_id, ConsolidationConfig(dedup_threshold=0.0)
        )
        == 0
    )
    assert (await backend.get_bullet("one")).is_active
    assert (await backend.get_bullet("two")).is_active


@pytest.mark.parametrize("storage", ["postgres"], indirect=True)
async def test_postgres_archive_between_candidate_list_and_lock_preserves_active_donor(
    storage, monkeypatch
):
    from engram.storage.postgres import PostgresBackend

    backend, context_id = storage
    await backend.add_bullet(
        context_id, Bullet(id="high", content="high", embedding=VECTOR, salience=0.9)
    )
    await backend.add_bullet(
        context_id, Bullet(id="low", content="low", embedding=VECTOR, salience=0.5)
    )
    second = PostgresBackend(backend.dsn, existing_only=True)
    original = type(backend).list_bullets
    injected = False

    async def archive_between(self, *args, **kwargs):
        nonlocal injected
        bullets = await original(self, *args, **kwargs)
        if self._scope is not None and not injected:
            injected = True
            assert await second.archive_bullet(context_id, "high")
        return bullets

    monkeypatch.setattr(type(backend), "list_bullets", archive_between)
    try:
        assert (
            await ConsolidationEngine(backend, None)._semantic_dedup(
                context_id, ConsolidationConfig()
            )
            == 0
        )
        assert (await backend.get_bullet("high")).is_archived
        assert (await backend.get_bullet("low")).is_active
    finally:
        await second.close()


async def test_reused_noop_operation_captures_fresh_inverse_when_target_appears(storage):
    backend, context_id = storage
    engine = DeltaEngine(backend)
    op = DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id="later", content="changed")
    await engine.apply_batch(DeltaBatch(context_id=context_id, operations=[op]))
    await backend.add_bullet(context_id, Bullet(id="later", content="original"))
    changed = await engine.apply_batch(DeltaBatch(context_id=context_id, operations=[op]))
    assert (await backend.get_bullet("later")).content == "changed"
    assert await engine.rollback_batch(changed.id)
    assert (await backend.get_bullet("later")).content == "original"
