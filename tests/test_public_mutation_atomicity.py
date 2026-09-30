"""All public ingestion writes share atomic metadata and capacity boundaries."""

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.delta import DeltaEngine
from engram.core.exceptions import CapacityExceededError
from engram.core.ingestion import CuratorEngine, IngestionEngine
from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    ConceptNode,
    ConceptType,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    ReExtractionRequest,
    Reflection,
    ReflectionInsight,
)
from engram.core.re_extraction import ReExtractionEngine

storage = transaction_storage
VECTOR = [1.0] + [0.0] * 1535


async def _engine(backend, context_id):
    llm = AsyncMock()

    async def embed(text):
        # Parent-handle reads are rejected inside the transaction, making this
        # assert that providers are called before acquiring the scoped handle.
        await backend.list_bullets(context_id)
        return VECTOR

    llm.embed.side_effect = embed
    return IngestionEngine(backend, llm)


async def _reextract(backend, context_id, count=1):
    await backend.add_activity(
        uuid.UUID(context_id),
        Activity(
            agent_id="old",
            action_type=ActionType.FACT_LEARNED,
            summary="original",
            raw_input="original raw",
        ),
    )
    reflector = AsyncMock()
    reflector.reflect.return_value = Reflection()
    curator = AsyncMock()
    curator.curate_re_extraction.return_value = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, content=f"new {i}") for i in range(count)
        ],
    )
    return ReExtractionEngine(reflector, curator, backend)


@pytest.mark.parametrize("entrypoint", ["decision", "reextract"])
async def test_activity_failure_rolls_back_every_public_entrypoint(
    storage, monkeypatch, entrypoint
):
    backend, context_id = storage
    engine = (
        await _engine(backend, context_id)
        if entrypoint == "decision"
        else await _reextract(backend, context_id)
    )

    async def fail(self, *args):
        raise RuntimeError("activity unavailable")

    monkeypatch.setattr(type(backend), "add_activity", fail)
    with pytest.raises(RuntimeError, match="activity unavailable"):
        if entrypoint == "decision":
            await engine.record_decision(
                uuid.UUID(context_id), "decision", "reason", ["alternative"], "agent"
            )
        else:
            await engine.re_extract(
                context_id, ReExtractionRequest(reflector_model="test", dry_run=False)
            )
    assert await backend.list_bullets(context_id) == []
    assert await backend.list_delta_batches(context_id) == []
    activities = await backend.get_activities_with_raw_input(context_id)
    assert len(activities) == (1 if entrypoint == "reextract" else 0)


async def test_direct_add_salience_failure_rolls_back_bullet_and_audit(storage, monkeypatch):
    backend, context_id = storage
    engine = await _engine(backend, context_id)

    async def fail(self, bullet):
        raise RuntimeError("salience unavailable")

    monkeypatch.setattr(type(backend), "update_bullet", fail)
    with pytest.raises(RuntimeError, match="salience unavailable"):
        await engine.add_bullet_directly(context_id, "fact", salience=0.9)
    assert await backend.list_bullets(context_id) == []
    assert await backend.list_delta_batches(context_id) == []


async def test_concurrent_direct_add_checks_capacity_after_embedding(storage):
    backend, context_id = storage
    context = await backend.get_context(uuid.UUID(context_id))
    context.lifecycle_config.max_active_bullets = 1
    await backend.update_context(context)
    arrived = 0
    both = asyncio.Event()

    async def embed(text):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5)
        return VECTOR

    llm = AsyncMock()
    llm.embed.side_effect = embed
    engine = IngestionEngine(backend, llm)
    results = await asyncio.gather(
        *[engine.add_bullet_directly(context_id, text, salience=0.8) for text in ("one", "two")],
        return_exceptions=True,
    )
    assert sum(isinstance(result, CapacityExceededError) for result in results) == 1
    assert sum(isinstance(result, tuple) for result in results) == 1
    assert len(await backend.list_bullets(context_id)) == 1
    assert len(await backend.list_delta_batches(context_id)) == 1


@pytest.mark.parametrize("entrypoint", ["decision", "reextract"])
async def test_other_public_entrypoints_reject_capacity_overflow(storage, entrypoint):
    backend, context_id = storage
    context = await backend.get_context(uuid.UUID(context_id))
    context.lifecycle_config.max_active_bullets = 1
    await backend.update_context(context)
    engine = (
        await _engine(backend, context_id)
        if entrypoint == "decision"
        else await _reextract(backend, context_id, count=2)
    )
    with pytest.raises(CapacityExceededError):
        if entrypoint == "decision":
            await engine.record_decision(
                uuid.UUID(context_id), "decision", "reason", ["alternative"], "agent"
            )
        else:
            await engine.re_extract(
                context_id, ReExtractionRequest(reflector_model="test", dry_run=False)
            )
    assert await backend.list_bullets(context_id) == []
    assert await backend.list_delta_batches(context_id) == []


async def test_snapshotless_historical_merge_refuses_entire_rollback(storage):
    backend, context_id = storage
    await backend.add_bullet(context_id, Bullet(id="keep", content="keep"))
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=DeltaOpType.MERGE_BULLETS, target_ids=["lost", "survivor"]),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="keep", content="keep"),
        ],
    )
    await backend.save_delta_batch(batch)
    assert not await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("keep")).is_active


@pytest.mark.parametrize("kind", ["bullet", "concept"])
async def test_zero_vector_does_not_hide_valid_similarity_at_limit_one(storage, kind):
    backend, context_id = storage
    if kind == "bullet":
        await backend.add_bullet(
            context_id, Bullet(id="invalid", content="invalid", embedding=[0.0] * 1536)
        )
        await backend.add_bullet(context_id, Bullet(id="valid", content="valid", embedding=VECTOR))
        matches = await backend.find_similar_bullets(context_id, VECTOR, limit=1, threshold=0.9)
    else:
        await backend.add_concept(
            uuid.UUID(context_id),
            ConceptNode(type=ConceptType.FACT, content="invalid", embedding=[0.0] * 1536),
        )
        await backend.add_concept(
            uuid.UUID(context_id),
            ConceptNode(type=ConceptType.FACT, content="valid", embedding=VECTOR),
        )
        matches = await backend.find_similar_concepts(
            uuid.UUID(context_id), VECTOR, limit=1, threshold=0.9
        )
    assert len(matches) == 1
    assert matches[0][0].content == "valid"
    assert matches[0][1] == pytest.approx(1.0)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
async def test_curator_rejects_nonfinite_custom_storage_scores(storage, monkeypatch, score):
    backend, context_id = storage
    existing = await backend.add_bullet(
        context_id, Bullet(id="keep", content="unrelated old", embedding=VECTOR)
    )

    async def invalid(*args, **kwargs):
        return [(existing, score)]

    monkeypatch.setattr(type(backend), "find_similar_bullets", invalid)
    llm = AsyncMock()
    llm.embed.return_value = VECTOR
    curator = CuratorEngine(backend, llm)
    op = await curator._process_insight(
        context_id,
        ReflectionInsight(content="different new fact", novelty=0.9),
        [existing],
        "agent",
        None,
    )
    assert op.op_type == DeltaOpType.ADD_BULLET
