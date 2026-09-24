"""Context boundaries must hold even when an item id belongs to another context."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from engram.core.delta import DeltaEngine
from engram.core.graph import ConceptGraph
from engram.core.ingestion import IngestionEngine
from engram.core.models import (
    Bullet,
    ConceptEdge,
    ConceptNode,
    ConceptType,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    EdgeType,
    ExecutionFeedback,
    FeedbackOutcome,
    IntentAnchor,
    InvalidateRequest,
    MaterializationRecord,
    SchemaNode,
)
from engram.server.routes.bullets import get_bullet
from engram.server.routes.contexts import invalidate_concepts
from engram.server.routes.deltas import get_delta, rollback_delta
from engram.server.routes.schemas import get_schema
from engram.storage.sqlite import SQLiteBackend


@pytest.fixture
async def contexts(tmp_path):
    storage = SQLiteBackend(str(tmp_path / "isolation.db"))
    await storage.initialize()
    a = Context(name="a", owner="owner", intent=IntentAnchor(objective="a"))
    b = Context(name="b", owner="owner", intent=IntentAnchor(objective="b"))
    await storage.create_context(a)
    await storage.create_context(b)
    request = SimpleNamespace(
        state=SimpleNamespace(user_id="owner"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                storage=storage,
                delta_engine=DeltaEngine(storage),
                graph=ConceptGraph(storage),
            )
        ),
    )
    yield storage, a, b, request
    await storage.close()


@pytest.mark.parametrize("kind", ["bullet", "schema", "delta"])
async def test_child_get_requires_matching_context(contexts, kind):
    storage, a, b, request = contexts
    if kind == "bullet":
        obj = await storage.add_bullet(str(a.id), Bullet(content="private"))
        getter = get_bullet
    elif kind == "schema":
        obj = await storage.add_schema(str(a.id), SchemaNode(name="private", description="pattern"))
        getter = get_schema
    else:
        obj = await storage.save_delta_batch(DeltaBatch(context_id=str(a.id)))
        getter = get_delta
    with pytest.raises(HTTPException) as exc:
        await getter(b.id, obj.id, request)
    assert exc.value.status_code == 404
    assert (await getter(a.id, obj.id, request))["id"] == obj.id


async def test_foreign_rollback_cannot_mutate(contexts):
    storage, a, b, request = contexts
    bullet = await storage.add_bullet(str(a.id), Bullet(content="before"))
    batch = DeltaBatch(
        context_id=str(a.id),
        operations=[
            DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id=bullet.id, content="after")
        ],
    )
    await request.app.state.delta_engine.apply_batch(batch)
    with pytest.raises(HTTPException) as exc:
        await rollback_delta(b.id, batch.id, request)
    assert exc.value.status_code == 404
    assert (await storage.get_bullet(bullet.id)).content == "after"
    await rollback_delta(a.id, batch.id, request)
    assert (await storage.get_bullet(bullet.id)).content == "before"


async def test_invalidation_ignores_foreign_concepts_and_bullets(contexts):
    storage, a, b, request = contexts
    own = await storage.add_concept(a.id, ConceptNode(type=ConceptType.FACT, content="own"))
    foreign = await storage.add_concept(b.id, ConceptNode(type=ConceptType.FACT, content="foreign"))
    own_bullet = await storage.add_bullet(str(a.id), Bullet(content="own"))
    foreign_bullet = await storage.add_bullet(str(b.id), Bullet(content="foreign"))
    await invalidate_concepts(
        a.id,
        InvalidateRequest(
            concept_ids=[own.id, foreign.id],
            bullet_ids=[own_bullet.id, foreign_bullet.id],
            reason="test",
        ),
        request,
    )
    assert not (await storage.get_concept(own.id)).is_valid
    assert (await storage.get_concept(foreign.id)).is_valid
    assert not (await storage.get_bullet(own_bullet.id)).is_active
    assert (await storage.get_bullet(foreign_bullet.id)).is_active


async def test_neighborhood_does_not_follow_foreign_nodes(contexts):
    storage, a, b, request = contexts
    own = await storage.add_concept(a.id, ConceptNode(type=ConceptType.FACT, content="own"))
    foreign = await storage.add_concept(b.id, ConceptNode(type=ConceptType.FACT, content="foreign"))
    await storage.add_edge(
        a.id, ConceptEdge(from_node=own.id, to_node=foreign.id, type=EdgeType.RELATED_TO)
    )
    graph = request.app.state.graph
    assert await graph.get_concept_neighborhood(a.id, foreign.id) == []
    assert [c.id for c in await graph.get_concept_neighborhood(a.id, own.id)] == [own.id]


async def test_feedback_ignores_foreign_materialization(contexts):
    storage, a, b, _ = contexts
    bullet = await storage.add_bullet(str(b.id), Bullet(content="foreign"))
    rec = MaterializationRecord(context_id=str(b.id), bullets_included=[bullet.id], token_count=1)
    await storage.save_materialization(rec)
    engine = IngestionEngine(storage, None)
    await engine._reconsolidate(
        rec.id, ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS), expected_context_id=str(a.id)
    )
    assert (await storage.get_bullet(bullet.id)).hit_count == 0
    await engine._reconsolidate(
        rec.id, ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS), expected_context_id=str(b.id)
    )
    assert (await storage.get_bullet(bullet.id)).hit_count == 1


async def test_feedback_checks_each_bullet_context(contexts):
    storage, a, b, _ = contexts
    own = await storage.add_bullet(str(a.id), Bullet(content="own"))
    foreign = await storage.add_bullet(str(b.id), Bullet(content="foreign"))
    rec = MaterializationRecord(
        context_id=str(a.id), bullets_included=[own.id, foreign.id], token_count=1
    )
    await storage.save_materialization(rec)
    await IngestionEngine(storage, None)._reconsolidate(
        rec.id, ExecutionFeedback(outcome=FeedbackOutcome.SUCCESS), expected_context_id=str(a.id)
    )
    assert (await storage.get_bullet(own.id)).hit_count == 1
    assert (await storage.get_bullet(foreign.id)).hit_count == 0
