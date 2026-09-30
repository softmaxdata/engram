"""Persisted-state regressions for semantic ingestion and delta rollback."""

import json
from types import SimpleNamespace

import pytest

from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.ingestion import CuratorEngine, IngestionEngine
from engram.core.models import (
    Bullet,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    IntentAnchor,
    RecordDecisionRequest,
    Reflection,
    ReflectionInsight,
)
from engram.llm.adapter import LLMAdapter
from engram.server.routes.concepts import record_decision
from engram.storage.sqlite import SQLiteBackend


class SemanticLLM(LLMAdapter):
    def __init__(self):
        self.reflection = Reflection()
        self.vectors = {}
        self.fail_embeddings = False

    async def complete(self, *args, **kwargs):
        return self.reflection.model_dump_json()

    async def embed(self, text):
        if self.fail_embeddings:
            raise RuntimeError("Embedding provider unavailable")
        return self.vectors.get(text, [1.0, 0.0])


@pytest.fixture
async def state(tmp_path):
    storage = SQLiteBackend(str(tmp_path / "core.db"))
    await storage.initialize()
    context = Context(name="regression", intent=IntentAnchor(objective="test"))
    await storage.create_context(context)
    llm = SemanticLLM()
    engine = IngestionEngine(
        storage, llm, ingestion_config=IngestionConfig(max_reflection_rounds=1)
    )
    yield storage, context, llm, engine
    await storage.close()


async def test_commit_semantic_search_and_paraphrase_dedup(state):
    storage, context, llm, engine = state
    short = "Postgres supports our application"
    rich = "PostgreSQL supports our application with indexed vector search"
    llm.vectors = {short: [1.0, 0.0], rich: [0.98, 0.2]}
    llm.reflection = Reflection(new_insights=[ReflectionInsight(content=short)])
    await engine.commit(context.id, "agent", "first raw input")
    hits = await storage.find_similar_bullets(str(context.id), [1.0, 0.0], threshold=0.9)
    assert len(hits) == 1
    original_id = hits[0][0].id

    llm.reflection = Reflection(new_insights=[ReflectionInsight(content=rich)])
    batch = await engine.commit(context.id, "agent", "second raw input")
    bullets = await storage.list_bullets(str(context.id))
    assert len(bullets) == 1
    assert bullets[0].id == original_id
    assert bullets[0].content == rich
    assert bullets[0].embedding == llm.vectors[rich]
    # Stored deltas survive JSON serialization and retain the matching vector.
    saved = await storage.get_delta_batch(batch.id)
    assert json.loads(saved.model_dump_json())["operations"][0]["embedding"] == llm.vectors[rich]


async def test_shorter_paraphrase_keeps_survivor_content_and_vector(state):
    storage, context, llm, engine = state
    long_text = "Postgres is fast for indexed vector search"
    await storage.add_bullet(
        str(context.id), Bullet(id="survivor", content=long_text, embedding=[1.0, 0.0])
    )
    llm.vectors["Postgres is fast"] = [0.98, 0.2]
    llm.reflection = Reflection(new_insights=[ReflectionInsight(content="Postgres is fast")])
    await engine.commit(context.id, "agent", "shorter paraphrase")
    bullet = await storage.get_bullet("survivor")
    assert bullet.content == long_text
    assert bullet.embedding == [1.0, 0.0]


async def test_auxiliary_memories_are_searchable(state):
    storage, context, llm, engine = state
    llm.reflection = Reflection(
        strategies_that_worked=["Benchmark before choosing"],
        failure_modes=["Requests timed out without retry"],
        prediction_errors=["Throughput was lower than expected"],
    )
    await engine.commit(context.id, "agent", "auxiliary reflection")
    bullets = await storage.list_bullets(str(context.id))
    assert len(bullets) == 3
    assert all(b.embedding == [1.0, 0.0] for b in bullets)


async def test_re_extraction_add_and_update_vectors(state):
    storage, context, llm, _ = state
    old = Bullet(id="old", content="Old wording", embedding=[1.0, 0.0])
    await storage.add_bullet(str(context.id), old)
    llm.vectors = {"Improved wording": [0.96, 0.28], "Independent fact": [0.0, 1.0]}
    reflection = Reflection(
        new_insights=[
            ReflectionInsight(content="Improved wording"),
            ReflectionInsight(content="Independent fact"),
        ]
    )
    batch = await CuratorEngine(storage, llm).curate_re_extraction(
        reflection, [old], str(context.id)
    )
    await DeltaEngine(storage).apply_batch(batch)
    bullets = await storage.list_bullets(str(context.id))
    assert len(bullets) == 2
    assert {b.content: b.embedding for b in bullets} == llm.vectors


async def test_decisions_and_direct_add_persist_vectors_in_delta(state):
    storage, context, llm, engine = state
    _, batch = await engine.record_decision(context.id, "Use Postgres", "Reliable", ["SQLite"], "a")
    direct, direct_batch = await engine.add_bullet_directly(str(context.id), "Manual fact")
    assert direct.embedding == [1.0, 0.0]
    assert len(await storage.find_similar_bullets(str(context.id), [1.0, 0.0])) == 3
    for item in (batch, direct_batch):
        saved = await storage.get_delta_batch(item.id)
        assert all(op.embedding == [1.0, 0.0] for op in saved.operations)


async def test_decision_route_returns_the_persisted_bullet_id(state):
    storage, context, _, engine = state
    request = SimpleNamespace(
        state=SimpleNamespace(user_id=context.owner),
        app=SimpleNamespace(state=SimpleNamespace(storage=storage, ingestion=engine)),
    )
    response = await record_decision(
        context.id,
        RecordDecisionRequest(decision="Use Postgres", rationale="Reliable", agent_id="test"),
        request,
    )
    data = response.model_dump() if hasattr(response, "model_dump") else response
    bullet = await storage.get_bullet(data["concept_id"])
    assert bullet is not None
    assert bullet.content == "Use Postgres. Rationale: Reliable"
    assert bullet.context_id == str(context.id)


async def test_embedding_failure_keeps_commit_available(state):
    storage, context, llm, engine = state
    llm.fail_embeddings = True
    llm.reflection = Reflection(new_insights=[ReflectionInsight(content="A durable fact")])
    await engine.commit(context.id, "agent", "content")
    bullets = await storage.list_bullets(str(context.id))
    assert len(bullets) == 1
    assert bullets[0].embedding is None


@pytest.mark.parametrize("embedding", [[0.0, 1.0], None])
async def test_update_vector_and_rollback_match_content(state, embedding):
    storage, context, _, _ = state
    await storage.add_bullet(
        str(context.id), Bullet(id="updatable", content="Original", embedding=[1.0, 0.0])
    )
    engine = DeltaEngine(storage)
    batch = DeltaBatch(
        context_id=str(context.id),
        operations=[
            DeltaOperation(
                op_type=DeltaOpType.UPDATE_BULLET,
                target_id="updatable",
                content="Changed",
                embedding=embedding,
            )
        ],
    )
    await engine.apply_batch(batch)
    updated = await storage.get_bullet("updatable")
    assert updated.content == "Changed"
    assert updated.embedding == embedding  # Failed embedding must clear an obsolete vector.
    await engine.rollback_batch(batch.id)
    restored = await storage.get_bullet("updatable")
    assert restored.content == "Original"
    assert restored.embedding == [1.0, 0.0]


async def test_metadata_update_retains_vector_and_legacy_delta_loads(state):
    storage, context, _, _ = state
    await storage.add_bullet(
        str(context.id), Bullet(id="metadata", content="Original", embedding=[1.0, 0.0])
    )
    legacy_op = DeltaOperation.model_validate(
        {"op_type": "update_bullet", "target_id": "metadata", "section": "new-section"}
    )
    await DeltaEngine(storage).apply_batch(
        DeltaBatch(context_id=str(context.id), operations=[legacy_op])
    )
    updated = await storage.get_bullet("metadata")
    assert updated.section == "new-section"
    assert updated.embedding == [1.0, 0.0]


async def test_legacy_rollback_does_not_keep_new_content_vector(state):
    storage, context, _, _ = state
    await storage.add_bullet(
        str(context.id), Bullet(id="legacy", content="New content", embedding=[0.0, 1.0])
    )
    legacy = DeltaBatch.model_validate(
        {
            "context_id": str(context.id),
            "operations": [
                {
                    "op_type": "update_bullet",
                    "target_id": "legacy",
                    "content": "New content",
                    "previous_state": {"content": "Original content"},
                }
            ],
        }
    )
    await storage.save_delta_batch(legacy)
    await DeltaEngine(storage).rollback_batch(legacy.id)
    restored = await storage.get_bullet("legacy")
    assert restored.content == "Original content"
    assert restored.embedding is None


@pytest.mark.parametrize("target_id", ["explicit", None])
async def test_add_rollback_without_snapshot_in_mixed_batch(state, target_id):
    storage, context, _, _ = state
    await storage.add_bullet(str(context.id), Bullet(id="existing", content="Original"))
    batch = DeltaBatch(
        context_id=str(context.id),
        operations=[
            DeltaOperation(
                op_type=DeltaOpType.UPDATE_BULLET, target_id="existing", content="Changed"
            ),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id=target_id, content="New"),
        ],
    )
    engine = DeltaEngine(storage)
    await engine.apply_batch(batch)
    saved = await storage.get_delta_batch(batch.id)
    assert saved.operations[1].target_id is not None
    assert await engine.rollback_batch(batch.id)
    active = await storage.list_bullets(str(context.id))
    assert [(b.id, b.content) for b in active] == [("existing", "Original")]
