"""Tests for Phase 4: Mem-α-inspired validity gate in the Curator.

Coverage:
1. Default off — gate doesn't run, all candidates pass through.
2. Enabled — judge rejects whitespace/trivial/malformed candidates.
3. Multiple rejections in one batch are all dropped.
4. Failed judge call falls back to keeping all ops (fail-open).
5. Only ADD_BULLET ops are subject to the gate; UPDATE/MERGE/etc. pass through.
"""

from __future__ import annotations

import json

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import CuratorEngine, IngestionEngine, ReflectorEngine
from engram.core.models import (
    ContentType,
    Context,
    DeltaOpType,
    IntentAnchor,
    Reflection,
    ReflectionInsight,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class SequencedLLM(LLMAdapter):
    """LLM that returns scripted responses in sequence — one for the Reflector,
    one for the Validity Gate. Used to keep tests deterministic."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        if self.calls >= len(self.responses):
            raise RuntimeError(f"Unexpected extra LLM call #{self.calls}")
        r = self.responses[self.calls]
        self.calls += 1
        self.last_model = model
        return r

    async def embed(self, text):
        vec = [0.0] * 16
        for w in text.lower().split():
            vec[hash(w) % 16] += 1.0
        import math
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


@pytest.fixture
async def storage(tmp_path):
    b = SQLiteBackend(db_path=str(tmp_path / "vg.db"))
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def context_id(storage):
    ctx = Context(name="vg", intent=IntentAnchor(objective="x"))
    await storage.create_context(ctx)
    return ctx.id


def _reflection_response(insights: list[dict]) -> str:
    return json.dumps({
        "new_insights": insights,
        "strategies_that_worked": [],
        "failure_modes": [],
        "prediction_errors": [],
        "open_questions": [],
        "confidence": 0.7,
    })


# ── 1. Default OFF: gate doesn't run, no extra LLM call ──────────────────────

async def test_validity_gate_off_by_default(storage, context_id):
    insights = [
        {"content": "valid insight 1", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
        {"content": "valid insight 2", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
    ]
    llm = SequencedLLM([_reflection_response(insights)])
    cfg = IngestionConfig()  # gate disabled by default
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    batch = await engine.commit(
        context_id=context_id, agent_id="t",
        content="raw input",
        content_type=ContentType.CONVERSATION,
    )
    # 2 ADD_BULLET ops should be present.
    add_ops = [op for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert len(add_ops) == 2
    # Only the Reflector call happened — no second judge call.
    assert llm.calls == 1


# ── 2. Enabled: judge rejects malformed candidates ───────────────────────────

async def test_validity_gate_drops_rejected_candidates(storage, context_id):
    insights = [
        {"content": "valid insight", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
        {"content": "ok thanks!", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
    ]
    verdicts = json.dumps({"verdicts": [
        {"idx": 0, "keep": True},
        {"idx": 1, "keep": False, "reason": "filler"},
    ]})
    llm = SequencedLLM([_reflection_response(insights), verdicts])
    cfg = IngestionConfig(enable_validity_gate=True)
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    batch = await engine.commit(
        context_id=context_id, agent_id="t", content="raw",
    )
    add_ops = [op for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert len(add_ops) == 1
    assert add_ops[0].content == "valid insight"


# ── 3. Multiple rejections in one batch ──────────────────────────────────────

async def test_validity_gate_drops_multiple(storage, context_id):
    insights = [
        {"content": f"insight {i}", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5}
        for i in range(4)
    ]
    verdicts = json.dumps({"verdicts": [
        {"idx": 0, "keep": True},
        {"idx": 1, "keep": False, "reason": "trivial"},
        {"idx": 2, "keep": False, "reason": "malformed"},
        {"idx": 3, "keep": True},
    ]})
    llm = SequencedLLM([_reflection_response(insights), verdicts])
    cfg = IngestionConfig(enable_validity_gate=True)
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    batch = await engine.commit(
        context_id=context_id, agent_id="t", content="raw",
    )
    kept = [op.content for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert kept == ["insight 0", "insight 3"]


# ── 4. Judge call fails → fail-open ──────────────────────────────────────────

async def test_validity_gate_fails_open(storage, context_id):
    insights = [
        {"content": "valid insight 1", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
        {"content": "valid insight 2", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
    ]

    class FailingJudgeLLM(SequencedLLM):
        async def complete(self, prompt, system=None, **kw):
            if self.calls == 0:
                # First call: Reflector — return canned insights.
                self.calls += 1
                return _reflection_response(insights)
            # Second call: judge — fail.
            raise RuntimeError("judge unavailable")

    cfg = IngestionConfig(enable_validity_gate=True)
    llm = FailingJudgeLLM([])
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    batch = await engine.commit(
        context_id=context_id, agent_id="t", content="raw",
    )
    # Both insights kept despite judge failure.
    kept = [op.content for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET]
    assert set(kept) == {"valid insight 1", "valid insight 2"}


# ── 4b. Judge call routes via the configured validity_gate_model ─────────────

async def test_validity_gate_passes_model_override(storage, context_id):
    """The judge call must use validity_gate_model, not the canonical Reflector."""
    insights = [
        {"content": "keep me", "insight_type": "fact",
         "suggested_section": "x", "evidence": "", "novelty": 0.5},
    ]
    verdicts = json.dumps({"verdicts": [{"idx": 0, "keep": True}]})

    class ModelTrackingLLM(SequencedLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.models_seen: list[str | None] = []

        async def complete(self, prompt, system=None, temperature=0.0,
                           max_tokens=4096, response_format=None, model=None):
            self.models_seen.append(model)
            return await super().complete(
                prompt, system, temperature, max_tokens, response_format, model,
            )

    llm = ModelTrackingLLM([_reflection_response(insights), verdicts])
    cfg = IngestionConfig(
        enable_validity_gate=True,
        validity_gate_model="judge-model-xyz",
    )
    engine = IngestionEngine(storage, llm, ingestion_config=cfg)
    await engine.commit(context_id=context_id, agent_id="t", content="raw")

    # First call is the Reflector (no override → None); second is the judge.
    assert llm.models_seen == [None, "judge-model-xyz"]


# ── 5. Non-ADD ops pass through unchanged ────────────────────────────────────

async def test_validity_gate_only_filters_add_ops(storage):
    # Build a fake DeltaOperation list with mixed types and verify _filter_invalid_ops
    # leaves non-ADD ops alone.
    from engram.core.models import DeltaOperation, DeltaSource

    ops = [
        DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="a", content="keep me",
                       source=DeltaSource.CURATOR),
        DeltaOperation(op_type=DeltaOpType.UPDATE_BULLET, target_id="b", content="updated",
                       source=DeltaSource.CURATOR),
        DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="c", content="drop me",
                       source=DeltaSource.CURATOR),
    ]
    verdicts = json.dumps({"verdicts": [
        {"idx": 0, "keep": True},
        {"idx": 1, "keep": False, "reason": "no"},
    ]})

    class JudgeLLM(SequencedLLM):
        pass

    llm = JudgeLLM([verdicts])
    cfg = IngestionConfig(enable_validity_gate=True)
    curator = CuratorEngine(storage, llm, ingestion_config=cfg)
    filtered = await curator._filter_invalid_ops(ops)
    types = [op.op_type for op in filtered]
    # UPDATE_BULLET remains (not subject to gate), and exactly one ADD_BULLET dropped.
    assert types.count(DeltaOpType.UPDATE_BULLET) == 1
    assert types.count(DeltaOpType.ADD_BULLET) == 1
    assert [op.content for op in filtered if op.op_type == DeltaOpType.ADD_BULLET] == ["keep me"]
