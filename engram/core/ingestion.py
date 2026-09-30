"""Ingestion pipeline — Reflector → Curator → Delta application.

v0.3: Adds concurrency control (lock during delta application),
capacity checks, delta revalidation, event emission, and
contradiction detection in the Curator.

v0.4: Canonical Reflector model via IngestionConfig, raw input
preservation in Activity ledger, dedup by hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from engram.core.concurrency import ContextLockManager
from engram.core.config import IngestionConfig
from engram.core.delta import DeltaEngine
from engram.core.events import EventBus
from engram.core.exceptions import CapacityExceededError, IngestionError
from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    BulletType,
    ContentType,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    ExecutionFeedback,
    FeedbackOutcome,
    Reflection,
    ReflectionInsight,
    SourceType,
    cap_core_memory,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Core-memory capping now lives in engram.core.models.cap_core_memory so it can
# be enforced at the delta-apply layer too (single source of truth for the bound).
_cap_core_memory = cap_core_memory


# ── Reflector ──────────────────────────────────────────────────────────────


REFLECTOR_SYSTEM_PROMPT = """\
You are the Reflector in a brain-inspired memory system. Your job is to analyze
raw input and extract structured insights. You do NOT decide what to store —
the Curator handles that.

Given raw input (and optional execution feedback), extract:
1. New insights — atomic, actionable knowledge bullets
2. Strategies that worked
3. Failure modes — what went wrong and why
4. Prediction errors — things that violated expectations (these get extra attention)
5. Open questions — unresolved uncertainties

For each insight:
- Make it ATOMIC (one idea)
- Classify its type: "strategy", "warning", "fact", "procedure", "exception",
  "principle", "decision", or "episodic"
- Use "episodic" for timestamped events ("At 14:30 UTC the user agreed to X");
  prefix the content with "At {timestamp}, {actor}" when classifying as episodic
- Suggest a section grouping
- Rate novelty (0-1, how new vs existing context)

You will also see the current CORE MEMORY — a short, always-in-context running
summary of who, what, and where this context is. If — and only if — there is a
meaningful update to make (new stable facts about the user/task, a resolved
ambiguity, a status change), emit `core_memory_update` containing the FULL new
core memory text. The new value REPLACES the old wholesale, so explicitly carry
forward anything still relevant. Keep it under ~400 words. Leave it null when no
change is warranted (the common case).

Respond ONLY with valid JSON:
{
  "new_insights": [
    {"content": "...", "insight_type": "fact", "suggested_section": "ocr_tools",
     "evidence": "...", "novelty": 0.8}
  ],
  "strategies_that_worked": ["..."],
  "failure_modes": ["..."],
  "prediction_errors": ["..."],
  "open_questions": ["..."],
  "core_memory_update": null,
  "confidence": 0.8
}"""


class ReflectorEngine:
    """Phase 1: Analyze raw input to extract structured insights.

    Brain analogy: Hippocampus encoding — processing new experiences
    into structured memory traces.

    v0.4: Accepts IngestionConfig to use the canonical Reflector model.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        config: IngestionConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or IngestionConfig()

    async def reflect(
        self,
        raw_input: str,
        feedback: ExecutionFeedback | None = None,
        existing_context_summary: str | None = None,
        existing_core_memory: str | None = None,
        max_rounds: int | None = None,
        model_override: str | None = None,
    ) -> Reflection:
        """Analyze raw input and produce a structured Reflection.

        Uses the CANONICAL Reflector model from server config unless
        model_override is specified (used by re-extraction).
        """
        feedback_text = ""
        if feedback:
            feedback_text = f"\n\nExecution feedback:\n- Outcome: {feedback.outcome.value}"
            if feedback.metrics:
                feedback_text += f"\n- Metrics: {json.dumps(feedback.metrics)}"
            if feedback.error_message:
                feedback_text += f"\n- Error: {feedback.error_message}"
            if feedback.tool_calls:
                calls = [f"  {tc.tool}: {tc.status.value}" + (f" ({tc.error})" if tc.error else "")
                         for tc in feedback.tool_calls]
                feedback_text += "\n- Tool calls:\n" + "\n".join(calls)

        context_text = ""
        if existing_context_summary:
            context_text = f"\n\nExisting context summary:\n{existing_context_summary}"

        core_text = ""
        if existing_core_memory is not None:
            core_text = (
                f"\n\nCurrent core memory (running always-in-context summary):\n"
                f"{existing_core_memory or '(empty)'}"
            )

        prompt = (
            f"Analyze this input and extract insights:"
            f"{context_text}{core_text}{feedback_text}\n\nRaw input:\n{raw_input}"
        )

        raw_response = await self.llm.complete(
            prompt=prompt,
            system=REFLECTOR_SYSTEM_PROMPT,
            temperature=0.0,
            response_format="json",
            model=model_override,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            try:
                start = raw_response.index("{")
                end = raw_response.rindex("}") + 1
                data = json.loads(raw_response[start:end])
            except (ValueError, json.JSONDecodeError) as exc:
                raise IngestionError(f"Reflector returned invalid JSON: {raw_response[:200]}") from exc

        insights = [
            ReflectionInsight(
                content=ins.get("content", ""),
                insight_type=ins.get("insight_type", "fact"),
                suggested_section=ins.get("suggested_section", "general"),
                evidence=ins.get("evidence", ""),
                novelty=ins.get("novelty", 0.5),
            )
            for ins in data.get("new_insights", [])
        ]

        core_memory_update = data.get("core_memory_update")
        if isinstance(core_memory_update, str) and not core_memory_update.strip():
            core_memory_update = None

        return Reflection(
            new_insights=insights,
            strategies_that_worked=data.get("strategies_that_worked", []),
            failure_modes=data.get("failure_modes", []),
            prediction_errors=data.get("prediction_errors", []),
            open_questions=data.get("open_questions", []),
            rounds_completed=1,
            confidence=data.get("confidence", 0.5),
            raw_input_type="conversation",
            core_memory_update=core_memory_update if isinstance(core_memory_update, str) else None,
        )


# ── Curator ────────────────────────────────────────────────────────────────


VALIDITY_GATE_SYSTEM_PROMPT = """\
You are the Validity Gate in a memory ingestion pipeline. Each proposed write
is a short bullet that will become a long-lived memory item. Your job is to
reject writes that are NOT worth storing:
- empty / whitespace only / placeholder
- trivially restates the input or a tautology
- malformed (truncated mid-sentence, garbled, not a complete idea)
- pure conversational filler ("ok", "sounds good", "thanks")
Keep writes that are concrete, non-trivial knowledge — facts, decisions,
strategies, warnings, procedures, exceptions, principles.

Return ONLY valid JSON of the form:
{"verdicts": [{"idx": 0, "keep": true}, {"idx": 1, "keep": false, "reason": "..."}, ...]}
"""


class CuratorEngine:
    """Phase 2: Decide what delta operations to apply.

    Brain analogy: Neocortex integration — deciding what to store,
    how to connect it, and what to update.

    KEY INSIGHT: ~70% of operations use lightweight, non-LLM techniques
    (embedding dedup, deterministic merge). Only complex conflicts need LLM.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm: LLMAdapter,
        ingestion_config: IngestionConfig | None = None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.ingestion_config = ingestion_config or IngestionConfig()

    async def curate(
        self,
        context_id: str,
        reflection: Reflection,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> DeltaBatch:
        """Given a Reflection, produce delta operations."""
        existing_bullets = await self.storage.list_bullets(context_id)
        operations: list[DeltaOperation] = []

        for insight in reflection.new_insights:
            op = await self._process_insight(
                context_id, insight, existing_bullets, agent_id, session_id
            )
            if op:
                operations.append(op)

        # Prediction errors become high-salience EXCEPTION bullets
        for error in reflection.prediction_errors:
            operations.append(DeltaOperation(
                op_type=DeltaOpType.ADD_BULLET,
                target_id=str(uuid.uuid4())[:8],
                section="prediction_errors",
                content=error,
                bullet_type="exception",
                reasoning="Prediction error detected by Reflector — high salience signal",
                source=DeltaSource.REFLECTOR,
                confidence=0.8,
                agent_id=agent_id,
                session_id=session_id,
            ))

        # Failure modes become WARNING bullets
        for failure in reflection.failure_modes:
            operations.append(DeltaOperation(
                op_type=DeltaOpType.ADD_BULLET,
                target_id=str(uuid.uuid4())[:8],
                section="failure_modes",
                content=failure,
                bullet_type="warning",
                reasoning="Failure mode identified by Reflector",
                source=DeltaSource.REFLECTOR,
                confidence=0.7,
                agent_id=agent_id,
                session_id=session_id,
            ))

        # Successful strategies → STRATEGY bullets
        for strategy in reflection.strategies_that_worked:
            operations.append(DeltaOperation(
                op_type=DeltaOpType.ADD_BULLET,
                target_id=str(uuid.uuid4())[:8],
                section="strategies",
                content=strategy,
                bullet_type="strategy",
                reasoning="Successful strategy recorded by Reflector",
                source=DeltaSource.REFLECTOR,
                confidence=0.7,
                agent_id=agent_id,
                session_id=session_id,
            ))

        # Mem-α-inspired validity gate: drop malformed/trivial ADD_BULLET ops
        # via a single batched LLM judge call. Opt-in to avoid the extra cost.
        if self.ingestion_config.enable_validity_gate and operations:
            operations = await self._filter_invalid_ops(operations)

        batch = DeltaBatch(
            context_id=context_id,
            operations=operations,
            trigger="commit",
        )
        return batch

    async def _filter_invalid_ops(
        self, ops: list[DeltaOperation],
    ) -> list[DeltaOperation]:
        """Run a batched LLM judge over all ADD_BULLET ops; drop the ones it
        rejects. Non-ADD ops pass through untouched. Errors fall back to
        keeping everything (fail-open — we'd rather store too much than too
        little when the judge is unavailable)."""
        add_ops_with_idx = [
            (i, op) for i, op in enumerate(ops)
            if op.op_type == DeltaOpType.ADD_BULLET and op.content
        ]
        if not add_ops_with_idx:
            return ops

        # Build a numbered list for the judge.
        candidate_text = "\n".join(
            f"[{i}] {op.content}" for i, (_, op) in enumerate(add_ops_with_idx)
        )
        prompt = (
            "Evaluate each candidate memory write below. Reject any that are "
            "empty, malformed, trivial, conversational filler, or otherwise "
            "not worth storing as a long-lived memory item.\n\n"
            f"Candidates:\n{candidate_text}\n\n"
            "Return JSON with verdicts for each candidate by its bracketed index."
        )

        # The validity gate runs on its own (typically cheaper) model rather than
        # the canonical Reflector — route the override through llm.complete().
        gate_model = self.ingestion_config.validity_gate_model
        try:
            raw = await self.llm.complete(
                prompt=prompt,
                system=VALIDITY_GATE_SYSTEM_PROMPT,
                temperature=0.0,
                response_format="json",
                model=gate_model,
            )
            data = json.loads(raw) if not isinstance(raw, dict) else raw
            verdicts = data.get("verdicts", [])
        except Exception as exc:
            logger.warning(
                "Validity gate (%s) failed, passing all ops through: %s",
                gate_model, exc,
            )
            return ops

        # Map judge indices back to op positions.
        rejected_judge_indices: set[int] = set()
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            try:
                jidx = int(v.get("idx", -1))
            except (TypeError, ValueError):
                continue
            if v.get("keep") is False and 0 <= jidx < len(add_ops_with_idx):
                rejected_judge_indices.add(jidx)
                logger.debug(
                    "Validity gate rejected: %s (reason: %s)",
                    add_ops_with_idx[jidx][1].content[:80],
                    v.get("reason", "unspecified"),
                )

        if not rejected_judge_indices:
            return ops

        # Surface the case where the gate rejected every candidate write — an
        # over-aggressive or misconfigured judge can otherwise silently swallow
        # a whole commit's worth of knowledge with no operator-visible signal.
        if len(rejected_judge_indices) == len(add_ops_with_idx):
            logger.warning(
                "Validity gate (%s) rejected ALL %d candidate bullet(s); "
                "this commit will add no bullets",
                gate_model, len(add_ops_with_idx),
            )

        rejected_op_positions = {
            add_ops_with_idx[j][0] for j in rejected_judge_indices
        }
        return [op for i, op in enumerate(ops) if i not in rejected_op_positions]

    async def _process_insight(
        self,
        context_id: str,
        insight: ReflectionInsight,
        existing_bullets: list[Bullet],
        agent_id: str | None,
        session_id: str | None,
    ) -> DeltaOperation | None:
        """Process a single insight — fast path (no LLM) or slow path (LLM)."""

        is_episodic = insight.insight_type == BulletType.EPISODIC.value

        # Fast path: exact content match → skip
        for existing in existing_bullets:
            if existing.content.strip().lower() == insight.content.strip().lower():
                logger.debug("Skipping exact duplicate: %s", insight.content[:60])
                return None

        # Fast path: embedding similarity → fold the new insight into the
        # existing bullet (UPDATE_BULLET). The new insight is not yet a stored
        # bullet, so a MERGE_BULLETS over two existing ids cannot represent it —
        # we update the survivor in place, keeping the richer content and its
        # accumulated usage stats.
        # Episodic bullets use a looser threshold (multiple agents often log
        # paraphrases of the same event) AND merge only against same-type
        # episodics — a paraphrased event should never collapse into a fact.
        embedding: list[float] | None = None
        try:
            embedding = await self.llm.embed(insight.content)
            threshold = 0.85 if is_episodic else 0.92
            similar = await self.storage.find_similar_bullets(
                context_id, embedding, limit=5, threshold=threshold,
            )
            if is_episodic:
                similar = [
                    (b, s) for b, s in similar
                    if (b.bullet_type.value if hasattr(b.bullet_type, "value")
                        else str(b.bullet_type)) == "episodic"
                ]
            if similar:
                existing_bullet, score = similar[0]
                # Keep the more specific (longer) phrasing of the two.
                merged_content = (
                    insight.content
                    if len(insight.content) > len(existing_bullet.content)
                    else existing_bullet.content
                )
                logger.debug("Found similar bullet (%.3f): folding into existing", score)
                return DeltaOperation(
                    op_type=DeltaOpType.UPDATE_BULLET,
                    target_id=existing_bullet.id,
                    content=merged_content,
                    reasoning=f"Merged similar insight into existing bullet (cosine={score:.3f})",
                    source=DeltaSource.CURATOR,
                    confidence=max(existing_bullet.confidence, insight.novelty),
                    agent_id=agent_id,
                    session_id=session_id,
                )
        except Exception as exc:
            logger.warning("Embedding dedup failed, adding as new: %s", exc)

        # Fast path: contradiction detection — skip for episodic since each
        # event is its own atomic record, not a competing claim.
        try:
            if embedding is not None and not is_episodic:
                contradiction = self._detect_contradiction(
                    insight.content, embedding, existing_bullets
                )
                if contradiction is not None:
                    existing_bullet, negation_score = contradiction
                    logger.info(
                        "Contradiction detected (score=%.2f) between new insight and bullet %s",
                        negation_score, existing_bullet.id,
                    )
                    # Add the new bullet with a note about the contradiction
                    return DeltaOperation(
                        op_type=DeltaOpType.ADD_BULLET,
                        target_id=str(uuid.uuid4())[:8],
                        section=insight.suggested_section,
                        content=insight.content,
                        bullet_type=insight.insight_type if insight.insight_type in [bt.value for bt in BulletType] else "fact",
                        reasoning=(
                            f"Contradicts existing bullet {existing_bullet.id}: "
                            f"'{existing_bullet.content[:60]}' — keeping both for review"
                        ),
                        source=DeltaSource.CURATOR,
                        confidence=insight.novelty,
                        agent_id=agent_id,
                        session_id=session_id,
                    )
        except Exception as exc:
            logger.warning("Contradiction detection failed: %s", exc)

        # Default: add as new bullet
        bullet_type = insight.insight_type
        try:
            BulletType(bullet_type)
        except ValueError:
            bullet_type = "fact"

        return DeltaOperation(
            op_type=DeltaOpType.ADD_BULLET,
            target_id=str(uuid.uuid4())[:8],
            section=insight.suggested_section,
            content=insight.content,
            bullet_type=bullet_type,
            reasoning=f"New insight (novelty={insight.novelty:.2f}): {insight.evidence[:80]}",
            source=DeltaSource.CURATOR,
            confidence=insight.novelty,
            agent_id=agent_id,
            session_id=session_id,
        )


    @staticmethod
    def _has_negation_signal(text_a: str, text_b: str) -> bool:
        """Fast heuristic: check if two texts have opposing signals."""
        negation_pairs = [
            ("not ", " not"), ("don't", "do"), ("doesn't", "does"),
            ("shouldn't", "should"), ("won't", "will"), ("can't", "can"),
            ("avoid", "use"), ("instead of", ""), ("rather than", ""),
            ("no longer", "still"), ("deprecated", "recommended"),
            ("worse", "better"), ("failed", "succeeded"),
        ]
        a_lower = text_a.lower()
        b_lower = text_b.lower()
        for neg, pos in negation_pairs:
            if (neg in a_lower and pos and pos in b_lower) or \
               (neg in b_lower and pos and pos in a_lower):
                return True
        return False

    def _detect_contradiction(
        self,
        new_content: str,
        new_embedding: list[float],
        existing_bullets: list[Bullet],
    ) -> tuple[Bullet, float] | None:
        """Detect if new content contradicts an existing bullet.

        Uses a two-pass approach:
        1. Find semantically similar bullets (same topic)
        2. Check for negation signals (opposing conclusions)
        """
        from engram.core.similarity import cosine_similarity as _cosine_similarity

        for bullet in existing_bullets:
            if bullet.embedding is None:
                continue
            sim = _cosine_similarity(new_embedding, bullet.embedding)
            # High similarity + negation signal = contradiction
            if sim >= 0.75 and self._has_negation_signal(new_content, bullet.content):
                return (bullet, sim)
        return None

    async def curate_re_extraction(
        self,
        new_reflection: Reflection,
        old_bullets: list[Bullet],
        context_id: str,
    ) -> DeltaBatch:
        """Compare new extraction against old bullets from the same raw input.

        For each new insight:
        - If it matches an old bullet (>0.92 similarity) → no change (or minor UPDATE)
        - If it's genuinely new (no match) → ADD_BULLET
        - If an old bullet has no match in new insights → REMOVE_BULLET
          (only if low-value; high-value bullets are kept)

        This is conservative by default — high-value bullets (high salience + hit rate)
        are kept even if the new model doesn't reproduce them.
        """
        from engram.core.similarity import cosine_similarity as _cosine_similarity

        operations: list[DeltaOperation] = []
        matched_old_ids: set[str] = set()

        for insight in new_reflection.new_insights:
            best_match = None
            best_similarity = 0.0

            try:
                new_embedding = await self.llm.embed(insight.content)
            except Exception:
                new_embedding = None

            if new_embedding is not None:
                for old_bullet in old_bullets:
                    if old_bullet.id in matched_old_ids:
                        continue
                    if old_bullet.embedding is None:
                        continue
                    sim = _cosine_similarity(new_embedding, old_bullet.embedding)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = old_bullet

            if best_match and best_similarity > 0.92:
                # Match found — minor update if content differs meaningfully
                matched_old_ids.add(best_match.id)
                if best_similarity < 0.98:
                    operations.append(DeltaOperation(
                        op_type=DeltaOpType.UPDATE_BULLET,
                        target_id=best_match.id,
                        content=insight.content,
                        reasoning=(
                            f"Re-extraction improved this bullet "
                            f"(similarity: {best_similarity:.2f})"
                        ),
                        source=DeltaSource.CURATOR,
                    ))
            else:
                # No match — genuinely new insight from better model
                bullet_type = insight.insight_type
                try:
                    BulletType(bullet_type)
                except ValueError:
                    bullet_type = "fact"

                operations.append(DeltaOperation(
                    op_type=DeltaOpType.ADD_BULLET,
                    target_id=str(uuid.uuid4())[:8],
                    section=insight.suggested_section,
                    content=insight.content,
                    bullet_type=bullet_type,
                    reasoning=(
                        f"New insight discovered by re-extraction "
                        f"(best match: {best_similarity:.2f})"
                    ),
                    source=DeltaSource.CURATOR,
                ))

        # Old bullets with no match in new extraction — consider removal
        for old_bullet in old_bullets:
            if old_bullet.id not in matched_old_ids:
                # Only remove if the old bullet has low salience and low hit rate
                # High-value bullets are kept even if re-extraction doesn't reproduce them
                if old_bullet.effective_salience < 0.3 and old_bullet.hit_rate < 0.3:
                    operations.append(DeltaOperation(
                        op_type=DeltaOpType.REMOVE_BULLET,
                        target_id=old_bullet.id,
                        reasoning=(
                            f"Not reproduced by re-extraction and low value "
                            f"(salience: {old_bullet.effective_salience:.2f}, "
                            f"hit_rate: {old_bullet.hit_rate:.2f})"
                        ),
                        source=DeltaSource.CURATOR,
                    ))

        return DeltaBatch(
            context_id=context_id,
            operations=operations,
            trigger="re_extraction",
        )


# ── Unified Ingestion Engine ──────────────────────────────────────────────


class IngestionEngine:
    """Unified ingestion: Reflector → Curator → Delta application.

    Also provides direct bullet addition and decision recording.
    v0.3: Optional lock_manager and event_bus for concurrency and notifications.
    v0.4: Canonical Reflector model via IngestionConfig, raw input preservation,
          dedup by hash.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm: LLMAdapter,
        lock_manager: ContextLockManager | None = None,
        event_bus: EventBus | None = None,
        ingestion_config: IngestionConfig | None = None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.lock_manager = lock_manager
        self.event_bus = event_bus
        self.ingestion_config = ingestion_config or IngestionConfig()
        self.reflector = ReflectorEngine(llm, config=self.ingestion_config)
        self.curator = CuratorEngine(storage, llm, ingestion_config=self.ingestion_config)
        self.delta_engine = DeltaEngine(storage)

    async def commit(
        self,
        context_id: uuid.UUID,
        agent_id: str,
        content: str,
        content_type: ContentType = ContentType.CONVERSATION,
        session_id: uuid.UUID | None = None,
        feedback: ExecutionFeedback | None = None,
        materialization_id: str | None = None,
        domain_tags: list[str] | None = None,
        source_model: str | None = None,
    ) -> DeltaBatch:
        """Full ingestion pipeline: Reflect → Curate → Apply deltas.

        If materialization_id is provided, triggers reconsolidation.
        v0.4: Dedup by hash, raw input preservation, extraction metadata.
        """
        ctx_id_str = str(context_id)

        # v0.4: Dedup check — skip if identical raw input already processed
        input_hash = hashlib.sha256(content.encode()).hexdigest()
        existing_activity = await self.storage.get_raw_input_by_hash(
            ctx_id_str, input_hash
        )
        if existing_activity is not None and existing_activity.delta_batch_id:
            existing_batch = await self.storage.get_delta_batch(
                existing_activity.delta_batch_id
            )
            if existing_batch is not None:
                logger.info(
                    "Duplicate raw input detected (hash=%s), returning existing batch",
                    input_hash[:16],
                )
                return existing_batch

        # Build context summary + load core memory for the Reflector
        existing_bullets = await self.storage.list_bullets(ctx_id_str)
        summary = self._summarize_bullets(existing_bullets) if existing_bullets else None
        existing_context = await self.storage.get_context(context_id)
        existing_core_memory = existing_context.core_memory if existing_context else ""

        # Phase 1: Reflect (using CANONICAL model from config)
        reflection = await self.reflector.reflect(
            raw_input=content,
            feedback=feedback,
            existing_context_summary=summary,
            existing_core_memory=existing_core_memory,
        )

        # Phase 2: Curate
        batch = await self.curator.curate(
            context_id=ctx_id_str,
            reflection=reflection,
            agent_id=agent_id,
            session_id=str(session_id) if session_id else None,
        )

        # Phase 2.5: Append a core memory update op if the Reflector proposed one.
        # Going through DeltaEngine keeps the change auditable + rollback-able.
        if reflection.core_memory_update is not None:
            new_core = _cap_core_memory(reflection.core_memory_update)
            if new_core != existing_core_memory:
                batch.operations.append(DeltaOperation(
                    op_type=DeltaOpType.UPDATE_CORE_MEMORY,
                    target_id=ctx_id_str,
                    content=new_core,
                    reasoning="Reflector emitted a core memory update.",
                    source=DeltaSource.REFLECTOR,
                    confidence=reflection.confidence,
                    agent_id=agent_id,
                    session_id=str(session_id) if session_id else None,
                ))

        # v0.3: Capacity check before applying deltas
        net_adds = sum(
            1 for op in batch.operations if op.op_type == DeltaOpType.ADD_BULLET
        ) - sum(
            1 for op in batch.operations if op.op_type == DeltaOpType.REMOVE_BULLET
        )
        if net_adds > 0:
            context = await self.storage.get_context(context_id)
            if context is not None:
                max_bullets = context.lifecycle_config.max_active_bullets
                capacity = await self.storage.get_capacity_status(
                    ctx_id_str, max_bullets
                )
                if capacity.pressure_level == "full":
                    raise CapacityExceededError(
                        ctx_id_str, capacity.active_bullet_count, max_bullets
                    )

        # Phase 3: Apply deltas (under lock if available)
        if self.lock_manager is not None:
            async with self.lock_manager.acquire(ctx_id_str):
                self._revalidate_deltas(batch)
                batch = await self.delta_engine.apply_batch(batch)
        else:
            batch = await self.delta_engine.apply_batch(batch)

        # v0.4: Collect bullet IDs produced by ADD_BULLET operations
        bullet_ids_produced = [
            op.target_id
            for op in batch.operations
            if op.op_type == DeltaOpType.ADD_BULLET and op.target_id
        ]

        # v0.5: embed raw input so future materializations can retrieve it as a
        # DC-style worked example. Best-effort — recall path tolerates absence.
        raw_input_embedding: list[float] | None = None
        try:
            raw_input_embedding = await self.llm.embed(content)
        except Exception as exc:
            logger.warning("Raw-input embedding failed (worked-example retrieval disabled for this commit): %s", exc)

        # Record activity WITH raw input preservation (v0.4) + embedding (v0.5)
        activity = Activity(
            agent_id=agent_id,
            session_id=session_id,
            action_type=ActionType.FACT_LEARNED,
            summary=(
                f"Ingested {content_type.value}: +{batch.bullets_added} bullets, "
                f"~{batch.bullets_updated} updated, -{batch.bullets_removed} removed, "
                f"⊕{batch.bullets_merged} merged"
            ),
            delta_batch_id=batch.id,
            # v0.4: Raw input preservation
            raw_input=content,
            raw_input_hash=input_hash,
            content_type=content_type.value,
            source_agent_model=source_model,
            feedback=feedback.model_dump() if feedback else None,
            # v0.4: Extraction metadata
            extraction_model=self.ingestion_config.reflector_model,
            extraction_prompt_version=self.ingestion_config.reflector_prompt_version,
            bullet_ids_produced=bullet_ids_produced,
            # v0.5: worked-example retrieval
            raw_input_embedding=raw_input_embedding,
        )
        await self.storage.add_activity(context_id, activity)

        # Reconsolidation: if this commit references a materialization, update
        # bullet stats via a delta batch (audit-clean, rollback-able).
        if materialization_id and feedback:
            await self._reconsolidate(
                materialization_id, feedback,
                agent_id=agent_id, session_id=session_id,
            )

        # v0.3: Emit event
        if self.event_bus is not None:
            self.event_bus.emit(
                ctx_id_str,
                event_type="commit",
                agent_id=agent_id,
                data={
                    "delta_batch_id": batch.id,
                    "bullets_added": batch.bullets_added,
                    "bullets_updated": batch.bullets_updated,
                    "bullets_removed": batch.bullets_removed,
                    "bullets_merged": batch.bullets_merged,
                },
            )

        return batch

    async def add_bullet_directly(
        self,
        context_id: str,
        content: str,
        section: str = "general",
        bullet_type: BulletType = BulletType.FACT,
        salience: float = 0.5,
        confidence: float = 0.5,
        agent_id: str = "manual",
        session_id: str | None = None,
    ) -> tuple[Bullet, DeltaBatch]:
        """Add a single bullet directly (bypass Reflector)."""
        # v0.3: Capacity check
        context = await self.storage.get_context(uuid.UUID(context_id))
        if context is not None:
            max_bullets = context.lifecycle_config.max_active_bullets
            capacity = await self.storage.get_capacity_status(context_id, max_bullets)
            if capacity.pressure_level == "full":
                raise CapacityExceededError(
                    context_id, capacity.active_bullet_count, max_bullets
                )

        bullet_id = str(uuid.uuid4())[:8]
        op = DeltaOperation(
            op_type=DeltaOpType.ADD_BULLET,
            target_id=bullet_id,
            section=section,
            content=content,
            bullet_type=bullet_type.value,
            reasoning="Direct bullet addition (bypass Reflector)",
            source=DeltaSource.USER,
            confidence=confidence,
            agent_id=agent_id,
            session_id=session_id,
        )
        batch = DeltaBatch(
            context_id=context_id,
            operations=[op],
            trigger="direct_add",
        )

        # v0.3: Apply under lock if available
        if self.lock_manager is not None:
            async with self.lock_manager.acquire(context_id):
                batch = await self.delta_engine.apply_batch(batch)
        else:
            batch = await self.delta_engine.apply_batch(batch)

        bullet = await self.storage.get_bullet(bullet_id)
        if bullet:
            bullet.salience = salience
            try:
                bullet.embedding = await self.llm.embed(content)
            except Exception:
                pass
            await self.storage.update_bullet(bullet)

        # v0.3: Emit event
        if self.event_bus is not None:
            self.event_bus.emit(
                context_id,
                event_type="bullet_added",
                agent_id=agent_id,
                data={"bullet_id": bullet_id},
            )

        return bullet or Bullet(id=bullet_id, content=content), batch

    async def record_decision(
        self,
        context_id: uuid.UUID,
        decision: str,
        rationale: str,
        alternatives: list[str],
        agent_id: str,
        domain_tags: list[str] | None = None,
        session_id: uuid.UUID | None = None,
    ) -> tuple[str, DeltaBatch]:
        """Record a decision as a DECISION bullet with alternatives."""
        ctx_id_str = str(context_id)
        operations: list[DeltaOperation] = []

        decision_id = str(uuid.uuid4())[:8]
        operations.append(DeltaOperation(
            op_type=DeltaOpType.ADD_BULLET,
            target_id=decision_id,
            section="decisions",
            content=f"{decision}. Rationale: {rationale}",
            bullet_type="decision",
            reasoning=f"Decision recorded: {decision}",
            source=DeltaSource.USER,
            confidence=0.9,
            agent_id=agent_id,
            session_id=str(session_id) if session_id else None,
        ))

        for alt in alternatives:
            operations.append(DeltaOperation(
                op_type=DeltaOpType.ADD_BULLET,
                target_id=str(uuid.uuid4())[:8],
                section="decisions",
                content=f"Alternative considered: {alt}",
                bullet_type="fact",
                reasoning=f"Alternative for decision: {decision}",
                source=DeltaSource.USER,
                confidence=0.7,
                agent_id=agent_id,
                session_id=str(session_id) if session_id else None,
            ))

        batch = DeltaBatch(
            context_id=ctx_id_str,
            operations=operations,
            trigger="decision",
        )
        batch = await self.delta_engine.apply_batch(batch)

        activity = Activity(
            agent_id=agent_id,
            session_id=session_id,
            action_type=ActionType.DECISION_MADE,
            summary=f"Decision: {decision}",
            delta_batch_id=batch.id,
        )
        await self.storage.add_activity(context_id, activity)

        return decision_id, batch

    async def _reconsolidate(
        self, materialization_id: str, feedback: ExecutionFeedback,
        agent_id: str | None = None, session_id: uuid.UUID | None = None,
    ) -> DeltaBatch | None:
        """Post-recall reconsolidation, routed through the DeltaEngine.

        Emits one RECONSOLIDATE_BULLET op per bullet in the recall, applied as
        a single batch so the audit trail / rollback story is consistent with
        every other mutation. (README claim: "all mutations through deltas.")
        """
        record = await self.storage.get_materialization(materialization_id)
        if record is None:
            logger.warning("Materialization %s not found for reconsolidation", materialization_id)
            return None

        # Idempotency: a materialization may be reconsolidated exactly once.
        # Without this, a replayed materialization_id (retry, at-least-once
        # delivery, or two commits citing one recall) double-counts hits and
        # compounds salience.
        if record.reconsolidated_at is not None:
            logger.info(
                "Materialization %s already reconsolidated at %s; skipping replay",
                materialization_id, record.reconsolidated_at,
            )
            return None

        # Decide the per-bullet effect from the outcome once.
        if feedback.outcome == FeedbackOutcome.SUCCESS:
            recall_delta, hit_delta, miss_delta, mult = 1, 1, 0, 1.05
        elif feedback.outcome == FeedbackOutcome.FAILURE:
            recall_delta, hit_delta, miss_delta, mult = 1, 0, 1, 0.95
        else:
            recall_delta, hit_delta, miss_delta, mult = 1, 0, 0, 1.00

        ops: list[DeltaOperation] = []
        for bullet_id in record.bullets_included:
            ops.append(DeltaOperation(
                op_type=DeltaOpType.RECONSOLIDATE_BULLET,
                target_id=bullet_id,
                reasoning=(
                    f"Reconsolidation from materialization {materialization_id} "
                    f"(outcome={feedback.outcome.value})"
                ),
                source=DeltaSource.REFLECTOR,
                confidence=0.9,
                agent_id=agent_id,
                session_id=str(session_id) if session_id else None,
                # previous_state carries the *input* deltas; DeltaEngine swaps
                # it for a rollback snapshot after applying.
                previous_state={
                    "recall_delta": recall_delta,
                    "hit_delta": hit_delta,
                    "miss_delta": miss_delta,
                    "salience_multiplier": mult,
                    "outcome": feedback.outcome.value,
                },
            ))
        if not ops:
            return None

        batch = DeltaBatch(
            context_id=record.context_id,
            operations=ops,
            trigger="reconsolidation",
        )
        if self.lock_manager is not None:
            async with self.lock_manager.acquire(record.context_id):
                # Re-check under the lock so two concurrent replays can't both
                # pass the pre-lock idempotency guard and double-apply.
                fresh = await self.storage.get_materialization(materialization_id)
                if fresh is not None and fresh.reconsolidated_at is not None:
                    logger.info(
                        "Materialization %s reconsolidated concurrently; skipping",
                        materialization_id,
                    )
                    return None
                batch = await self.delta_engine.apply_batch(batch)
                await self.storage.mark_materialization_reconsolidated(
                    materialization_id, _utcnow(),
                )
        else:
            batch = await self.delta_engine.apply_batch(batch)
            await self.storage.mark_materialization_reconsolidated(
                materialization_id, _utcnow(),
            )

        logger.info(
            "Reconsolidation: %d bullets updated from materialization %s (outcome=%s)",
            len(ops), materialization_id, feedback.outcome.value,
        )
        return batch

    @staticmethod
    def _revalidate_deltas(batch: DeltaBatch) -> None:
        """Revalidate delta operations inside the lock.

        After parallel computation but before application, check that
        operations targeting existing bullets are still valid. This is
        a lightweight check — actual existence is verified by delta_engine.
        """
        seen_targets: set[str] = set()
        valid_ops: list[DeltaOperation] = []
        for op in batch.operations:
            # Deduplicate operations targeting the same bullet
            target = op.target_id or ""
            if op.op_type in (DeltaOpType.UPDATE_BULLET, DeltaOpType.REMOVE_BULLET):
                if target in seen_targets:
                    logger.debug("Dropping duplicate op for target %s", target)
                    continue
                seen_targets.add(target)
            valid_ops.append(op)
        batch.operations = valid_ops

    @staticmethod
    def _summarize_bullets(bullets: list[Bullet]) -> str:
        if not bullets:
            return ""
        lines = []
        for b in bullets[:40]:
            lines.append(f"- [{b.bullet_type.value if hasattr(b.bullet_type, 'value') else b.bullet_type}] {b.content}")
        return "\n".join(lines)

    # ── Legacy compat ──────────────────────────────────────────────────

    async def add_concept_directly(
        self, context_id: uuid.UUID, concept: Any,
        relationships: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, list[uuid.UUID]]:
        """Legacy wrapper — adds concept as a bullet internally."""
        from engram.core.models import ConceptNode
        if isinstance(concept, ConceptNode):
            try:
                concept.embedding = await self.llm.embed(concept.content)
            except Exception:
                pass
            await self.storage.add_concept(context_id, concept)
            return concept, []
        return concept, []
