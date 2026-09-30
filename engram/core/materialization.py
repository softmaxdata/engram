"""Materialization engine — assembles relevant context from the graph and renders it.

v0.2: Uses effective_salience, schema-aware assembly, recall tracking,
and returns materialization_id for reconsolidation.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from engram.core.exceptions import LLMAdapterError, StorageError
from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    ConceptEdge,
    ConceptNode,
    ConceptType,
    IntentAnchor,
    MaterializationRecord,
    SchemaNode,
)
from engram.core.similarity import cosine_similarity
from engram.llm.adapter import LLMAdapter
from engram.renderers.base import ContextRenderer
from engram.renderers.claude import ClaudeRenderer
from engram.renderers.generic import GenericRenderer
from engram.renderers.gpt import GPTRenderer
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)

RENDERER_MAP: dict[str, type[ContextRenderer]] = {
    "claude": ClaudeRenderer,
    "gpt": GPTRenderer,
    "gpt-4": GPTRenderer,
    "gpt-4o": GPTRenderer,
    "gpt-4o-mini": GPTRenderer,
    "o1": GPTRenderer,
    "o3": GPTRenderer,
    "generic": GenericRenderer,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# MMR is only ever used to fill a token budget — at typical bullet sizes that
# caps how many we'll actually pick. Prefiltering to MMR_CANDIDATE_CAP best-by-
# relevance candidates keeps the loop O(K^2) instead of O(N^2) on large contexts.
MMR_CANDIDATE_CAP = 64

# Worked examples are drawn from raw activity inputs, which are unbounded in
# length. Cap each example's input/output so the <worked_examples> block (which
# renderers append after the budgeted bullet packing) can't blow the token
# budget on a single large prior input. ~800 chars ≈ 200 tokens per field.
WORKED_EXAMPLE_INPUT_MAX_CHARS = 800
WORKED_EXAMPLE_OUTPUT_MAX_CHARS = 800


def _truncate(text: str, max_chars: int) -> str:
    """Truncate to a char budget with an ellipsis marker when cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …[truncated]"


def _mmr_order(
    bullets: list[Bullet],
    relevance: dict[str, float],
    lambda_: float,
    candidate_cap: int = MMR_CANDIDATE_CAP,
) -> list[Bullet]:
    """Maximal Marginal Relevance ordering.

    score(b) = λ * relevance(b) - (1 - λ) * max_sim(b, already_picked)

    Falls back to greedy-by-relevance when λ ≥ 0.999 (legacy behavior). Bullets
    without embeddings contribute zero redundancy — they're ranked purely on
    relevance (ordering remains stable, just no diversity penalty).

    To stay tractable on large contexts, the candidate set is pre-trimmed to
    the top `candidate_cap` by relevance before the O(K^2) selection loop runs.
    """
    if lambda_ >= 0.999:
        return [b for b, _ in sorted(
            [(b, relevance.get(b.id, 0.0)) for b in bullets],
            key=lambda x: x[1], reverse=True,
        )]
    ranked = sorted(bullets, key=lambda b: relevance.get(b.id, 0.0), reverse=True)
    remaining = ranked[:candidate_cap]
    picked: list[Bullet] = []
    while remaining:
        best_idx = 0
        best_score = -math.inf
        for i, cand in enumerate(remaining):
            rel = relevance.get(cand.id, 0.0)
            redundancy = 0.0
            if cand.embedding is not None and picked:
                redundancy = max(
                    (cosine_similarity(cand.embedding, p.embedding) for p in picked
                     if p.embedding is not None),
                    default=0.0,
                )
            mmr = lambda_ * rel - (1 - lambda_) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        picked.append(remaining.pop(best_idx))
    return picked


class MaterializationEngine:
    """Assembles and renders context from the concept graph.

    v0.2 changes:
    - Prefers effective_salience (salience × hit_rate) over raw salience
    - Schema-aware: includes schema descriptions for token efficiency
    - Tracks which bullets were included for reconsolidation
    - Returns materialization_id
    """

    def __init__(self, storage: StorageBackend, llm: LLMAdapter) -> None:
        self.storage = storage
        self.llm = llm

    async def materialize(
        self,
        context_id: uuid.UUID,
        query: str | None = None,
        task: str | None = None,
        agent_role: str | None = None,
        focus_domains: list[str] | None = None,
        focus_sections: list[str] | None = None,
        token_budget: int = 4000,
        target_model: str = "claude",
        include_intent: bool = True,
        include_decisions: bool = True,
        include_schemas: bool = True,
        recency_weight: float = 0.5,
        max_concept_age_days: int | None = None,
        include_worked_examples: bool = True,
        worked_example_threshold: float = 0.85,
        worked_example_limit: int = 2,
        include_usage_stats: bool = False,
        mmr_lambda: float = 0.7,
    ) -> dict[str, Any]:
        """Materialize context from the graph.

        Returns dict with: materialization_id, rendered_text, bullets_included,
        schemas_included, concepts_included, token_count, coverage_score
        """
        ctx_id_str = str(context_id)

        # 1. Load intent + core memory (Mem-α: always-in-context summary)
        intent: IntentAnchor | None = None
        if include_intent:
            intent = await self.storage.get_intent(context_id)
        context_obj = await self.storage.get_context(context_id)
        core_memory = context_obj.core_memory if context_obj else ""

        # 2. Load bullets (primary v0.2 storage)
        all_bullets = await self.storage.list_bullets(ctx_id_str)
        if max_concept_age_days is not None:
            cutoff = _utcnow() - timedelta(days=max_concept_age_days)
            all_bullets = [b for b in all_bullets if b.created_at >= cutoff]

        # Filter by section
        if focus_sections:
            section_set = set(focus_sections)
            all_bullets = [b for b in all_bullets if b.section in section_set] or all_bullets

        # 3. Load schemas
        schemas: list[SchemaNode] = []
        if include_schemas:
            schemas = await self.storage.list_schemas(ctx_id_str)

        # 4. Also load legacy concepts (backward compat)
        all_concepts = await self.storage.list_concepts(context_id, include_invalid=False)

        renderer = self._get_renderer(target_model)

        # v0.5: fetch worked examples (DC-style nearest-prior-input retrieval).
        search_text = query or task or ""
        worked_examples: list[dict[str, str]] = []
        if include_worked_examples and search_text:
            worked_examples = await self._fetch_worked_examples(
                ctx_id_str, search_text,
                threshold=worked_example_threshold,
                limit=worked_example_limit,
            )

        # If we have bullets, use bullet-based materialization
        if all_bullets:
            result = await self._materialize_bullets(
                ctx_id_str, context_id, all_bullets, schemas, intent,
                search_text, focus_domains or [], recency_weight,
                include_decisions, token_budget, target_model, renderer,
                core_memory=core_memory,
                worked_examples=worked_examples,
                include_usage_stats=include_usage_stats,
                mmr_lambda=mmr_lambda,
            )
        elif all_concepts:
            # Fallback to legacy concept-based materialization
            result = await self._materialize_concepts(
                context_id, all_concepts, intent, search_text,
                focus_domains or [], recency_weight, include_decisions,
                token_budget, target_model, renderer,
                core_memory=core_memory,
                worked_examples=worked_examples,
            )
        else:
            text = renderer.render(
                [], intent, token_budget,
                core_memory=core_memory, worked_examples=worked_examples or None,
            )
            result = {
                "materialization_id": str(uuid.uuid4()),
                "rendered_text": text,
                "bullets_included": [],
                "schemas_included": [],
                "concepts_included": [],
                "token_count": renderer.estimate_tokens(text),
                "coverage_score": 0.0,
            }

        # Save materialization record for reconsolidation tracking
        mat_record = MaterializationRecord(
            id=result["materialization_id"],
            context_id=ctx_id_str,
            bullets_included=result["bullets_included"],
            token_count=result["token_count"],
            target_model=target_model,
            query=query or task,
        )
        await self.storage.save_materialization(mat_record)

        # Log activity
        activity = Activity(
            agent_id="system",
            action_type=ActionType.MATERIALIZATION_OCCURRED,
            summary=f"Materialized {len(result['bullets_included'])} bullets for {target_model}",
            materialization_id=result["materialization_id"],
        )
        await self.storage.add_activity(context_id, activity)

        return result

    async def recall(
        self,
        context_id: uuid.UUID,
        query: str,
        token_budget: int = 2000,
        target_model: str = "claude",
    ) -> str:
        """Simplified materialization — returns just the rendered text."""
        result = await self.materialize(
            context_id=context_id, query=query,
            token_budget=token_budget, target_model=target_model,
        )
        return result["rendered_text"]

    async def _materialize_bullets(
        self, ctx_id_str: str, context_id: uuid.UUID,
        bullets: list[Bullet], schemas: list[SchemaNode],
        intent: IntentAnchor | None, search_text: str,
        focus_domains: list[str], recency_weight: float,
        include_decisions: bool, token_budget: int,
        target_model: str, renderer: ContextRenderer,
        core_memory: str = "",
        worked_examples: list[dict[str, str]] | None = None,
        include_usage_stats: bool = False,
        mmr_lambda: float = 0.7,
    ) -> dict[str, Any]:
        """Bullet-based materialization with effective salience ranking + MMR diversity."""
        mat_id = str(uuid.uuid4())

        # Score bullets
        scores = await self._score_bullets(
            ctx_id_str, bullets, search_text, recency_weight, include_decisions,
        )

        # MMR re-ordering: pick each next bullet to maximize relevance minus the
        # max similarity to already-picked. Avoids packing the token budget with
        # 10 near-duplicates of the top-scoring bullet.
        ordered_bullets = _mmr_order(bullets, scores, mmr_lambda)

        # Convert top bullets to ConceptNodes for renderer (backward compat with renderers)
        selected_concepts: list[ConceptNode] = []
        selected_bullet_ids: list[str] = []
        usage_stats: dict[str, str] = {}
        total_tokens = 0

        intent_budget = 0
        if intent or core_memory:
            intent_text = renderer.render([], intent, token_budget, core_memory=core_memory)
            intent_budget = renderer.estimate_tokens(intent_text)

        # Reserve some headroom for worked examples (rendered at the end).
        worked_example_budget = 0
        if worked_examples:
            wx_text_estimate = sum(
                renderer.estimate_tokens((ex.get("input") or "") + (ex.get("output") or ""))
                + 30
                for ex in worked_examples
            )
            worked_example_budget = wx_text_estimate

        remaining = max(0, token_budget - intent_budget - worked_example_budget)

        # Include schema summaries first (token-efficient)
        schema_ids: list[str] = []
        if schemas:
            for schema in schemas[:5]:
                schema_text = f"[Pattern: {schema.name}] {schema.description}"
                est = renderer.estimate_tokens(schema_text) + 10
                if total_tokens + est <= remaining:
                    selected_concepts.append(ConceptNode(
                        type=ConceptType.PATTERN,
                        content=schema_text,
                        salience=schema.confidence,
                    ))
                    total_tokens += est
                    schema_ids.append(schema.id)

        # Pack bullets into remaining budget
        for bullet in ordered_bullets:
            est = renderer.estimate_tokens(bullet.content) + 10
            if total_tokens + est > remaining:
                break
            concept_type = self._bullet_type_to_concept_type(
                bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else str(bullet.bullet_type)
            )
            selected_concepts.append(ConceptNode(
                type=concept_type,
                content=bullet.content,
                salience=bullet.salience,
                confidence=bullet.confidence,
            ))
            selected_bullet_ids.append(bullet.id)
            if include_usage_stats and (bullet.recall_count or bullet.hit_count):
                # "(used N×, success Y/Z)" surfaces reinforcement signal to the consumer.
                # Denominator floors at max(recall_count, hit_count) so we never render
                # "success 3/0" when hit_count was incremented out-of-band from recall.
                denom = max(bullet.recall_count, bullet.hit_count)
                usage_stats[bullet.content] = (
                    f"(used {bullet.recall_count}×, success {bullet.hit_count}/{denom})"
                )
            total_tokens += est

        rendered = renderer.render(
            selected_concepts, intent, token_budget,
            core_memory=core_memory,
            worked_examples=worked_examples or None,
            usage_stats=usage_stats or None,
        )
        actual_tokens = renderer.estimate_tokens(rendered)
        coverage = len(selected_bullet_ids) / len(bullets) if bullets else 0.0

        return {
            "materialization_id": mat_id,
            "rendered_text": rendered,
            "bullets_included": selected_bullet_ids,
            "schemas_included": schema_ids,
            "concepts_included": [],
            "token_count": actual_tokens,
            "coverage_score": round(coverage, 3),
        }

    async def _materialize_concepts(
        self, context_id: uuid.UUID, concepts: list[ConceptNode],
        intent: IntentAnchor | None, search_text: str,
        focus_domains: list[str], recency_weight: float,
        include_decisions: bool, token_budget: int,
        target_model: str, renderer: ContextRenderer,
        core_memory: str = "",
        worked_examples: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Legacy concept-based materialization."""
        mat_id = str(uuid.uuid4())
        scores = await self._score_concepts(
            context_id, concepts, search_text, focus_domains,
            recency_weight, include_decisions,
        )
        edges = await self.storage.get_edges(context_id)
        scores = self._spreading_activation(scores, edges)

        sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        id_to_concept = {c.id: c for c in concepts}
        selected: list[ConceptNode] = []
        total_tokens = 0

        intent_budget = 0
        if intent or core_memory:
            intent_text = renderer.render([], intent, token_budget, core_memory=core_memory)
            intent_budget = renderer.estimate_tokens(intent_text)
        remaining = token_budget - intent_budget

        for concept_id, score in sorted_concepts:
            concept = id_to_concept.get(concept_id)
            if concept is None:
                continue
            est = renderer.estimate_tokens(concept.content) + 10
            if total_tokens + est > remaining:
                break
            selected.append(concept)
            total_tokens += est

        rendered = renderer.render(
            selected, intent, token_budget,
            core_memory=core_memory,
            worked_examples=worked_examples or None,
        )
        actual_tokens = renderer.estimate_tokens(rendered)
        coverage = len(selected) / len(concepts) if concepts else 0.0

        return {
            "materialization_id": mat_id,
            "rendered_text": rendered,
            "bullets_included": [],
            "schemas_included": [],
            "concepts_included": [c.id for c in selected],
            "token_count": actual_tokens,
            "coverage_score": round(coverage, 3),
        }

    async def _score_bullets(
        self, context_id: str, bullets: list[Bullet],
        search_text: str, recency_weight: float, include_decisions: bool,
    ) -> dict[str, float]:
        """Score bullets by effective_salience, relevance, and recency."""
        scores: dict[str, float] = {}
        embedding_scores: dict[str, float] = {}

        if search_text:
            try:
                query_embedding = await self.llm.embed(search_text)
                similar = await self.storage.find_similar_bullets(
                    context_id, query_embedding, limit=100, threshold=0.3,
                )
                for bullet, sim in similar:
                    embedding_scores[bullet.id] = sim
            except Exception as exc:
                logger.warning("Embedding search failed: %s", exc)

        now = _utcnow()

        for bullet in bullets:
            relevance = embedding_scores.get(bullet.id, 0.1 if not search_text else 0.0)
            eff_salience = bullet.effective_salience
            age_days = max(0.001, (now - bullet.created_at).total_seconds() / 86400)
            recency_score = math.exp(-0.05 * age_days)
            decision_boost = 0.3 if (
                include_decisions and
                (bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else bullet.bullet_type) == "decision"
            ) else 0.0

            combined = (
                relevance * (1 - recency_weight)
                + eff_salience * 0.3
                + recency_score * recency_weight
                + decision_boost
            )
            scores[bullet.id] = combined

        return scores

    async def _score_concepts(
        self, context_id: uuid.UUID, concepts: list[ConceptNode],
        search_text: str, focus_domains: list[str],
        recency_weight: float, include_decisions: bool,
    ) -> dict[uuid.UUID, float]:
        """Score legacy concepts."""
        scores: dict[uuid.UUID, float] = {}
        embedding_scores: dict[uuid.UUID, float] = {}

        if search_text:
            try:
                query_embedding = await self.llm.embed(search_text)
                similar = await self.storage.find_similar_concepts(
                    context_id, query_embedding, limit=100, threshold=0.3,
                )
                for concept, sim in similar:
                    embedding_scores[concept.id] = sim
            except Exception:
                pass

        now = _utcnow()
        focus_set = set(focus_domains)

        for concept in concepts:
            relevance = embedding_scores.get(concept.id, 0.1 if not search_text else 0.0)
            if focus_set and concept.domain_tags:
                overlap = len(focus_set.intersection(set(concept.domain_tags)))
                relevance += 0.2 * overlap
            age_days = max(0.001, (now - concept.created_at).total_seconds() / 86400)
            recency_score = math.exp(-0.05 * age_days)
            decision_boost = 0.3 if include_decisions and concept.type == ConceptType.DECISION else 0.0

            combined = (
                relevance * (1 - recency_weight)
                + concept.salience * 0.3
                + recency_score * recency_weight
                + decision_boost
            )
            scores[concept.id] = combined

        return scores

    async def _fetch_worked_examples(
        self,
        context_id: str,
        search_text: str,
        threshold: float,
        limit: int,
    ) -> list[dict[str, str]]:
        """DC-inspired nearest-prior-input retrieval.

        Embeds the query, finds the closest prior activities (≥ threshold cosine),
        and returns up to `limit` worked examples each containing the original
        raw_input and the bullets it produced. Known-recoverable errors (LLM /
        storage failures) are caught and logged so materialization still
        succeeds; unexpected errors propagate.
        """
        try:
            query_embedding = await self.llm.embed(search_text)
        except LLMAdapterError as exc:
            logger.warning("Worked-example query embedding failed: %s", exc)
            return []

        try:
            hits = await self.storage.find_similar_activities(
                context_id, query_embedding,
                limit=limit, threshold=threshold,
            )
        except StorageError as exc:
            logger.warning("find_similar_activities failed: %s", exc)
            return []

        examples: list[dict[str, str]] = []
        for activity, sim in hits:
            if not activity.raw_input:
                continue
            bullets_text = ""
            if activity.bullet_ids_produced:
                try:
                    produced = await self.storage.get_bullets_by_ids(
                        context_id, activity.bullet_ids_produced,
                    )
                    bullets_text = "; ".join(b.content for b in produced)
                except StorageError as exc:
                    logger.warning(
                        "Worked-example bullet fetch failed for activity %s: %s",
                        activity.id, exc,
                    )
                    bullets_text = ""
            examples.append({
                "input": _truncate(activity.raw_input, WORKED_EXAMPLE_INPUT_MAX_CHARS),
                "output": _truncate(bullets_text, WORKED_EXAMPLE_OUTPUT_MAX_CHARS),
                "similarity": f"{sim:.2f}",
            })
        return examples

    @staticmethod
    def _spreading_activation(
        scores: dict[uuid.UUID, float], edges: list[ConceptEdge],
        decay: float = 0.5, iterations: int = 2,
    ) -> dict[uuid.UUID, float]:
        adjacency: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.from_node].append((edge.to_node, edge.weight))
            adjacency[edge.to_node].append((edge.from_node, edge.weight))
        for _ in range(iterations):
            additions: dict[uuid.UUID, float] = defaultdict(float)
            for node_id, score in scores.items():
                for neighbor_id, weight in adjacency.get(node_id, []):
                    additions[neighbor_id] += score * weight * decay
            for node_id, boost in additions.items():
                scores[node_id] = scores.get(node_id, 0.0) + boost
        return scores

    @staticmethod
    def _get_renderer(target_model: str) -> ContextRenderer:
        target_lower = target_model.lower()
        for key, renderer_cls in RENDERER_MAP.items():
            if target_lower.startswith(key):
                return renderer_cls()
        return GenericRenderer()

    @staticmethod
    def _bullet_type_to_concept_type(bullet_type: str) -> ConceptType:
        mapping = {
            "strategy": ConceptType.PROCEDURE,
            "warning": ConceptType.CONSTRAINT,
            "fact": ConceptType.FACT,
            "procedure": ConceptType.PROCEDURE,
            "exception": ConceptType.EXCEPTION,
            "principle": ConceptType.PATTERN,
            "decision": ConceptType.DECISION,
        }
        return mapping.get(bullet_type, ConceptType.FACT)
