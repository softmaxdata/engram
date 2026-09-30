"""Consolidation engine — background "sleep cycle" that reorganizes the concept graph.

Brain analogy: During sleep, the hippocampus replays recent experiences while
the neocortex reorganizes — abstracting patterns, strengthening important
connections, weakening irrelevant ones, and forming schemas.

v0.3: Adds aggressive mode at high capacity, purge phase for expired archives,
lock-protected consolidation, and event emission.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from engram.core.concurrency import ContextLockManager
from engram.core.delta import DeltaEngine
from engram.core.events import EventBus
from engram.core.models import (
    Bullet,
    BulletType,
    ConsolidationConfig,
    ConsolidationReport,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    LifecycleConfig,
    SchemaNode,
)
from engram.llm.adapter import LLMAdapter
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ConsolidationEngine:
    """Background process that reorganizes the concept graph.

    Runs periodically: every N commits, on a timer, or manually triggered.
    v0.3: Optional lock_manager and event_bus for concurrency and notifications.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm: LLMAdapter,
        lock_manager: ContextLockManager | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.lock_manager = lock_manager
        self.event_bus = event_bus
        self.delta_engine = DeltaEngine(storage)

    async def consolidate(
        self, context_id: str, config: ConsolidationConfig | None = None
    ) -> ConsolidationReport:
        """Full consolidation cycle."""
        if config is None:
            config = ConsolidationConfig()

        start_time = time.monotonic()
        report = ConsolidationReport(context_id=context_id)

        # v0.3: Check capacity and switch to aggressive mode if needed
        lifecycle_config = LifecycleConfig()
        try:
            import uuid as _uuid
            context = await self.storage.get_context(_uuid.UUID(context_id))
            if context is not None:
                lifecycle_config = context.lifecycle_config
        except (ValueError, Exception):
            pass

        capacity = await self.storage.get_capacity_status(
            context_id, lifecycle_config.max_active_bullets
        )
        if capacity.pressure_level in ("high", "critical"):
            config = self._make_aggressive(config, lifecycle_config)
            report.mode = "aggressive"
            logger.info(
                "Consolidation for %s: switching to aggressive mode (pressure=%s, %d/%d)",
                context_id, capacity.pressure_level,
                capacity.active_bullet_count, capacity.max_active_bullets,
            )

        # Phase 1: Forgetting curve
        report.decayed = await self._apply_forgetting_curve(context_id, config)

        # Phase 2: Semantic deduplication
        report.deduplicated = await self._semantic_dedup(context_id, config)

        # Phase 3: Schema induction (LLM required)
        report.schemas_formed = await self._induce_schemas(context_id, config)

        # Phase 4: Archive stale bullets
        report.archived = await self._archive_stale(context_id, config)

        # Phase 5: Purge expired archives
        report.purged = await self._purge_expired(context_id, lifecycle_config)

        # Phase 6: Promote observations to principles
        report.promoted = await self._promote_to_principles(context_id, config)

        report.duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "Consolidation for %s [%s]: decayed=%d, deduped=%d, schemas=%d, archived=%d, "
            "purged=%d, promoted=%d, took=%dms",
            context_id, report.mode, report.decayed, report.deduplicated,
            report.schemas_formed, report.archived, report.purged,
            report.promoted, report.duration_ms,
        )

        # v0.3: Emit event
        if self.event_bus is not None:
            self.event_bus.emit(
                context_id,
                event_type="consolidation_ran",
                data={
                    "mode": report.mode,
                    "decayed": report.decayed,
                    "deduplicated": report.deduplicated,
                    "archived": report.archived,
                    "purged": report.purged,
                    "duration_ms": report.duration_ms,
                },
            )

        return report

    async def _apply_forgetting_curve(
        self, context_id: str, config: ConsolidationConfig
    ) -> int:
        """Ebbinghaus forgetting curve — unused memories decay.

        Concepts that are never recalled decay in salience over time.
        Concepts that ARE recalled get their decay reset (spaced repetition).
        DECISION and SCHEMA bullets have slower decay rates.
        """
        bullets = await self.storage.list_bullets(context_id, include_archived=False)
        decayed_count = 0
        now = _utcnow()

        for bullet in bullets:
            reference_time = bullet.last_recalled_at or bullet.created_at
            days_since_active = max(0, (now - reference_time).total_seconds() / 86400)

            if days_since_active <= 1:
                continue

            bt = bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else str(bullet.bullet_type)
            if bt in ("decision", "principle"):
                decay_rate = config.slow_decay_rate
            elif bt == "exception":
                decay_rate = 0.99
            else:
                decay_rate = config.fast_decay_rate

            new_salience = bullet.salience * (decay_rate ** days_since_active)
            new_salience = max(config.min_salience, new_salience)

            if abs(new_salience - bullet.salience) > 0.001:
                bullet.salience = new_salience
                await self.storage.update_bullet(bullet)
                decayed_count += 1

        return decayed_count

    async def _semantic_dedup(
        self, context_id: str, config: ConsolidationConfig
    ) -> int:
        """Merge near-duplicate bullets using embedding similarity.

        Fast path — no LLM call needed. Pure embedding comparison.
        If cosine similarity > threshold (default 0.92), merge by
        keeping the more specific/higher-salience bullet.
        """
        bullets = await self.storage.list_bullets(context_id, include_archived=False)
        merged_ids: set[str] = set()
        merge_count = 0

        for i, bullet_a in enumerate(bullets):
            if bullet_a.id in merged_ids or bullet_a.embedding is None:
                continue
            for bullet_b in bullets[i + 1:]:
                if bullet_b.id in merged_ids or bullet_b.embedding is None:
                    continue
                sim = _cosine_similarity(bullet_a.embedding, bullet_b.embedding)
                if sim >= config.dedup_threshold:
                    # Keep the one with higher salience
                    keep, remove = (bullet_a, bullet_b) if bullet_a.salience >= bullet_b.salience else (bullet_b, bullet_a)
                    keep.recall_count += remove.recall_count
                    keep.hit_count += remove.hit_count
                    keep.miss_count += remove.miss_count
                    keep.salience = max(keep.salience, remove.salience)
                    await self.storage.update_bullet(keep)
                    await self.storage.remove_bullet(remove.id)
                    merged_ids.add(remove.id)
                    merge_count += 1

        return merge_count

    async def _induce_schemas(
        self, context_id: str, config: ConsolidationConfig
    ) -> int:
        """Detect recurring patterns and form abstract SchemaNodes.

        Uses section grouping to find clusters, then LLM to name/describe.
        Requires schema_min_instances (default 3) bullets in the same section.
        """
        bullets = await self.storage.list_bullets(context_id, include_archived=False)
        existing_schemas = await self.storage.list_schemas(context_id)
        existing_schema_sections = {s.name for s in existing_schemas}

        # Group by section
        section_bullets: dict[str, list[Bullet]] = {}
        for bullet in bullets:
            if bullet.schema_id is None:
                section_bullets.setdefault(bullet.section, []).append(bullet)

        schemas_formed = 0
        for section, section_list in section_bullets.items():
            if section in existing_schema_sections:
                continue
            if len(section_list) < config.schema_min_instances:
                continue

            # Form a schema from this cluster
            bullet_contents = [b.content for b in section_list[:10]]
            try:
                description = await self.llm.complete(
                    prompt=(
                        f"These are related knowledge bullets from the '{section}' section:\n"
                        + "\n".join(f"- {c}" for c in bullet_contents)
                        + "\n\nDescribe the abstract pattern they share in 1-2 sentences."
                    ),
                    system="You are a knowledge organizer. Respond with a concise description only.",
                    temperature=0.0,
                    max_tokens=200,
                )
            except Exception:
                description = f"Pattern in {section} ({len(section_list)} bullets)"

            schema = SchemaNode(
                name=section,
                description=description.strip(),
                instance_count=len(section_list),
                confidence=min(1.0, len(section_list) / 10),
                bullet_ids=[b.id for b in section_list],
            )
            await self.storage.add_schema(context_id, schema)

            # Link bullets to schema
            for bullet in section_list:
                bullet.schema_id = schema.id
                await self.storage.update_bullet(bullet)

            schemas_formed += 1

        return schemas_formed

    async def _archive_stale(
        self, context_id: str, config: ConsolidationConfig
    ) -> int:
        """Move very low-salience bullets to cold storage.

        Not deletion — archived bullets can be retrieved if needed.
        Threshold: salience < archive_salience_threshold AND not recalled
        in archive_days_threshold+ days AND not a DECISION or PRINCIPLE type.
        v0.3: Uses storage.archive_bullet() for proper lifecycle state transitions.
        """
        bullets = await self.storage.list_bullets(context_id, include_archived=False)
        archived_count = 0
        now = _utcnow()

        for bullet in bullets:
            bt = bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else str(bullet.bullet_type)
            if bt in ("decision", "principle"):
                continue

            if bullet.salience >= config.archive_salience_threshold:
                continue

            reference_time = bullet.last_recalled_at or bullet.created_at
            days_inactive = (now - reference_time).total_seconds() / 86400
            if days_inactive < config.archive_days_threshold:
                continue

            success = await self.storage.archive_bullet(
                context_id, bullet.id, reason="consolidation_stale"
            )
            if success:
                archived_count += 1

        return archived_count

    async def _promote_to_principles(
        self, context_id: str, config: ConsolidationConfig
    ) -> int:
        """When multiple concrete observations support the same conclusion,
        promote to a general principle.

        Brain analogy: Episodic → semantic transformation.
        """
        # Find groups of similar FACT/OBSERVATION bullets (3+ in same section with similar content)
        bullets = await self.storage.list_bullets(
            context_id, bullet_type="fact", include_archived=False,
        )
        if len(bullets) < config.schema_min_instances:
            return 0

        # Group by section
        section_facts: dict[str, list[Bullet]] = {}
        for bullet in bullets:
            if bullet.hit_count >= 2:  # Only promote well-tested facts
                section_facts.setdefault(bullet.section, []).append(bullet)

        promoted = 0
        for section, facts in section_facts.items():
            if len(facts) < config.schema_min_instances:
                continue

            # Check for existing principles in this section
            existing_principles = await self.storage.list_bullets(
                context_id, section=section, bullet_type="principle",
            )
            if existing_principles:
                continue

            # Use LLM to synthesize a principle
            fact_contents = [f.content for f in facts[:8]]
            try:
                principle = await self.llm.complete(
                    prompt=(
                        "These related observations have been confirmed through repeated use:\n"
                        + "\n".join(f"- {c}" for c in fact_contents)
                        + "\n\nSynthesize a single general principle (1-2 sentences)."
                    ),
                    system="You are a knowledge synthesizer. Respond with the principle only.",
                    temperature=0.0,
                    max_tokens=200,
                )
            except Exception:
                continue

            # Create principle bullet
            principle_bullet = Bullet(
                section=section,
                content=principle.strip(),
                bullet_type=BulletType.PRINCIPLE,
                source_type="consolidation",
                salience=0.8,
                confidence=min(1.0, len(facts) / 10),
            )
            await self.storage.add_bullet(context_id, principle_bullet)
            promoted += 1

        return promoted

    @staticmethod
    def _make_aggressive(
        config: ConsolidationConfig, lifecycle: LifecycleConfig
    ) -> ConsolidationConfig:
        """Return an aggressive consolidation config for high-pressure situations."""
        return ConsolidationConfig(
            fast_decay_rate=config.fast_decay_rate,
            slow_decay_rate=config.slow_decay_rate,
            min_salience=config.min_salience,
            dedup_threshold=lifecycle.aggressive_dedup_threshold,
            schema_min_instances=lifecycle.aggressive_schema_min_instances,
            archive_salience_threshold=lifecycle.aggressive_archive_salience,
            archive_days_threshold=max(7, config.archive_days_threshold // 2),
            consolidation_trigger=config.consolidation_trigger,
        )

    async def _purge_expired(
        self, context_id: str, lifecycle: LifecycleConfig
    ) -> int:
        """Permanently delete archived bullets past the purge threshold."""
        return await self.storage.purge_expired_archives(
            context_id, lifecycle.purge_after_days
        )
