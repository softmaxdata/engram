"""Delta operation engine — all graph mutations go through here.

Every mutation to the concept graph is expressed as atomic delta operations.
The full context is NEVER regenerated wholesale. This prevents context collapse
and enables full auditability + rollback.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from engram.core.exceptions import CapacityExceededError
from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    BulletType,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    LifecycleState,
    SchemaNode,
    SourceType,
    cap_core_memory,
)
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@asynccontextmanager
async def capacity_checked_transaction(storage: StorageBackend, context_id: str):
    """Atomically validate actual capacity around a public mutation and ledger."""
    async with storage.transaction(context_id) as tx:
        try:
            context_uuid = uuid.UUID(context_id)
        except (ValueError, AttributeError):
            context = None
        else:
            context = await tx.get_context(context_uuid)
        before_count = 0
        max_bullets = None
        if context is not None:
            max_bullets = context.lifecycle_config.max_active_bullets
            before = await tx.get_capacity_status(context_id, max_bullets)
            before_count = before.active_bullet_count
        yield tx
        if max_bullets is not None:
            after = await tx.get_capacity_status(context_id, max_bullets)
            # Existing overfull contexts may still be corrected or reduced.
            # Missing/duplicate removes do not create fictitious capacity.
            if (after.active_bullet_count > max_bullets
                    and after.active_bullet_count > before_count):
                raise CapacityExceededError(context_id, after.active_bullet_count, max_bullets)


class DeltaEngine:
    """Applies delta operations atomically to the concept graph.

    ALL mutations to the graph MUST go through this engine.
    No direct writes to bullets or edges that bypass delta tracking.
    """

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    async def apply_batch(self, batch: DeltaBatch) -> DeltaBatch:
        # Work on a copy so a failed attempt cannot poison retry snapshots/ids.
        working = batch.model_copy(deep=True)
        async with self.storage.transaction(batch.context_id) as tx:
            result = await DeltaEngine(tx)._apply_batch(working)
        # Successful calls historically update the caller's batch and operation
        # objects. Publish only after commit so failures remain safe to retry.
        for original, applied in zip(batch.operations, result.operations):
            for field in type(applied).model_fields:
                setattr(original, field, getattr(applied, field))
        for field in type(result).model_fields:
            if field != "operations":
                setattr(batch, field, getattr(result, field))
        return batch

    async def _validate_operation_context(self, context_id: str, op: DeltaOperation) -> None:
        """A batch lock is not authorization to write globally addressed rows."""
        if op.op_type in (DeltaOpType.ADD_SCHEMA, DeltaOpType.UPDATE_SCHEMA):
            schema = await self.storage.get_schema(op.target_id) if op.target_id else None
            if schema is not None and schema.context_id != context_id:
                raise ValueError("Delta target does not belong to the batch context")
            return
        targets = op.target_ids if op.op_type == DeltaOpType.MERGE_BULLETS else [op.target_id]
        for target in dict.fromkeys(targets or []):
            if target:
                bullet = await self.storage.get_bullet(target)
                if bullet is not None and bullet.context_id != context_id:
                    raise ValueError("Delta target does not belong to the batch context")

    async def _apply_batch(self, batch: DeltaBatch) -> DeltaBatch:
        """Apply a batch of delta operations atomically.

        Returns the batch with updated stats.
        """
        added = 0
        updated = 0
        removed = 0
        merged = 0

        for op in batch.operations:
            await self._validate_operation_context(batch.context_id, op)
            match op.op_type:
                case DeltaOpType.ADD_BULLET:
                    await self._apply_add_bullet(batch.context_id, op)
                    added += 1

                case DeltaOpType.UPDATE_BULLET:
                    await self._apply_update_bullet(op)
                    updated += 1

                case DeltaOpType.REMOVE_BULLET:
                    await self._apply_remove_bullet(op)
                    removed += 1

                case DeltaOpType.MERGE_BULLETS:
                    await self._apply_merge_bullets(batch.context_id, op)
                    merged += 1

                case DeltaOpType.ADD_SCHEMA:
                    await self._apply_add_schema(batch.context_id, op)

                case DeltaOpType.UPDATE_SCHEMA:
                    await self._apply_update_schema(op)

                case DeltaOpType.UPDATE_CORE_MEMORY:
                    await self._apply_update_core_memory(batch.context_id, op)

                case DeltaOpType.RECONSOLIDATE_BULLET:
                    await self._apply_reconsolidate_bullet(op)

                case _:
                    logger.warning("Unknown delta op type: %s", op.op_type)

        batch.bullets_added = added
        batch.bullets_updated = updated
        batch.bullets_removed = removed
        batch.bullets_merged = merged

        # Persist the batch for auditability
        await self.storage.save_delta_batch(batch)

        return batch

    async def rollback_batch(self, delta_batch_id: str) -> bool:
        batch = await self.storage.get_delta_batch(delta_batch_id)
        if batch is None:
            return False
        async with self.storage.transaction(batch.context_id) as tx:
            return await DeltaEngine(tx)._rollback_batch(delta_batch_id)

    async def _rollback_batch(self, delta_batch_id: str) -> bool:
        """Roll back a delta batch by applying inverse operations."""
        batch = await self.storage.get_delta_batch(delta_batch_id)
        if batch is None:
            return False

        # Preflight every inverse before any write. Historical records may
        # lack generated identities or snapshots; partial rollback is unsafe.
        for op in batch.operations:
            await self._validate_operation_context(batch.context_id, op)
            snapshot = op.rollback_state or op.previous_state
            if snapshot is not None and snapshot.get("noop") is True:
                continue
            if op.op_type == DeltaOpType.ADD_BULLET:
                if not op.target_id:
                    return False
            elif op.op_type == DeltaOpType.UPDATE_BULLET:
                if not op.target_id or not snapshot or not any(
                    field in snapshot
                    for field in ("content", "embedding", "salience", "confidence", "section")
                ):
                    return False
            elif op.op_type == DeltaOpType.REMOVE_BULLET:
                if not op.target_id or snapshot is None or "is_active" not in snapshot:
                    return False
            elif op.op_type == DeltaOpType.MERGE_BULLETS:
                if snapshot is None or not isinstance(snapshot.get("bullets"), dict):
                    return False
            elif op.op_type == DeltaOpType.ADD_SCHEMA:
                if not op.target_id or not snapshot or snapshot.get("schema_created") is not True:
                    return False
            elif op.op_type == DeltaOpType.UPDATE_SCHEMA:
                if not op.target_id or snapshot is None or "description" not in snapshot:
                    return False
            elif op.op_type == DeltaOpType.UPDATE_CORE_MEMORY:
                if snapshot is None or "core_memory" not in snapshot:
                    return False
            elif op.op_type == DeltaOpType.RECONSOLIDATE_BULLET:
                if not op.target_id or snapshot is None or not all(
                    field in snapshot for field in ("recall_count", "hit_count", "miss_count", "salience")
                ):
                    return False

        # Apply inverse operations in reverse order. Newer ops use rollback_state
        # for the snapshot; older records (pre-rollback_state) fall back to
        # previous_state for backward compatibility.
        for op in reversed(batch.operations):
            await self._validate_operation_context(batch.context_id, op)
            snapshot = op.rollback_state or op.previous_state
            if snapshot is not None and snapshot.get("noop") is True:
                continue
            # ADD has no previous state. Its persisted target identifies the
            # new bullet even for callers that did not provide an id.
            if op.op_type == DeltaOpType.ADD_BULLET:
                if op.target_id:
                    await self.storage.remove_bullet(op.target_id)
                continue
            if op.op_type == DeltaOpType.ADD_SCHEMA:
                if op.content and op.target_id:
                    await self.storage.remove_schema(batch.context_id, op.target_id)
                continue
            snapshot = op.rollback_state or op.previous_state
            if snapshot is None:
                continue

            match op.op_type:
                case DeltaOpType.UPDATE_BULLET:
                    # Undo update → restore previous state
                    if op.target_id:
                        bullet = await self.storage.get_bullet(op.target_id)
                        if bullet:
                            restored_content = snapshot.get("content", bullet.content)
                            if "embedding" in snapshot or restored_content != bullet.content:
                                # Legacy snapshots lack vectors; clear the current
                                # one if it describes the text being undone.
                                bullet.embedding = snapshot.get("embedding")
                            bullet.content = restored_content
                            bullet.salience = snapshot.get("salience", bullet.salience)
                            bullet.confidence = snapshot.get("confidence", bullet.confidence)
                            bullet.section = snapshot.get("section", bullet.section)
                            await self.storage.update_bullet(bullet)

                case DeltaOpType.REMOVE_BULLET:
                    # Undo remove → reactivate
                    if op.target_id:
                        bullet = await self.storage.get_bullet(op.target_id)
                        if bullet:
                            bullet.is_active = snapshot.get("is_active", True)
                            await self.storage.update_bullet(bullet)

                case DeltaOpType.UPDATE_CORE_MEMORY:
                    prev_core = snapshot.get("core_memory", "")
                    await self.storage.update_core_memory(batch.context_id, prev_core)

                case DeltaOpType.RECONSOLIDATE_BULLET:
                    if op.target_id:
                        bullet = await self.storage.get_bullet(op.target_id)
                        if bullet:
                            bullet.recall_count = int(snapshot.get("recall_count", bullet.recall_count))
                            bullet.hit_count = int(snapshot.get("hit_count", bullet.hit_count))
                            bullet.miss_count = int(snapshot.get("miss_count", bullet.miss_count))
                            bullet.salience = float(snapshot.get("salience", bullet.salience))
                            lr = snapshot.get("last_recalled_at")
                            bullet.last_recalled_at = (
                                datetime.fromisoformat(lr) if isinstance(lr, str) else None
                            )
                            await self.storage.update_bullet(bullet)

                case DeltaOpType.UPDATE_SCHEMA:
                    if op.target_id and "description" in snapshot:
                        schema = await self.storage.get_schema(op.target_id)
                        if schema is not None:
                            schema.description = snapshot["description"]
                            await self.storage.update_schema(schema)

                case DeltaOpType.MERGE_BULLETS:
                    for bullet_id, previous in snapshot.get("bullets", {}).items():
                        bullet = await self.storage.get_bullet(bullet_id)
                        if bullet is None:
                            continue
                        if bullet.context_id != batch.context_id:
                            raise ValueError("Delta snapshot does not belong to the batch context")
                        for field in (
                            "content", "embedding", "recall_count", "hit_count",
                            "miss_count", "salience", "is_active",
                        ):
                            if field in previous:
                                setattr(bullet, field, previous[field])
                        await self.storage.update_bullet(bullet)

        return True

    async def _apply_add_bullet(self, context_id: str, op: DeltaOperation) -> None:
        # Map DeltaSource → SourceType (they have different enum values)
        source_map = {
            DeltaSource.REFLECTOR: SourceType.REFLECTION,
            DeltaSource.CURATOR: SourceType.REFLECTION,
            DeltaSource.USER: SourceType.USER_INPUT,
            DeltaSource.CONSOLIDATION: SourceType.CONSOLIDATION,
        }
        source_type = source_map.get(op.source, SourceType.REFLECTION) if op.source else SourceType.REFLECTION

        op.target_id = op.target_id or str(uuid.uuid4())[:8]
        bullet = Bullet(
            id=op.target_id,
            section=op.section or "general",
            content=op.content or "",
            embedding=op.embedding,
            bullet_type=BulletType(op.bullet_type) if op.bullet_type else BulletType.FACT,
            source_type=source_type,
            salience=op.confidence,
            confidence=op.confidence,
            source_session=op.session_id,
            source_agent=op.agent_id,
        )
        await self.storage.add_bullet(context_id, bullet)

    async def _apply_update_bullet(self, op: DeltaOperation) -> None:
        if not op.target_id:
            op.rollback_state = {"noop": True}
            return
        bullet = await self.storage.get_bullet(op.target_id)
        if bullet is None:
            op.rollback_state = {"noop": True}
            return

        # Each committed application needs its own inverse. Failed attempts
        # operate on a copy, so they cannot overwrite the caller's snapshot.
        op.rollback_state = {
            "content": bullet.content,
            "embedding": bullet.embedding,
            "salience": bullet.salience,
            "confidence": bullet.confidence,
            "section": bullet.section,
        }

        if op.content is not None:
            if op.content != bullet.content:
                # A missing new embedding must not leave an old-content vector.
                bullet.embedding = op.embedding
            bullet.content = op.content
        if op.embedding is not None:
            bullet.embedding = op.embedding
        if op.section is not None:
            bullet.section = op.section
        if op.confidence is not None:
            bullet.confidence = op.confidence
        await self.storage.update_bullet(bullet)

    async def _apply_remove_bullet(self, op: DeltaOperation) -> None:
        if not op.target_id:
            op.rollback_state = {"noop": True}
            return
        bullet = await self.storage.get_bullet(op.target_id)
        if bullet:
            op.rollback_state = {"is_active": bullet.is_active}
            await self.storage.remove_bullet(op.target_id)
        else:
            op.rollback_state = {"noop": True}

    async def _apply_merge_bullets(self, context_id: str, op: DeltaOperation) -> None:
        """Merge multiple bullets into one — keep the most specific/highest salience."""
        # Mark even a no-op merge as known; older records lack inverse evidence.
        op.rollback_state = {"bullets": {}}
        if not op.target_ids or len(op.target_ids) < 2:
            return

        bullets_to_merge = []
        for bid in dict.fromkeys(op.target_ids):
            bullet = await self.storage.get_bullet(bid)
            if (bullet is not None and bullet.is_active and not bullet.is_archived
                    and bullet.lifecycle_state == LifecycleState.ACTIVE):
                bullets_to_merge.append(bullet)

        if len(bullets_to_merge) < 2:
            return

        op.rollback_state = {
            "bullets": {
                bullet.id: {
                    field: getattr(bullet, field)
                    for field in (
                        "content", "embedding", "recall_count", "hit_count",
                        "miss_count", "salience", "is_active",
                    )
                }
                for bullet in bullets_to_merge
            }
        }

        # Keep the bullet with highest salience, deactivate others
        best = max(bullets_to_merge, key=lambda b: b.salience)
        if op.content:
            if op.content != best.content:
                best.embedding = op.embedding
            best.content = op.content
        best.recall_count = sum(b.recall_count for b in bullets_to_merge)
        best.hit_count = sum(b.hit_count for b in bullets_to_merge)
        best.miss_count = sum(b.miss_count for b in bullets_to_merge)
        best.salience = max(b.salience for b in bullets_to_merge)
        await self.storage.update_bullet(best)

        for b in bullets_to_merge:
            if b.id != best.id:
                await self.storage.remove_bullet(b.id)

    async def _apply_add_schema(self, context_id: str, op: DeltaOperation) -> None:
        if not op.content:
            op.rollback_state = {"noop": True}
            return
        op.target_id = op.target_id or str(uuid.uuid4())[:8]
        schema = SchemaNode(
            id=op.target_id,
            name=op.content,
            description=op.reasoning,
        )
        await self.storage.add_schema(context_id, schema)
        op.rollback_state = {"schema_created": True}

    async def _apply_update_schema(self, op: DeltaOperation) -> None:
        if not op.target_id:
            op.rollback_state = {"noop": True}
            return
        schema = await self.storage.get_schema(op.target_id)
        if schema and op.content:
            op.rollback_state = {"description": schema.description}
            schema.description = op.content
            await self.storage.update_schema(schema)
        else:
            op.rollback_state = {"noop": True}

    async def _apply_update_core_memory(
        self, context_id: str, op: DeltaOperation,
    ) -> None:
        """Replace the always-in-context core memory blob.

        The new value is in op.content; capture the previous value into
        rollback_state so a retry doesn't overwrite the original snapshot.
        """
        if op.content is None:
            op.rollback_state = {"noop": True}
            return
        # Enforce the ≤512-token bound here, at the canonical mutation point, so
        # it holds for every caller of the delta op — not just the ingestion
        # commit path. cap_core_memory is idempotent on already-capped text.
        capped = cap_core_memory(op.content)
        try:
            ctx = await self.storage.get_context(uuid.UUID(context_id))
        except (ValueError, AttributeError):
            ctx = None
        if ctx is None:
            op.rollback_state = {"noop": True}
            return
        op.rollback_state = {"core_memory": ctx.core_memory}
        await self.storage.update_core_memory(context_id, capped)

    async def _apply_reconsolidate_bullet(self, op: DeltaOperation) -> None:
        """Audit-clean reconsolidation: update a bullet's usage stats and salience.

        op.previous_state — set by caller — carries the deltas to apply:
          {"recall_delta": int, "hit_delta": int, "miss_delta": int,
           "salience_multiplier": float, "outcome": "success|failure|partial"}
        Rollback snapshot is written to op.rollback_state (separate field) so
        a retry/idempotent re-apply reads the same deltas, not the snapshot.
        """
        if not op.target_id or op.previous_state is None:
            op.rollback_state = {"noop": True}
            return
        bullet = await self.storage.get_bullet(op.target_id)
        if bullet is None:
            op.rollback_state = {"noop": True}
            return
        # Don't reinforce a bullet that was archived or deactivated between the
        # materialization and this feedback commit — get_bullet returns rows
        # regardless of lifecycle state, so revalidate here.
        if not bullet.is_active or bullet.is_archived:
            logger.debug(
                "Skipping reconsolidation of inactive/archived bullet %s",
                op.target_id,
            )
            op.rollback_state = {"noop": True}
            return
        deltas = op.previous_state
        op.rollback_state = {
            "recall_count": bullet.recall_count,
            "hit_count": bullet.hit_count,
            "miss_count": bullet.miss_count,
            "salience": bullet.salience,
            "last_recalled_at": bullet.last_recalled_at.isoformat()
                if bullet.last_recalled_at else None,
        }
        bullet.recall_count += int(deltas.get("recall_delta", 0))
        bullet.hit_count += int(deltas.get("hit_delta", 0))
        bullet.miss_count += int(deltas.get("miss_delta", 0))
        mult = float(deltas.get("salience_multiplier", 1.0))
        bullet.salience = max(0.05, min(1.0, bullet.salience * mult))
        bullet.last_recalled_at = _utcnow()
        await self.storage.update_bullet(bullet)
