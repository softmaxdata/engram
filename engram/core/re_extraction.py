"""Re-extraction engine — recompile bullets from raw input history.

v0.4: Enables reprocessing of historical raw inputs with a new/better
Reflector model. Like upgrading a compiler and recompiling from source.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from engram.core.concurrency import ContextLockManager
from engram.core.events import EventBus
from engram.core.ingestion import CuratorEngine, ReflectorEngine
from engram.core.models import (
    ActionType,
    Activity,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    DeltaSource,
    ExecutionFeedback,
    ReExtractionPreview,
    ReExtractionRequest,
    ReExtractionResult,
)
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class ReExtractionEngine:
    """Recompile bullets from raw input history with a new/better Reflector.

    Analogy: This is like upgrading your compiler and recompiling from source.
    The raw inputs (source code) are immutable. The bullets (compiled output)
    get regenerated. The concept graph evolves to reflect better extraction.
    """

    def __init__(
        self,
        reflector: ReflectorEngine,
        curator: CuratorEngine,
        storage: StorageBackend,
        lock_manager: ContextLockManager | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.reflector = reflector
        self.curator = curator
        self.storage = storage
        self.lock_manager = lock_manager
        self.event_bus = event_bus

    async def re_extract(
        self,
        context_id: str,
        request: ReExtractionRequest,
    ) -> ReExtractionPreview | ReExtractionResult:
        """Re-extract bullets from raw history.

        Process:
        1. Load all activity records with raw_input (chronological order)
        2. For each record:
           a. Run the NEW Reflector model on the raw_input
           b. Compare new bullets against existing bullets from that activity
           c. Curator produces delta operations (ADD new, UPDATE stale, REMOVE obsolete)
        3. If dry_run: return preview of changes
        4. If not dry_run: apply deltas atomically
        """
        start_time = time.monotonic()

        # Load activities with raw input
        activities = await self.storage.get_activities_with_raw_input(
            context_id, since=request.since
        )

        if not activities:
            return ReExtractionPreview(
                activities_to_process=0,
                estimated_bullets_added=0,
                estimated_bullets_updated=0,
                estimated_bullets_removed=0,
                estimated_bullets_unchanged=0,
                estimated_input_tokens=0,
                estimated_output_tokens=0,
            )

        all_operations: list[DeltaOperation] = []

        for activity in activities:
            if not activity.raw_input:
                continue

            # Reconstruct feedback if available
            feedback = None
            if activity.feedback:
                feedback = ExecutionFeedback(**activity.feedback)

            # Run NEW Reflector on the original raw input
            new_reflection = await self.reflector.reflect(
                raw_input=activity.raw_input,
                feedback=feedback,
                model_override=request.reflector_model,
            )

            # Get the bullets that were originally produced from this activity
            old_bullets = await self.storage.get_bullets_by_ids(
                context_id, activity.bullet_ids_produced
            )

            # Curator compares new extraction against old bullets
            deltas = await self.curator.curate_re_extraction(
                new_reflection=new_reflection,
                old_bullets=old_bullets,
                context_id=context_id,
            )

            all_operations.extend(deltas.operations)

        if request.dry_run:
            return self._build_preview(all_operations, activities)

        # Apply all deltas
        from engram.core.delta import DeltaEngine
        delta_engine = DeltaEngine(self.storage)

        batch = DeltaBatch(
            context_id=context_id,
            operations=all_operations,
            trigger="re_extraction",
        )

        if self.lock_manager is not None:
            async with self.lock_manager.acquire(context_id):
                result_batch = await delta_engine.apply_batch(batch)
        else:
            result_batch = await delta_engine.apply_batch(batch)

        # Record re-extraction event in activity ledger
        import uuid as _uuid
        await self.storage.add_activity(
            _uuid.UUID(context_id) if len(context_id) >= 32 else _uuid.uuid4(),
            Activity(
                agent_id="engram-system",
                action_type=ActionType.RE_EXTRACTION_RAN,
                summary=(
                    f"Re-extracted with {request.reflector_model}: "
                    f"{len(activities)} activities processed, "
                    f"{len(all_operations)} delta operations"
                ),
                delta_batch_id=result_batch.id,
                extraction_model=request.reflector_model,
            ),
        )

        # Emit event
        if self.event_bus is not None:
            self.event_bus.emit(
                context_id,
                event_type="re_extraction_ran",
                data={
                    "model": request.reflector_model,
                    "activities_processed": len(activities),
                    "operations": len(all_operations),
                },
            )

        duration = time.monotonic() - start_time
        return self._build_result(
            all_operations, activities, request.reflector_model,
            [result_batch.id], duration,
        )

    def _build_preview(
        self, deltas: list[DeltaOperation], activities: list[Activity],
    ) -> ReExtractionPreview:
        added = sum(1 for d in deltas if d.op_type == DeltaOpType.ADD_BULLET)
        updated = sum(1 for d in deltas if d.op_type == DeltaOpType.UPDATE_BULLET)
        removed = sum(1 for d in deltas if d.op_type == DeltaOpType.REMOVE_BULLET)

        # Estimate tokens from raw input lengths
        total_chars = sum(len(a.raw_input or "") for a in activities)
        est_input_tokens = total_chars // 4

        return ReExtractionPreview(
            activities_to_process=len(activities),
            estimated_bullets_added=added,
            estimated_bullets_updated=updated,
            estimated_bullets_removed=removed,
            estimated_bullets_unchanged=0,
            estimated_input_tokens=est_input_tokens,
            estimated_output_tokens=est_input_tokens // 3,
        )

    def _build_result(
        self,
        deltas: list[DeltaOperation],
        activities: list[Activity],
        model: str,
        batch_ids: list[str],
        duration: float,
    ) -> ReExtractionResult:
        return ReExtractionResult(
            activities_processed=len(activities),
            bullets_added=sum(1 for d in deltas if d.op_type == DeltaOpType.ADD_BULLET),
            bullets_updated=sum(1 for d in deltas if d.op_type == DeltaOpType.UPDATE_BULLET),
            bullets_removed=sum(1 for d in deltas if d.op_type == DeltaOpType.REMOVE_BULLET),
            bullets_unchanged=0,
            delta_batches_applied=batch_ids,
            new_extraction_model=model,
            duration_seconds=round(duration, 2),
        )
