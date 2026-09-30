"""Concept graph engine — high-level operations over the concept graph."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from engram.core.models import (
    ActionType,
    Activity,
    ConceptNode,
)
from engram.storage.base import StorageBackend


class ConceptGraph:
    """High-level graph operations: invalidation, querying, statistics."""

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    async def invalidate_concepts(
        self,
        context_id: uuid.UUID,
        concept_ids: list[uuid.UUID],
        reason: str,
        agent_id: str = "system",
    ) -> list[uuid.UUID]:
        """Mark concepts as invalid (soft delete) and log activity."""
        invalidated: list[uuid.UUID] = []
        now = datetime.now(timezone.utc)

        for cid in concept_ids:
            concept = await self.storage.get_concept(cid)
            if concept is None or not concept.is_valid:
                continue
            concept.is_valid = False
            concept.invalidated_at = now
            concept.invalidation_reason = reason
            concept.version += 1
            await self.storage.update_concept(concept)
            invalidated.append(cid)

        if invalidated:
            activity = Activity(
                agent_id=agent_id,
                action_type=ActionType.FACT_LEARNED,
                summary=f"Invalidated {len(invalidated)} concepts: {reason}",
                concepts_invalidated=invalidated,
            )
            await self.storage.add_activity(context_id, activity)

        return invalidated

    async def get_concept_neighborhood(
        self,
        context_id: uuid.UUID,
        concept_id: uuid.UUID,
        depth: int = 1,
    ) -> list[ConceptNode]:
        """Get a concept and its neighbors up to a given depth."""
        visited: set[uuid.UUID] = set()
        frontier = {concept_id}
        result: list[ConceptNode] = []

        for _ in range(depth + 1):
            next_frontier: set[uuid.UUID] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                concept = await self.storage.get_concept(nid)
                if concept and concept.is_valid:
                    result.append(concept)
                edges = await self.storage.get_edges(context_id, node_id=nid)
                for edge in edges:
                    next_frontier.add(edge.from_node)
                    next_frontier.add(edge.to_node)
            frontier = next_frontier - visited

        return result
