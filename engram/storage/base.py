"""Abstract storage backend interface for Engram v0.3."""

from __future__ import annotations

import abc
import uuid
from datetime import datetime
from typing import Any

from engram.core.models import (
    Activity,
    Bullet,
    CapacityStatus,
    ConceptEdge,
    ConceptNode,
    Context,
    DeltaBatch,
    IntentAnchor,
    MaterializationRecord,
    SchemaNode,
)


class StorageBackend(abc.ABC):
    """Abstract interface that all storage backends must implement."""

    # ── Lifecycle ──────────────────────────────────────────────────────

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Create tables / indexes if they don't exist."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release connections and resources."""

    # ── Context CRUD ───────────────────────────────────────────────────

    @abc.abstractmethod
    async def create_context(self, context: Context) -> Context:
        """Persist a new context with its intent anchor."""

    @abc.abstractmethod
    async def get_context(self, context_id: uuid.UUID) -> Context | None:
        """Return a context by ID, or None if not found."""

    @abc.abstractmethod
    async def list_contexts(
        self, owner: str | None = None, status: str | None = None
    ) -> list[Context]:
        """List contexts, optionally filtered."""

    @abc.abstractmethod
    async def update_context(self, context: Context) -> Context:
        """Update an existing context."""

    @abc.abstractmethod
    async def delete_context(self, context_id: uuid.UUID) -> None:
        """Hard-delete a context and all associated data."""

    @abc.abstractmethod
    async def update_core_memory(
        self, context_id: str, core_memory: str
    ) -> None:
        """Replace the always-in-context core memory blob for a context."""

    # ── Intent ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_intent(self, context_id: uuid.UUID) -> IntentAnchor | None:
        """Load the intent anchor for a context."""

    @abc.abstractmethod
    async def update_intent(self, intent: IntentAnchor) -> IntentAnchor:
        """Update mutable parts of an intent."""

    # ── Bullets (v0.2) ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def add_bullet(self, context_id: str, bullet: Bullet) -> Bullet:
        """Add a bullet to a context."""

    @abc.abstractmethod
    async def get_bullet(self, bullet_id: str) -> Bullet | None:
        """Get a single bullet by ID."""

    @abc.abstractmethod
    async def list_bullets(
        self,
        context_id: str,
        section: str | None = None,
        bullet_type: str | None = None,
        include_archived: bool = False,
        min_salience: float | None = None,
    ) -> list[Bullet]:
        """List bullets in a context with optional filters."""

    @abc.abstractmethod
    async def update_bullet(self, bullet: Bullet) -> Bullet:
        """Update an existing bullet."""

    @abc.abstractmethod
    async def remove_bullet(self, bullet_id: str) -> None:
        """Soft-delete a bullet (mark inactive)."""

    @abc.abstractmethod
    async def find_similar_bullets(
        self,
        context_id: str,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[Bullet, float]]:
        """Find bullets by embedding cosine similarity."""

    @abc.abstractmethod
    async def count_bullets(self, context_id: str) -> int:
        """Count active bullets in a context."""

    # ── Schemas (v0.2) ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def add_schema(self, context_id: str, schema: SchemaNode) -> SchemaNode:
        """Add a schema node."""

    @abc.abstractmethod
    async def get_schema(self, schema_id: str) -> SchemaNode | None:
        """Get a schema by ID."""

    @abc.abstractmethod
    async def list_schemas(self, context_id: str) -> list[SchemaNode]:
        """List all schemas in a context."""

    @abc.abstractmethod
    async def update_schema(self, schema: SchemaNode) -> SchemaNode:
        """Update an existing schema."""

    # ── Delta History (v0.2) ───────────────────────────────────────────

    @abc.abstractmethod
    async def save_delta_batch(self, delta_batch: DeltaBatch) -> DeltaBatch:
        """Persist a delta batch."""

    @abc.abstractmethod
    async def get_delta_batch(self, delta_batch_id: str) -> DeltaBatch | None:
        """Get a delta batch by ID."""

    @abc.abstractmethod
    async def list_delta_batches(
        self, context_id: str, limit: int = 50, offset: int = 0
    ) -> list[DeltaBatch]:
        """List delta batches for a context, newest first."""

    # ── Materialization Tracking (v0.2) ────────────────────────────────

    @abc.abstractmethod
    async def save_materialization(self, record: MaterializationRecord) -> MaterializationRecord:
        """Save a materialization record for reconsolidation tracking."""

    @abc.abstractmethod
    async def get_materialization(self, materialization_id: str) -> MaterializationRecord | None:
        """Get a materialization record by ID."""

    @abc.abstractmethod
    async def mark_materialization_reconsolidated(
        self, materialization_id: str, when: datetime,
    ) -> None:
        """Stamp a materialization as reconsolidated so replays are no-ops."""

    # ── Lifecycle Management (v0.3) ─────────────────────────────────

    @abc.abstractmethod
    async def archive_bullet(
        self, context_id: str, bullet_id: str, reason: str = "manual"
    ) -> bool:
        """Transition a bullet to archived state. Returns True if successful."""

    @abc.abstractmethod
    async def restore_bullet(self, context_id: str, bullet_id: str) -> Bullet | None:
        """Restore an archived bullet to active state. Returns the bullet or None."""

    @abc.abstractmethod
    async def purge_bullet(self, context_id: str, bullet_id: str) -> bool:
        """Permanently delete a bullet and its connected edges. Returns True if deleted."""

    @abc.abstractmethod
    async def purge_expired_archives(
        self, context_id: str, purge_after_days: int = 180
    ) -> int:
        """Permanently delete archived bullets older than purge_after_days. Returns count."""

    @abc.abstractmethod
    async def get_archived_bullets(
        self, context_id: str, offset: int = 0, limit: int = 50
    ) -> list[Bullet]:
        """List archived bullets with pagination."""

    @abc.abstractmethod
    async def get_capacity_status(
        self, context_id: str, max_active_bullets: int = 10000
    ) -> CapacityStatus:
        """Get current capacity metrics for a context."""

    @abc.abstractmethod
    async def purge_context(self, context_id: str) -> bool:
        """Permanently delete all data for a context (GDPR erasure). Returns True if deleted."""

    @abc.abstractmethod
    async def purge_user(self, user_id: str) -> int:
        """Permanently delete all contexts owned by a user. Returns count of contexts purged."""

    # ── Legacy Concepts (backward compat) ──────────────────────────────

    @abc.abstractmethod
    async def add_concept(self, context_id: uuid.UUID, concept: ConceptNode) -> ConceptNode:
        """Add a concept node (legacy)."""

    @abc.abstractmethod
    async def get_concept(self, concept_id: uuid.UUID) -> ConceptNode | None:
        """Get a concept by ID (legacy)."""

    @abc.abstractmethod
    async def list_concepts(
        self, context_id: uuid.UUID, include_invalid: bool = False,
        type_filter: str | None = None, domain_tags: list[str] | None = None,
    ) -> list[ConceptNode]:
        """List concepts (legacy)."""

    @abc.abstractmethod
    async def update_concept(self, concept: ConceptNode) -> ConceptNode:
        """Update a concept (legacy)."""

    @abc.abstractmethod
    async def find_similar_concepts(
        self, context_id: uuid.UUID, embedding: list[float],
        limit: int = 10, threshold: float = 0.7,
    ) -> list[tuple[ConceptNode, float]]:
        """Find concepts by embedding similarity (legacy)."""

    @abc.abstractmethod
    async def count_concepts(self, context_id: uuid.UUID) -> int:
        """Count valid concepts (legacy)."""

    # ── Edges ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def add_edge(self, context_id: uuid.UUID, edge: ConceptEdge) -> ConceptEdge:
        """Add an edge."""

    @abc.abstractmethod
    async def get_edges(
        self, context_id: uuid.UUID, node_id: uuid.UUID | None = None,
        edge_type: str | None = None,
    ) -> list[ConceptEdge]:
        """Get edges."""

    @abc.abstractmethod
    async def delete_edge(self, edge_id: uuid.UUID) -> None:
        """Delete an edge."""

    # ── Activity Ledger ────────────────────────────────────────────────

    @abc.abstractmethod
    async def add_activity(self, context_id: uuid.UUID, activity: Activity) -> Activity:
        """Append to activity ledger."""

    @abc.abstractmethod
    async def list_activities(
        self, context_id: uuid.UUID, limit: int = 50, offset: int = 0,
    ) -> list[Activity]:
        """List activities, newest first."""

    # ── Raw Input Queries (v0.4) ───────────────────────────────────────

    @abc.abstractmethod
    async def get_activities_with_raw_input(
        self,
        context_id: str,
        since: Any = None,
        content_type: str | None = None,
    ) -> list[Activity]:
        """Get activity records that have raw_input, for re-extraction.
        Ordered chronologically (oldest first)."""

    @abc.abstractmethod
    async def get_raw_input_by_hash(
        self, context_id: str, raw_input_hash: str,
    ) -> Activity | None:
        """Check if this exact raw input has already been ingested (dedup)."""

    @abc.abstractmethod
    async def get_bullets_by_ids(
        self, context_id: str, bullet_ids: list[str],
    ) -> list[Bullet]:
        """Get multiple bullets by their IDs."""

    @abc.abstractmethod
    async def find_similar_activities(
        self,
        context_id: str,
        embedding: list[float],
        limit: int = 3,
        threshold: float = 0.85,
        exclude_hash: str | None = None,
    ) -> list[tuple[Activity, float]]:
        """Find prior activities (with raw_input + embedding) most similar to
        the given embedding. Returns [(activity, cosine_similarity), ...] sorted
        descending by similarity, filtered to similarity >= threshold.

        exclude_hash skips an activity whose raw_input_hash matches (e.g., the
        commit currently being materialized against)."""
