"""Pydantic data models for the Engram concept graph.

v0.3 — Adds concurrency control, three-tier data lifecycle (active → archived → purged),
capacity management, and event bus for multi-agent coordination.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _short_id() -> str:
    """Generate a short 8-character ID."""
    return str(uuid.uuid4())[:8]


# ── Enums ──────────────────────────────────────────────────────────────────


class BulletType(str, enum.Enum):
    STRATEGY = "strategy"
    WARNING = "warning"
    FACT = "fact"
    PROCEDURE = "procedure"
    EXCEPTION = "exception"
    PRINCIPLE = "principle"
    DECISION = "decision"
    # v0.5: Mem-alpha distinguishes episodic memories (timestamped events)
    # from semantic facts. Episodic bullets follow "At {timestamp}, {actor}
    # {did X}" and are merged with a looser embedding threshold (multiple
    # agents often log paraphrases of the same event).
    EPISODIC = "episodic"


class SourceType(str, enum.Enum):
    REFLECTION = "reflection"
    EXECUTION_FEEDBACK = "execution_feedback"
    USER_INPUT = "user_input"
    CONSOLIDATION = "consolidation"
    SCHEMA_DERIVED = "schema_derived"


class ConceptType(str, enum.Enum):
    """Legacy concept types — kept for backward compatibility."""
    ENTITY = "entity"
    FACT = "fact"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    GOAL = "goal"
    OBSERVATION = "observation"
    PROCEDURE = "procedure"
    EXCEPTION = "exception"


class EdgeType(str, enum.Enum):
    CAUSED_BY = "caused_by"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    PREFERRED_OVER = "preferred_over"
    PART_OF = "part_of"
    PRECEDED_BY = "preceded_by"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    BLOCKS = "blocks"
    RELATED_TO = "related_to"
    CO_RECALLED = "co_recalled"


class IntentStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ActionType(str, enum.Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    DECISION_MADE = "decision_made"
    FACT_LEARNED = "fact_learned"
    HYPOTHESIS_FORMED = "hypothesis_formed"
    TOOL_CALLED = "tool_called"
    USER_INPUT_RECEIVED = "user_input_received"
    CONTEXT_BRANCHED = "context_branched"
    CONTEXT_MERGED = "context_merged"
    CONSOLIDATION_RAN = "consolidation_ran"
    MATERIALIZATION_OCCURRED = "materialization_occurred"
    RE_EXTRACTION_RAN = "re_extraction_ran"


class ContentType(str, enum.Enum):
    CONVERSATION = "conversation"
    TOOL_OUTPUT = "tool_output"
    DOCUMENT = "document"
    OBSERVATION = "observation"


class DeltaOpType(str, enum.Enum):
    ADD_BULLET = "add_bullet"
    UPDATE_BULLET = "update_bullet"
    REMOVE_BULLET = "remove_bullet"
    MERGE_BULLETS = "merge_bullets"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_SCHEMA = "add_schema"
    UPDATE_SCHEMA = "update_schema"
    UPDATE_CORE_MEMORY = "update_core_memory"
    RECONSOLIDATE_BULLET = "reconsolidate_bullet"


class DeltaSource(str, enum.Enum):
    REFLECTOR = "reflector"
    CURATOR = "curator"
    USER = "user"
    CONSOLIDATION = "consolidation"


class LifecycleState(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    PURGED = "purged"


class FeedbackOutcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ToolCallStatus(str, enum.Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ── Bullet (Atomic Knowledge Unit) ────────────────────────────────────────


class Bullet(BaseModel):
    """Atomic unit of stored knowledge — a single actionable insight.

    Inspired by ACE's itemized context representation. Each bullet is
    independently retrievable, updatable, and deletable. Usage tracking
    enables the system to learn which bullets are actually useful over time.
    """

    id: str = Field(default_factory=_short_id)
    context_id: str | None = None
    section: str = "general"
    content: str
    bullet_type: BulletType = BulletType.FACT
    source_type: SourceType = SourceType.REFLECTION

    embedding: list[float] | None = None

    # Usage tracking — enables reinforcement learning on context quality
    hit_count: int = 0
    miss_count: int = 0
    recall_count: int = 0
    last_recalled_at: datetime | None = None

    # Salience and lifecycle
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Provenance
    source_session: str | None = None
    source_agent: str | None = None
    parent_concept_id: str | None = None
    schema_id: str | None = None

    # Soft-delete and lifecycle
    is_active: bool = True
    is_archived: bool = False
    archived_at: datetime | None = None
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    archive_reason: str | None = None

    @property
    def hit_rate(self) -> float:
        if self.recall_count == 0:
            return 0.0
        return self.hit_count / self.recall_count

    @property
    def effective_salience(self) -> float:
        """Salience adjusted by usage history — bullets that prove useful get boosted."""
        base = self.salience
        if self.recall_count > 5:
            base *= 0.5 + self.hit_rate
        return min(1.0, base)


# ── Schema Node ───────────────────────────────────────────────────────────


class SlotDefinition(BaseModel):
    """A slot in a schema — represents an expected component of a pattern."""

    name: str
    description: str
    typical_values: list[str] = Field(default_factory=list)
    required: bool = False


class SchemaNode(BaseModel):
    """Abstract pattern derived from multiple concrete experiences.

    Brain-inspired: The neocortex builds schemas (abstract frameworks)
    from repeated experiences. New experiences are encoded relative to
    schemas — only the deltas from the expected pattern are stored.
    Things that violate schemas (prediction errors) get extra attention.
    """

    id: str = Field(default_factory=_short_id)
    context_id: str | None = None
    name: str
    description: str
    slots: dict[str, SlotDefinition] = Field(default_factory=dict)
    typical_values: dict[str, str] = Field(default_factory=dict)
    exceptions: list[str] = Field(default_factory=list)
    instance_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bullet_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Delta Operations ──────────────────────────────────────────────────────


class DeltaOperation(BaseModel):
    """A single atomic change to the concept graph.

    Every mutation goes through delta operations. This prevents
    context collapse and enables full auditability + rollback.

    `previous_state` carries caller-provided input to the apply function
    (e.g. RECONSOLIDATE_BULLET deltas) when present pre-apply. The apply
    function writes the snapshot needed for rollback into `rollback_state`
    so it never clobbers caller input — important for idempotency on retry.
    Pre-rollback_state op records still place the snapshot in `previous_state`,
    so the rollback path reads `rollback_state` with fallback to `previous_state`.
    """

    id: str = Field(default_factory=_short_id)
    op_type: DeltaOpType
    target_id: str | None = None
    target_ids: list[str] | None = None
    section: str | None = None
    content: str | None = None
    bullet_type: str | None = None
    reasoning: str = ""
    source: DeltaSource = DeltaSource.CURATOR
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=_utcnow)
    session_id: str | None = None
    agent_id: str | None = None
    previous_state: dict[str, Any] | None = None
    rollback_state: dict[str, Any] | None = None


class DeltaBatch(BaseModel):
    """A batch of delta operations applied atomically."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str
    operations: list[DeltaOperation] = Field(default_factory=list)
    trigger: str = "commit"
    timestamp: datetime = Field(default_factory=_utcnow)
    bullets_added: int = 0
    bullets_updated: int = 0
    bullets_removed: int = 0
    bullets_merged: int = 0


# ── Execution Feedback ────────────────────────────────────────────────────


class ToolCallResult(BaseModel):
    """Result of a single tool call during execution."""

    tool: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    error: str | None = None
    duration_ms: int | None = None


class ExecutionFeedback(BaseModel):
    """Structured feedback about task execution.

    ACE finding: Using execution signals produces much higher quality
    reflections than parsing raw conversation text alone.
    """

    outcome: FeedbackOutcome = FeedbackOutcome.UNKNOWN
    metrics: dict[str, float] | None = None
    tool_calls: list[ToolCallResult] | None = None
    user_accepted: bool | None = None
    error_message: str | None = None
    notes: str | None = None


# ── Reflection (Reflector output) ─────────────────────────────────────────


class ReflectionInsight(BaseModel):
    """A single insight extracted by the Reflector."""

    content: str
    insight_type: str = "fact"
    suggested_section: str = "general"
    evidence: str = ""
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)


class Reflection(BaseModel):
    """Output of the Reflector — structured analysis of what happened."""

    new_insights: list[ReflectionInsight] = Field(default_factory=list)
    strategies_that_worked: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    prediction_errors: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    rounds_completed: int = 1
    confidence: float = 0.5
    raw_input_type: str = "conversation"

    # Mem-α inspired: optional rewrite of the always-in-context core memory blob.
    # None means "leave unchanged". A non-None value REPLACES core_memory wholesale,
    # so the Reflector is instructed to preserve information it wants to keep.
    core_memory_update: str | None = None


# ── Materialization Tracking ──────────────────────────────────────────────


class MaterializationRecord(BaseModel):
    """Record of a materialization event for reconsolidation tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str
    bullets_included: list[str] = Field(default_factory=list)
    token_count: int = 0
    target_model: str = "claude"
    query: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    # v0.5: set once the recall has been reconsolidated, so a replayed
    # materialization_id (retry, at-least-once delivery) can't double-count.
    reconsolidated_at: datetime | None = None


# ── Consolidation ─────────────────────────────────────────────────────────


class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation process."""

    fast_decay_rate: float = 0.97
    slow_decay_rate: float = 0.995
    min_salience: float = 0.02
    dedup_threshold: float = 0.92
    schema_min_instances: int = 3
    archive_salience_threshold: float = 0.05
    archive_days_threshold: int = 60
    consolidation_trigger: str = "every_10_commits"


class LifecycleConfig(BaseModel):
    """Per-context configuration for data lifecycle management."""

    max_active_bullets: int = 10000
    archive_after_days: int = 60
    archive_salience_below: float = 0.05
    purge_after_days: int = 180
    protected_types: list[str] = Field(
        default_factory=lambda: ["decision", "principle"]
    )

    # Aggressive thresholds (used when capacity pressure is high)
    aggressive_dedup_threshold: float = 0.88
    aggressive_archive_salience: float = 0.10
    aggressive_schema_min_instances: int = 2


class CapacityStatus(BaseModel):
    """Current capacity metrics for a context."""

    active_bullet_count: int = 0
    max_active_bullets: int = 10000
    archived_bullet_count: int = 0
    schema_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def capacity_percent(self) -> float:
        if self.max_active_bullets == 0:
            return 100.0
        return round(
            (self.active_bullet_count / self.max_active_bullets) * 100, 1
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pressure_level(self) -> Literal["normal", "high", "critical", "full"]:
        pct = self.capacity_percent
        if pct >= 100:
            return "full"
        if pct >= 95:
            return "critical"
        if pct >= 80:
            return "high"
        return "normal"


class ContextEvent(BaseModel):
    """An event emitted when something happens to a context."""

    event_type: str
    context_id: str
    agent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)


class ConsolidationReport(BaseModel):
    """Summary of what happened during consolidation."""

    context_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    decayed: int = 0
    deduplicated: int = 0
    schemas_formed: int = 0
    edges_strengthened: int = 0
    archived: int = 0
    promoted: int = 0
    purged: int = 0
    mode: str = "normal"
    duration_ms: int = 0


# ── Legacy ConceptNode (kept for backward compat) ─────────────────────────


class ConceptNode(BaseModel):
    """A discrete unit of knowledge in the concept graph.

    Note: In v0.2+, prefer Bullet for new storage. ConceptNode is retained
    for backward compatibility and as a higher-level grouping construct.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: ConceptType
    content: str = Field(description="Concise natural language, 1-3 sentences max")
    embedding: list[float] | None = Field(default=None)

    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime | None = None
    version: int = 1
    source_session: uuid.UUID | None = None
    source_agent: str | None = None

    domain_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    is_valid: bool = True
    invalidated_at: datetime | None = None
    invalidation_reason: str | None = None

    # v0.2: link to bullets
    bullet_ids: list[str] = Field(default_factory=list)


class ConceptEdge(BaseModel):
    """A typed relationship between two nodes (bullets, concepts, or schemas)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    from_node: uuid.UUID
    to_node: uuid.UUID
    type: EdgeType
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    source_session: uuid.UUID | None = None


# ── Intent Anchors ────────────────────────────────────────────────────────


class IntentAnchor(BaseModel):
    """Top-level structure that prevents context drift."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    context_id: uuid.UUID | None = None

    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    status: IntentStatus = IntentStatus.ACTIVE
    sub_intents: list[IntentAnchor] = Field(default_factory=list)
    progress_notes: list[str] = Field(default_factory=list)


# ── Activity Ledger ───────────────────────────────────────────────────────


class Activity(BaseModel):
    """A chronological entry in the activity ledger.

    v0.4: Preserves raw input alongside extracted bullets, like git commits.
    The raw_input is the immutable "source code" — bullets are the "compiled
    output" that can be regenerated when a better Reflector is available.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=_utcnow)
    agent_id: str
    session_id: uuid.UUID | None = None
    action_type: ActionType
    summary: str
    concepts_created: list[uuid.UUID] = Field(default_factory=list)
    concepts_updated: list[uuid.UUID] = Field(default_factory=list)
    concepts_invalidated: list[uuid.UUID] = Field(default_factory=list)
    delta_batch_id: str | None = None
    materialization_id: str | None = None

    # v0.4: Raw input preservation — the "git history"
    raw_input: str | None = None
    raw_input_hash: str | None = None
    content_type: str | None = None
    source_agent_model: str | None = None
    feedback: dict[str, Any] | None = None

    # v0.4: Extraction metadata — tracks HOW bullets were extracted
    extraction_model: str | None = None
    extraction_prompt_version: str | None = None
    bullet_ids_produced: list[str] = Field(default_factory=list)

    # v0.5: DC-inspired worked-example retrieval at materialization time.
    # Stored alongside the raw input so we can find nearest prior cases by
    # semantic similarity without re-embedding them on the read path.
    raw_input_embedding: list[float] | None = None

    @staticmethod
    def compute_hash(raw_input: str) -> str:
        """Compute SHA-256 hash of raw input for dedup."""
        import hashlib
        return hashlib.sha256(raw_input.encode()).hexdigest()


# ── Context (Top-Level Container) ─────────────────────────────────────────


CORE_MEMORY_MAX_TOKENS = 512  # Mem-α: bounded always-in-context summary slot.


def cap_core_memory(text: str, max_tokens: int = CORE_MEMORY_MAX_TOKENS) -> str:
    """Truncate core memory text to a token budget. Uses tiktoken when available,
    otherwise falls back to ~4 chars/token. Truncation is on token boundaries so
    we never emit a half-decoded codepoint.

    This is the single enforcement point for the ≤512-token core-memory invariant;
    it is called both by the ingestion commit path and by the UPDATE_CORE_MEMORY
    delta apply, so the bound holds for every writer of the delta op.
    """
    if not text:
        return ""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        char_budget = max_tokens * 4
        return text if len(text) <= char_budget else text[:char_budget]


class Context(BaseModel):
    """Top-level container holding an intent anchor, concept graph, and activity ledger."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str = ""
    owner: str = "default"

    intent: IntentAnchor
    version: int = 1
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # v0.5: Mem-α inspired core memory — running summary always included in materialization,
    # rewritten by the Reflector (full-replace) and capped at CORE_MEMORY_MAX_TOKENS.
    core_memory: str = ""

    # v0.3 lifecycle configuration
    lifecycle_config: LifecycleConfig = Field(default_factory=LifecycleConfig)

    # v0.2 health metrics (populated by storage queries)
    total_bullets: int = 0
    avg_salience: float = 0.0
    stale_bullet_count: int = 0
    schema_count: int = 0
    last_consolidation: datetime | None = None


# ── API Request / Response Models ──────────────────────────────────────────


class IntentInput(BaseModel):
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class CreateContextRequest(BaseModel):
    name: str
    description: str = ""
    owner: str = "default"
    intent: IntentInput


class CreateContextResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    owner: str
    intent: IntentAnchor
    created_at: datetime


class ContextListItem(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    owner: str
    status: IntentStatus
    concept_count: int = 0
    bullet_count: int = 0
    schema_count: int = 0
    created_at: datetime
    updated_at: datetime


class CommitRequest(BaseModel):
    agent_id: str
    session_id: uuid.UUID | None = None
    content: str
    content_type: ContentType = ContentType.CONVERSATION
    feedback: ExecutionFeedback | None = None
    materialization_id: str | None = None
    source_model: str | None = None  # v0.4: "claude-sonnet-4-5", "gpt-4o", etc.


class CommitResponse(BaseModel):
    delta_batch_id: str
    bullets_added: int = 0
    bullets_updated: int = 0
    bullets_removed: int = 0
    bullets_merged: int = 0
    activity_id: uuid.UUID


class RelationshipInput(BaseModel):
    type: EdgeType
    target_content: str
    rationale: str | None = None


class AddBulletRequest(BaseModel):
    content: str
    section: str = "general"
    bullet_type: BulletType = BulletType.FACT
    salience: float = 0.5
    confidence: float = 0.5
    agent_id: str = "manual"
    session_id: str | None = None


class AddBulletResponse(BaseModel):
    bullet_id: str
    delta_batch_id: str


# Keep legacy AddConceptRequest for backward compat
class AddConceptRequest(BaseModel):
    type: ConceptType
    content: str
    salience: float = 0.5
    confidence: float = 0.8
    domain_tags: list[str] = Field(default_factory=list)
    relationships: list[RelationshipInput] = Field(default_factory=list)
    agent_id: str = "manual"
    session_id: uuid.UUID | None = None


class AddConceptResponse(BaseModel):
    concept_id: uuid.UUID
    edges_created: list[uuid.UUID]


class RecordDecisionRequest(BaseModel):
    decision: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)
    agent_id: str
    domain_tags: list[str] = Field(default_factory=list)


class MaterializeRequest(BaseModel):
    query: str | None = None
    task: str | None = None
    agent_role: str | None = None
    focus_domains: list[str] = Field(default_factory=list)
    focus_sections: list[str] = Field(default_factory=list)
    token_budget: int = 4000
    target_model: str = "claude"
    include_intent: bool = True
    include_decisions: bool = True
    include_schemas: bool = True
    recency_weight: float = 0.5
    max_concept_age_days: int | None = None

    # v0.5: DC-inspired retrieval of nearest prior raw inputs as worked examples.
    include_worked_examples: bool = True
    worked_example_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    worked_example_limit: int = Field(default=2, ge=0)

    # v0.5: surface per-bullet usage stats in the rendered text (Phase 5).
    include_usage_stats: bool = False

    # v0.5: MMR diversity λ in materialization ranking. 1.0 = relevance only
    # (legacy greedy behavior); 0.7 mixes in diversity (the default).
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)


class MaterializeResponse(BaseModel):
    materialization_id: str
    rendered_text: str
    concepts_included: list[uuid.UUID] = Field(default_factory=list)
    bullets_included: list[str] = Field(default_factory=list)
    schemas_included: list[str] = Field(default_factory=list)
    token_count: int
    coverage_score: float


class RecallRequest(BaseModel):
    query: str
    token_budget: int = 2000
    target_model: str = "claude"


class InvalidateRequest(BaseModel):
    concept_ids: list[uuid.UUID] = Field(default_factory=list)
    bullet_ids: list[str] = Field(default_factory=list)
    reason: str


class ContextHealthResponse(BaseModel):
    context_id: str
    total_bullets: int
    active_bullets: int
    archived_bullets: int
    avg_salience: float
    avg_effective_salience: float
    stale_bullet_count: int
    schema_count: int
    total_edges: int
    total_activities: int
    last_consolidation: datetime | None = None
    top_sections: list[dict[str, Any]] = Field(default_factory=list)
    # v0.3 capacity
    capacity: CapacityStatus | None = None


class ConsolidateRequest(BaseModel):
    config: ConsolidationConfig | None = None


# ── Lifecycle API Models ──────────────────────────────────────────────────


class ArchiveBulletRequest(BaseModel):
    reason: str = "manual"


class LifecycleStatusResponse(BaseModel):
    capacity: CapacityStatus
    lifecycle_config: LifecycleConfig


class SyncRequest(BaseModel):
    since: datetime


class SyncResponse(BaseModel):
    delta_batches: list[DeltaBatch] = Field(default_factory=list)
    events: list[ContextEvent] = Field(default_factory=list)


# ── Re-Extraction API Models (v0.4) ──────────────────────────────────────


class ReExtractionRequest(BaseModel):
    """Request to re-extract bullets from raw input history."""
    reflector_model: str
    dry_run: bool = True
    since: datetime | None = None
    prompt_version: str | None = None


class ReExtractionPreview(BaseModel):
    """What would change if we re-extracted."""
    activities_to_process: int
    estimated_bullets_added: int
    estimated_bullets_updated: int
    estimated_bullets_removed: int
    estimated_bullets_unchanged: int
    estimated_input_tokens: int
    estimated_output_tokens: int


class ReExtractionResult(BaseModel):
    """What actually changed after re-extraction."""
    activities_processed: int
    bullets_added: int
    bullets_updated: int
    bullets_removed: int
    bullets_unchanged: int
    delta_batches_applied: list[str] = Field(default_factory=list)
    new_extraction_model: str
    duration_seconds: float


# ── Configuration ──────────────────────────────────────────────────────────


class EngineConfig(BaseModel):
    """Runtime configuration loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 5820
    log_level: str = "info"
    storage_backend: str = "sqlite"
    sqlite_path: str = "./engram.db"
    postgres_url: str = "postgresql://engram:engram@localhost:5432/engram"
    llm_model: str = "anthropic/claude-sonnet-4-20250514"
    llm_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
