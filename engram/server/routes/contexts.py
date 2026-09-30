"""Context management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from engram.core.exceptions import ContextNotFoundError
from engram.core.models import (
    CommitRequest,
    CommitResponse,
    ConsolidateRequest,
    ConsolidationReport,
    Context,
    ContextHealthResponse,
    ContextListItem,
    CreateContextRequest,
    CreateContextResponse,
    IntentAnchor,
    InvalidateRequest,
)

router = APIRouter()


@router.post("", response_model=CreateContextResponse, status_code=201)
async def create_context(req: CreateContextRequest, request: Request) -> CreateContextResponse:
    """Create a new context with an intent anchor."""
    storage = request.app.state.storage
    intent = IntentAnchor(
        objective=req.intent.objective,
        success_criteria=req.intent.success_criteria,
        constraints=req.intent.constraints,
    )
    context = Context(
        name=req.name,
        description=req.description,
        owner=req.owner,
        intent=intent,
    )
    created = await storage.create_context(context)
    return CreateContextResponse(
        id=created.id,
        name=created.name,
        description=created.description,
        owner=created.owner,
        intent=created.intent,
        created_at=created.created_at,
    )


@router.get("", response_model=list[ContextListItem])
async def list_contexts(
    request: Request,
    owner: str | None = None,
    status: str | None = None,
) -> list[ContextListItem]:
    """List all contexts, optionally filtered by owner or status."""
    storage = request.app.state.storage
    contexts = await storage.list_contexts(owner=owner, status=status)
    items: list[ContextListItem] = []
    for ctx in contexts:
        concept_count = await storage.count_concepts(ctx.id)
        bullet_count = await storage.count_bullets(str(ctx.id))
        schema_count = len(await storage.list_schemas(str(ctx.id)))
        items.append(
            ContextListItem(
                id=ctx.id,
                name=ctx.name,
                description=ctx.description,
                owner=ctx.owner,
                status=ctx.intent.status,
                concept_count=concept_count,
                bullet_count=bullet_count,
                schema_count=schema_count,
                created_at=ctx.created_at,
                updated_at=ctx.updated_at,
            )
        )
    return items


@router.get("/{context_id}")
async def get_context(context_id: uuid.UUID, request: Request) -> Context:
    """Get context metadata by ID."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")
    return ctx


@router.post("/{context_id}/commit", response_model=CommitResponse)
async def commit_content(
    context_id: uuid.UUID, req: CommitRequest, request: Request
) -> CommitResponse:
    """Ingest raw content — triggers Reflector → Curator → Delta pipeline."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    ingestion = request.app.state.ingestion
    batch = await ingestion.commit(
        context_id=context_id,
        agent_id=req.agent_id,
        content=req.content,
        content_type=req.content_type,
        session_id=req.session_id,
        feedback=req.feedback,
        materialization_id=req.materialization_id,
        source_model=req.source_model,
    )

    # Get the most recent activity for the activity_id
    activities = await storage.list_activities(context_id, limit=1)
    activity_id = activities[0].id if activities else uuid.uuid4()

    return CommitResponse(
        delta_batch_id=batch.id,
        bullets_added=batch.bullets_added,
        bullets_updated=batch.bullets_updated,
        bullets_removed=batch.bullets_removed,
        bullets_merged=batch.bullets_merged,
        activity_id=activity_id,
    )


@router.get("/{context_id}/activity")
async def get_activity(
    context_id: uuid.UUID,
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get the activity ledger for a context."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")
    activities = await storage.list_activities(context_id, limit=limit, offset=offset)
    return [a.model_dump(mode="json") for a in activities]


@router.post("/{context_id}/invalidate")
async def invalidate_concepts(
    context_id: uuid.UUID, req: InvalidateRequest, request: Request
) -> dict:
    """Mark concepts or bullets as invalid (soft delete)."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    result: dict = {"invalidated_concepts": [], "invalidated_bullets": []}

    # Legacy concept invalidation
    if req.concept_ids:
        graph = request.app.state.graph
        invalidated = await graph.invalidate_concepts(
            context_id, req.concept_ids, req.reason
        )
        result["invalidated_concepts"] = [str(uid) for uid in invalidated]

    # Bullet invalidation
    if req.bullet_ids:
        for bullet_id in req.bullet_ids:
            bullet = await storage.get_bullet(bullet_id)
            if bullet:
                bullet.is_active = False
                await storage.update_bullet(bullet)
                result["invalidated_bullets"].append(bullet_id)

    return result


@router.post("/{context_id}/consolidate")
async def consolidate(
    context_id: uuid.UUID, req: ConsolidateRequest, request: Request
) -> dict:
    """Run the consolidation engine (sleep cycle) on this context."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    consolidation = request.app.state.consolidation
    report = await consolidation.consolidate(
        context_id=str(context_id),
        config=req.config,
    )
    return report.model_dump(mode="json")


@router.get("/{context_id}/health", response_model=ContextHealthResponse)
async def get_health(
    context_id: uuid.UUID, request: Request
) -> ContextHealthResponse:
    """Get health metrics for a context — bullet stats, staleness, etc."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    ctx_id_str = str(context_id)
    all_bullets = await storage.list_bullets(ctx_id_str, include_archived=True)
    active_bullets = [b for b in all_bullets if b.is_active and not b.is_archived]
    archived_bullets = [b for b in all_bullets if b.is_archived]
    stale = [b for b in active_bullets if b.salience < 0.1]
    schemas = await storage.list_schemas(ctx_id_str)
    edges = await storage.get_edges(context_id)
    activities = await storage.list_activities(context_id, limit=1000)

    avg_salience = (
        sum(b.salience for b in active_bullets) / len(active_bullets)
        if active_bullets else 0.0
    )
    avg_effective = (
        sum(b.effective_salience for b in active_bullets) / len(active_bullets)
        if active_bullets else 0.0
    )

    # Count bullets by section
    section_counts: dict[str, int] = {}
    for b in active_bullets:
        section_counts[b.section] = section_counts.get(b.section, 0) + 1
    top_sections = [
        {"section": k, "count": v}
        for k, v in sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    # v0.3: Capacity status
    capacity = await storage.get_capacity_status(
        ctx_id_str, ctx.lifecycle_config.max_active_bullets
    )

    return ContextHealthResponse(
        context_id=ctx_id_str,
        total_bullets=len(all_bullets),
        active_bullets=len(active_bullets),
        archived_bullets=len(archived_bullets),
        avg_salience=round(avg_salience, 4),
        avg_effective_salience=round(avg_effective, 4),
        stale_bullet_count=len(stale),
        schema_count=len(schemas),
        total_edges=len(edges),
        total_activities=len(activities),
        top_sections=top_sections,
        capacity=capacity,
    )
