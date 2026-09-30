"""Bullet management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from engram.core.models import (
    AddBulletRequest,
    AddBulletResponse,
    BulletType,
)

router = APIRouter()


@router.get("/{context_id}/bullets")
async def list_bullets(
    context_id: uuid.UUID,
    request: Request,
    section: str | None = None,
    bullet_type: str | None = None,
    include_archived: bool = False,
    limit: int = 100,
) -> list[dict]:
    """List bullets in a context, optionally filtered by section or type."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    bullets = await storage.list_bullets(
        str(context_id),
        section=section,
        bullet_type=bullet_type,
        include_archived=include_archived,
    )

    # Exclude embeddings and limit results
    return [
        {
            k: v for k, v in b.model_dump(mode="json").items()
            if k != "embedding"
        }
        for b in bullets[:limit]
    ]


@router.post("/{context_id}/bullets", response_model=AddBulletResponse, status_code=201)
async def add_bullet(
    context_id: uuid.UUID, req: AddBulletRequest, request: Request
) -> AddBulletResponse:
    """Directly add a bullet (bypass Reflector pipeline)."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    ingestion = request.app.state.ingestion
    bullet, batch = await ingestion.add_bullet_directly(
        context_id=str(context_id),
        content=req.content,
        section=req.section,
        bullet_type=req.bullet_type,
        salience=req.salience,
        confidence=req.confidence,
        agent_id=req.agent_id,
        session_id=req.session_id,
    )

    return AddBulletResponse(
        bullet_id=bullet.id,
        delta_batch_id=batch.id,
    )


@router.get("/{context_id}/bullets/{bullet_id}")
async def get_bullet(
    context_id: uuid.UUID, bullet_id: str, request: Request
) -> dict:
    """Get a single bullet by ID."""
    storage = request.app.state.storage
    bullet = await storage.get_bullet(bullet_id)
    if bullet is None:
        raise HTTPException(status_code=404, detail=f"Bullet {bullet_id} not found")

    data = bullet.model_dump(mode="json")
    data.pop("embedding", None)
    return data
