"""Lifecycle management API routes — v0.3 + v0.4 additions.

Archive, restore, purge bullets. Browse archived bullets. GDPR erasure.
SSE subscription for real-time events. Sync polling endpoint.
v0.4: Re-extraction endpoint, ingestion config management.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from engram.core.config import IngestionConfig
from engram.core.exceptions import CapacityExceededError
from engram.core.models import (
    ArchiveBulletRequest,
    Bullet,
    LifecycleStatusResponse,
    ReExtractionRequest,
)

router = APIRouter()


@router.get("/{context_id}/lifecycle", response_model=LifecycleStatusResponse)
async def get_lifecycle(context_id: str, request: Request) -> LifecycleStatusResponse:
    """Get lifecycle status including capacity and config."""
    storage = request.app.state.storage
    ctx = await storage.get_context(uuid.UUID(context_id))
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    capacity = await storage.get_capacity_status(
        context_id, ctx.lifecycle_config.max_active_bullets
    )
    return LifecycleStatusResponse(
        capacity=capacity,
        lifecycle_config=ctx.lifecycle_config,
    )


@router.post("/{context_id}/bullets/{bullet_id}/archive")
async def archive_bullet(
    context_id: str, bullet_id: str, request: Request,
    req: ArchiveBulletRequest | None = None,
) -> dict:
    """Manually archive a bullet."""
    storage = request.app.state.storage
    reason = req.reason if req else "manual"
    success = await storage.archive_bullet(context_id, bullet_id, reason)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Bullet {bullet_id} not found or not in active state",
        )
    return {"archived": True, "bullet_id": bullet_id, "reason": reason}


@router.post("/{context_id}/bullets/{bullet_id}/restore")
async def restore_bullet(context_id: str, bullet_id: str, request: Request) -> dict:
    """Restore an archived bullet to active state."""
    storage = request.app.state.storage

    # Check capacity before restoring
    ctx = await storage.get_context(uuid.UUID(context_id))
    if ctx is not None:
        capacity = await storage.get_capacity_status(
            context_id, ctx.lifecycle_config.max_active_bullets
        )
        if capacity.pressure_level == "full":
            raise HTTPException(
                status_code=409,
                detail="Cannot restore: context at capacity",
            )

    bullet = await storage.restore_bullet(context_id, bullet_id)
    if bullet is None:
        raise HTTPException(
            status_code=404,
            detail=f"Bullet {bullet_id} not found or not in archived state",
        )
    return {"restored": True, "bullet_id": bullet_id}


@router.get("/{context_id}/archived-bullets", response_model=list[Bullet])
async def list_archived_bullets(
    context_id: str, request: Request,
    offset: int = 0, limit: int = 50,
) -> list[Bullet]:
    """Browse archived bullets with pagination."""
    storage = request.app.state.storage
    return await storage.get_archived_bullets(context_id, offset=offset, limit=limit)


@router.delete("/{context_id}/purge")
async def purge_context(context_id: str, request: Request) -> dict:
    """Permanently delete ALL data for a context (GDPR erasure)."""
    storage = request.app.state.storage

    # Use lock if available
    lock_manager = getattr(request.app.state, "lock_manager", None)
    if lock_manager is not None:
        async with lock_manager.acquire(context_id):
            success = await storage.purge_context(context_id)
        lock_manager.cleanup(context_id)
    else:
        success = await storage.purge_context(context_id)

    # Clean up event bus
    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus is not None:
        event_bus.cleanup(context_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")
    return {"purged": True, "context_id": context_id}


@router.get("/{context_id}/subscribe")
async def subscribe_events(context_id: str, request: Request) -> StreamingResponse:
    """SSE stream for real-time context events."""
    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus is None:
        raise HTTPException(status_code=501, detail="Event bus not available")

    queue = event_bus.subscribe(context_id)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = json.dumps(event.model_dump(mode="json"))
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(context_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/{context_id}/sync")
async def sync_context(
    context_id: str, request: Request, since: str | None = None,
) -> dict:
    """Polling alternative to SSE — get delta batches since a timestamp."""
    storage = request.app.state.storage
    batches = await storage.list_delta_batches(context_id, limit=100)

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            batches = [b for b in batches if b.timestamp > since_dt]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' timestamp")

    return {
        "delta_batches": [b.model_dump(mode="json") for b in batches],
    }


# ── Re-extraction (v0.4) ──────────────────────────────────────────────


@router.post("/{context_id}/re-extract")
async def re_extract(
    context_id: str, req: ReExtractionRequest, request: Request,
) -> dict:
    """Re-extract bullets from raw input history with a new Reflector model.

    Use dry_run=True to preview changes without applying them.
    """
    re_extraction_engine = getattr(request.app.state, "re_extraction", None)
    if re_extraction_engine is None:
        raise HTTPException(
            status_code=501,
            detail="Re-extraction engine not available",
        )

    storage = request.app.state.storage
    ctx = await storage.get_context(uuid.UUID(context_id))
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    result = await re_extraction_engine.re_extract(context_id, req)
    return result.model_dump(mode="json")


# ── Ingestion Config (v0.4) ──────────────────────────────────────────

config_router = APIRouter()


@config_router.get("/config/ingestion")
async def get_ingestion_config(request: Request) -> dict:
    """Get the current server-level ingestion configuration."""
    ingestion_config = getattr(request.app.state, "ingestion_config", None)
    if ingestion_config is None:
        ingestion_config = IngestionConfig()
    return ingestion_config.model_dump(mode="json")


@config_router.put("/config/ingestion")
async def update_ingestion_config(request: Request) -> dict:
    """Update server-level ingestion configuration.

    Note: changes only affect NEW commits. Existing bullets are NOT
    retroactively re-extracted. Use the re-extract endpoint for that.
    """
    body = await request.json()
    current = getattr(request.app.state, "ingestion_config", IngestionConfig())

    # Merge: only update fields that are present in the request body
    updated_data = current.model_dump()
    updated_data.update(body)
    new_config = IngestionConfig(**updated_data)

    # Store updated config on app state
    request.app.state.ingestion_config = new_config

    # Update engines that depend on config
    ingestion = getattr(request.app.state, "ingestion", None)
    if ingestion is not None:
        ingestion.ingestion_config = new_config
        if hasattr(ingestion, "reflector") and ingestion.reflector is not None:
            ingestion.reflector.config = new_config

    return {
        "updated": True,
        "config": new_config.model_dump(mode="json"),
    }


# ── User-level purge (registered at root, not under /contexts) ────────

user_router = APIRouter()


@user_router.delete("/users/{user_id}/purge")
async def purge_user(user_id: str, request: Request) -> dict:
    """Permanently delete ALL contexts owned by a user (GDPR erasure)."""
    storage = request.app.state.storage
    count = await storage.purge_user(user_id)
    return {"purged_contexts": count, "user_id": user_id}
