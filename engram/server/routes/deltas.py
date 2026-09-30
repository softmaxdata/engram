"""Delta operations API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/{context_id}/deltas")
async def list_deltas(
    context_id: uuid.UUID,
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List delta batches for a context (audit trail)."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    batches = await storage.list_delta_batches(
        str(context_id), limit=limit, offset=offset
    )
    return [b.model_dump(mode="json") for b in batches]


@router.get("/{context_id}/deltas/{delta_id}")
async def get_delta(
    context_id: uuid.UUID, delta_id: str, request: Request
) -> dict:
    """Get a specific delta batch by ID."""
    storage = request.app.state.storage
    batch = await storage.get_delta_batch(delta_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Delta batch {delta_id} not found")
    return batch.model_dump(mode="json")


@router.post("/{context_id}/deltas/{delta_id}/rollback")
async def rollback_delta(
    context_id: uuid.UUID, delta_id: str, request: Request
) -> dict:
    """Roll back a delta batch — undo its operations."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    delta_engine = request.app.state.delta_engine
    success = await delta_engine.rollback_batch(delta_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Delta batch {delta_id} not found or has no rollback data")

    return {"status": "rolled_back", "delta_batch_id": delta_id}
