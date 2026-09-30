"""Schema management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/{context_id}/schemas")
async def list_schemas(
    context_id: uuid.UUID, request: Request
) -> list[dict]:
    """List all schemas (abstract patterns) in a context."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    schemas = await storage.list_schemas(str(context_id))
    return [s.model_dump(mode="json") for s in schemas]


@router.get("/{context_id}/schemas/{schema_id}")
async def get_schema(
    context_id: uuid.UUID, schema_id: str, request: Request
) -> dict:
    """Get a single schema by ID."""
    storage = request.app.state.storage
    schema = await storage.get_schema(schema_id)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Schema {schema_id} not found")
    return schema.model_dump(mode="json")
