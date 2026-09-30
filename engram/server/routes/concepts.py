"""Concept management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from engram.core.models import (
    AddConceptRequest,
    AddConceptResponse,
    ConceptNode,
    RecordDecisionRequest,
)

router = APIRouter()


@router.post("/{context_id}/concepts", response_model=AddConceptResponse, status_code=201)
async def add_concept(
    context_id: uuid.UUID, req: AddConceptRequest, request: Request
) -> AddConceptResponse:
    """Directly add a concept to the graph (no LLM extraction)."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    concept = ConceptNode(
        type=req.type,
        content=req.content,
        salience=req.salience,
        confidence=req.confidence,
        domain_tags=req.domain_tags,
        source_agent=req.agent_id,
        source_session=req.session_id,
    )

    ingestion = request.app.state.ingestion
    relationships = [r.model_dump() for r in req.relationships] if req.relationships else None
    concept, edge_ids = await ingestion.add_concept_directly(
        context_id, concept, relationships
    )

    return AddConceptResponse(concept_id=concept.id, edges_created=edge_ids)


@router.get("/{context_id}/concepts")
async def list_concepts(
    context_id: uuid.UUID,
    request: Request,
    include_invalid: bool = False,
    type_filter: str | None = None,
) -> list[dict]:
    """List concepts in a context."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    concepts = await storage.list_concepts(
        context_id, include_invalid=include_invalid, type_filter=type_filter
    )
    # Exclude embeddings from response to keep it concise
    return [
        {**c.model_dump(mode="json", exclude={"embedding"})}
        for c in concepts
    ]


@router.delete("/{context_id}/concepts/{concept_id}")
async def invalidate_concept(
    context_id: uuid.UUID, concept_id: uuid.UUID, request: Request
) -> dict:
    """Soft-delete a concept by marking it invalid."""
    graph = request.app.state.graph
    invalidated = await graph.invalidate_concepts(
        context_id, [concept_id], "Manually invalidated via API"
    )
    if not invalidated:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found or already invalid")
    return {"status": "invalidated", "concept_id": str(concept_id)}


@router.post("/{context_id}/decisions")
async def record_decision(
    context_id: uuid.UUID, req: RecordDecisionRequest, request: Request
) -> dict:
    """Record a decision explicitly as a DECISION concept."""
    storage = request.app.state.storage
    ctx = await storage.get_context(context_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    ingestion = request.app.state.ingestion
    concept_id = await ingestion.record_decision(
        context_id=context_id,
        decision=req.decision,
        rationale=req.rationale,
        alternatives=req.alternatives_considered,
        agent_id=req.agent_id,
        domain_tags=req.domain_tags,
    )
    return {"concept_id": str(concept_id)}
