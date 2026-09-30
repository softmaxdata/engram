"""MCP (Model Context Protocol) server for Claude integration.

v0.2: Added save_bullet, list_bullets, consolidate, get_health tools.
Updated save_context to show bullet-based results.
v0.4: Added re_extract_context and get_ingestion_config tools.
v0.4.1: Added engram_ prefixed tool names per app.engram.so spec.

Run locally:   python -m engram.integrations.mcp_server
Run hosted:    Served at mcp.engram.so for cloud customers

Add to Claude:
  claude mcp add engram -- python -m engram.integrations.mcp_server

Or configure in claude_desktop_config.json:
  {
    "mcpServers": {
      "engram": {
        "command": "python",
        "args": ["-m", "engram.integrations.mcp_server"],
        "env": {
          "ENGRAM_API_URL": "http://localhost:5820",
          "ENGRAM_API_KEY": "eng_sk_..."
        }
      }
    }
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# MCP SDK is an optional dependency
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def create_mcp_server(engram_url: str = "http://localhost:5820") -> Any:
    """Create an MCP server that exposes Engram operations as tools for Claude.

    Requires the `mcp` extra: pip install engram[mcp]
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install engram[mcp]"
        )

    from engram.sdk.client import Engram

    server = Server("engram")
    client = Engram(url=engram_url)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="recall_context",
                description=(
                    "Recall relevant context from the Engram knowledge graph. "
                    "Returns context formatted for your model."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context to recall from",
                        },
                        "query": {
                            "type": "string",
                            "description": "What information do you need?",
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Maximum tokens for the response (default: 2000)",
                            "default": 2000,
                        },
                    },
                    "required": ["context_id", "query"],
                },
            ),
            Tool(
                name="save_context",
                description=(
                    "Save conversation content to the Engram knowledge graph. "
                    "The system will extract insights via the Reflector → Curator pipeline. "
                    "Optionally pass materialization_id + outcome for reconsolidation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context to save to",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to save (conversation, findings, etc.)",
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Identifier for this agent (default: 'claude')",
                            "default": "claude",
                        },
                        "materialization_id": {
                            "type": "string",
                            "description": "ID from a previous recall — enables reconsolidation",
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["success", "failure", "partial", "unknown"],
                            "description": "How the recalled context performed (for reconsolidation)",
                        },
                    },
                    "required": ["context_id", "content"],
                },
            ),
            Tool(
                name="save_bullet",
                description=(
                    "Save a single knowledge bullet directly (bypass LLM pipeline). "
                    "Use for atomic, well-formed insights."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context",
                        },
                        "content": {
                            "type": "string",
                            "description": "The bullet content — one atomic insight",
                        },
                        "section": {
                            "type": "string",
                            "description": "Section grouping (e.g., 'architecture', 'debugging')",
                            "default": "general",
                        },
                        "bullet_type": {
                            "type": "string",
                            "enum": ["strategy", "warning", "fact", "procedure", "exception", "principle", "decision"],
                            "description": "Type of bullet",
                            "default": "fact",
                        },
                    },
                    "required": ["context_id", "content"],
                },
            ),
            Tool(
                name="record_decision",
                description="Record an explicit decision with rationale to the knowledge graph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context",
                        },
                        "decision": {
                            "type": "string",
                            "description": "What was decided",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this decision was made",
                        },
                        "alternatives_considered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Other options that were considered",
                            "default": [],
                        },
                    },
                    "required": ["context_id", "decision", "rationale"],
                },
            ),
            Tool(
                name="list_contexts",
                description="List available Engram contexts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "owner": {
                            "type": "string",
                            "description": "Filter by owner (optional)",
                        },
                    },
                },
            ),
            Tool(
                name="get_health",
                description="Get health metrics for a context — bullet stats, staleness, schemas.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name="archive_bullet",
                description="Archive a bullet — moves it to cold storage. Can be restored later.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context",
                        },
                        "bullet_id": {
                            "type": "string",
                            "description": "ID of the bullet to archive",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for archiving",
                            "default": "manual",
                        },
                    },
                    "required": ["context_id", "bullet_id"],
                },
            ),
            Tool(
                name="get_lifecycle",
                description="Get lifecycle status — capacity metrics and configuration for a context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name="re_extract_context",
                description=(
                    "Re-extract bullets from raw input history with a new Reflector model. "
                    "Like upgrading a compiler and recompiling from source. "
                    "Use dry_run=True to preview changes first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "UUID of the context to re-extract",
                        },
                        "reflector_model": {
                            "type": "string",
                            "description": "New Reflector model to use (e.g., 'claude-sonnet-4-20250514')",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, preview changes without applying (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name="get_ingestion_config",
                description="Get the server-level ingestion configuration (canonical Reflector model, etc.).",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            # ── engram_ prefixed tools (v0.4.1) ────────────────────────
            # These are the canonical tool names for the app.engram.so MCP spec.
            # They map to the same handlers as the original tools above.
            Tool(
                name="engram_list_contexts",
                description=(
                    "List all available Engram contexts. Use this to see what contexts "
                    "exist and their current health status before recalling or committing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "archived", "all"],
                            "description": "Filter by context status",
                            "default": "active",
                        },
                    },
                },
            ),
            Tool(
                name="engram_recall",
                description=(
                    "Recall relevant context from an Engram context for a given query. "
                    "Returns materialized knowledge bullets relevant to your current task. "
                    "The returned materialization_id should be passed back when committing "
                    "results to enable the reconsolidation feedback loop."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "ID of the context to recall from",
                        },
                        "query": {
                            "type": "string",
                            "description": "What knowledge you need — be specific",
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Max tokens for returned context (default 4000)",
                            "default": 4000,
                        },
                    },
                    "required": ["context_id", "query"],
                },
            ),
            Tool(
                name="engram_commit",
                description=(
                    "Save new knowledge, observations, or results to an Engram context. "
                    "Include execution feedback (success/failure) when possible — this "
                    "dramatically improves future context quality. If you previously "
                    "called engram_recall, include the materialization_id to enable "
                    "the reconsolidation feedback loop."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "ID of the context to commit to",
                        },
                        "content": {
                            "type": "string",
                            "description": "What happened, what you learned, what worked/failed",
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["conversation", "tool_output", "document"],
                            "default": "conversation",
                        },
                        "feedback": {
                            "type": "object",
                            "description": "Structured execution feedback",
                            "properties": {
                                "outcome": {
                                    "type": "string",
                                    "enum": ["success", "failure", "partial", "unknown"],
                                },
                                "error_message": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                        },
                        "materialization_id": {
                            "type": "string",
                            "description": "ID from a previous engram_recall call (for reconsolidation)",
                        },
                    },
                    "required": ["context_id", "content"],
                },
            ),
            Tool(
                name="engram_decide",
                description=(
                    "Record an important decision in an Engram context. Decisions are "
                    "stored as high-priority bullets that resist forgetting and are "
                    "always included in materialized context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "ID of the context",
                        },
                        "decision": {
                            "type": "string",
                            "description": "What was decided",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this decision was made",
                        },
                        "alternatives_considered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Other options that were considered and rejected",
                        },
                    },
                    "required": ["context_id", "decision", "rationale"],
                },
            ),
            Tool(
                name="engram_health",
                description=(
                    "Check the health and capacity of an Engram context. Shows "
                    "bullet count, salience distribution, stale bullet count, "
                    "contradiction count, and capacity usage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "ID of the context to check",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name="engram_create_context",
                description=(
                    "Create a new Engram context with an intent anchor. Every context "
                    "has an objective, success criteria, and optional constraints that "
                    "prevent drift over long-running agent sessions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Context name",
                        },
                        "objective": {
                            "type": "string",
                            "description": "What are we trying to achieve?",
                        },
                        "success_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "How do we know we're done?",
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Boundaries and limitations",
                        },
                    },
                    "required": ["name", "objective", "success_criteria"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            match name:
                case "recall_context":
                    text = await client.recall(
                        context_id=arguments["context_id"],
                        query=arguments["query"],
                        token_budget=arguments.get("token_budget", 2000),
                        target_model="claude",
                    )
                    return [TextContent(type="text", text=text)]

                case "save_context":
                    feedback = None
                    if arguments.get("outcome"):
                        feedback = {"outcome": arguments["outcome"]}
                    result = await client.commit(
                        context_id=arguments["context_id"],
                        agent_id=arguments.get("agent_id", "claude"),
                        content=arguments["content"],
                        feedback=feedback,
                        materialization_id=arguments.get("materialization_id"),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=(
                                f"Saved. Bullets: +{result.get('bullets_added', 0)} added, "
                                f"~{result.get('bullets_updated', 0)} updated, "
                                f"⊕{result.get('bullets_merged', 0)} merged. "
                                f"Delta batch: {result.get('delta_batch_id', 'N/A')}"
                            ),
                        )
                    ]

                case "save_bullet":
                    result = await client.add_bullet(
                        context_id=arguments["context_id"],
                        content=arguments["content"],
                        section=arguments.get("section", "general"),
                        bullet_type=arguments.get("bullet_type", "fact"),
                        agent_id="claude",
                    )
                    return [
                        TextContent(
                            type="text",
                            text=f"Bullet saved. ID: {result.get('bullet_id')}",
                        )
                    ]

                case "record_decision":
                    result = await client.record_decision(
                        context_id=arguments["context_id"],
                        decision=arguments["decision"],
                        rationale=arguments["rationale"],
                        alternatives_considered=arguments.get(
                            "alternatives_considered", []
                        ),
                        agent_id="claude",
                    )
                    return [
                        TextContent(
                            type="text",
                            text=f"Decision recorded. Bullet ID: {result.get('decision_id', 'N/A')}",
                        )
                    ]

                case "list_contexts":
                    contexts = await client.list_contexts(
                        owner=arguments.get("owner"),
                    )
                    if not contexts:
                        return [TextContent(type="text", text="No contexts found.")]
                    lines = []
                    for ctx in contexts:
                        lines.append(
                            f"- **{ctx['name']}** (ID: {ctx['id']}, "
                            f"bullets: {ctx.get('bullet_count', 0)}, "
                            f"schemas: {ctx.get('schema_count', 0)})"
                        )
                    return [TextContent(type="text", text="\n".join(lines))]

                case "get_health":
                    health = await client.get_health(
                        context_id=arguments["context_id"],
                    )
                    lines = [
                        f"**Context Health**: {health['context_id']}",
                        f"- Total bullets: {health['total_bullets']} ({health['active_bullets']} active, {health['archived_bullets']} archived)",
                        f"- Avg salience: {health['avg_salience']:.3f} (effective: {health['avg_effective_salience']:.3f})",
                        f"- Stale bullets: {health['stale_bullet_count']}",
                        f"- Schemas: {health['schema_count']}",
                        f"- Edges: {health['total_edges']}",
                    ]
                    if health.get("top_sections"):
                        lines.append("- Top sections: " + ", ".join(
                            f"{s['section']} ({s['count']})" for s in health["top_sections"][:5]
                        ))
                    return [TextContent(type="text", text="\n".join(lines))]

                case "archive_bullet":
                    result = await client.archive_bullet(
                        context_id=arguments["context_id"],
                        bullet_id=arguments["bullet_id"],
                        reason=arguments.get("reason", "manual"),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=f"Bullet {arguments['bullet_id']} archived. Reason: {result.get('reason', 'manual')}",
                        )
                    ]

                case "get_lifecycle":
                    lifecycle = await client.get_lifecycle(
                        context_id=arguments["context_id"],
                    )
                    cap = lifecycle.get("capacity", {})
                    lines = [
                        f"**Lifecycle Status**: {arguments['context_id']}",
                        f"- Active bullets: {cap.get('active_bullet_count', 0)}/{cap.get('max_active_bullets', 10000)}",
                        f"- Capacity: {cap.get('capacity_percent', 0):.1f}% ({cap.get('pressure_level', 'unknown')})",
                        f"- Archived bullets: {cap.get('archived_bullet_count', 0)}",
                        f"- Schemas: {cap.get('schema_count', 0)}",
                    ]
                    return [TextContent(type="text", text="\n".join(lines))]

                case "re_extract_context":
                    result = await client.re_extract(
                        context_id=arguments["context_id"],
                        reflector_model=arguments.get("reflector_model"),
                        dry_run=arguments.get("dry_run", True),
                    )
                    if arguments.get("dry_run", True):
                        lines = [
                            "**Re-extraction Preview**",
                            f"- Activities to process: {result.get('activities_to_process', 0)}",
                            f"- Estimated: +{result.get('estimated_bullets_added', 0)} added, "
                            f"~{result.get('estimated_bullets_updated', 0)} updated, "
                            f"-{result.get('estimated_bullets_removed', 0)} removed",
                            f"- Est. input tokens: {result.get('estimated_input_tokens', 0)}",
                        ]
                    else:
                        lines = [
                            "**Re-extraction Complete**",
                            f"- Activities processed: {result.get('activities_processed', 0)}",
                            f"- Bullets: +{result.get('bullets_added', 0)} added, "
                            f"~{result.get('bullets_updated', 0)} updated, "
                            f"-{result.get('bullets_removed', 0)} removed",
                            f"- Model: {result.get('new_extraction_model', 'N/A')}",
                            f"- Duration: {result.get('duration_seconds', 0)}s",
                        ]
                    return [TextContent(type="text", text="\n".join(lines))]

                case "get_ingestion_config":
                    config = await client.get_ingestion_config()
                    lines = [
                        "**Ingestion Config**",
                        f"- Reflector model: {config.get('reflector_model', 'N/A')}",
                        f"- Prompt version: {config.get('reflector_prompt_version', 'N/A')}",
                        f"- Max reflection rounds: {config.get('max_reflection_rounds', 'N/A')}",
                        f"- Dedup threshold: {config.get('curator_dedup_threshold', 'N/A')}",
                        f"- Embedding model: {config.get('embedding_model', 'N/A')}",
                    ]
                    return [TextContent(type="text", text="\n".join(lines))]

                # ── engram_ prefixed handlers (v0.4.1) ─────────────────
                case "engram_list_contexts":
                    contexts = await client.list_contexts(
                        owner=arguments.get("owner"),
                    )
                    if not contexts:
                        return [TextContent(type="text", text="No contexts found.")]
                    lines = []
                    for ctx in contexts:
                        health = "●" if ctx.get("bullet_count", 0) > 0 else "○"
                        lines.append(
                            f"{health} {ctx['name']} (id: {ctx['id']}) — "
                            f"{ctx.get('bullet_count', 0)} bullets, "
                            f"{ctx.get('schema_count', 0)} schemas"
                        )
                    return [TextContent(type="text", text="\n".join(lines))]

                case "engram_recall":
                    text = await client.recall(
                        context_id=arguments["context_id"],
                        query=arguments["query"],
                        token_budget=arguments.get("token_budget", 4000),
                        target_model="claude",
                    )
                    return [TextContent(type="text", text=text)]

                case "engram_commit":
                    feedback = arguments.get("feedback")
                    result = await client.commit(
                        context_id=arguments["context_id"],
                        agent_id="claude-mcp",
                        content=arguments["content"],
                        content_type=arguments.get("content_type", "conversation"),
                        feedback=feedback,
                        materialization_id=arguments.get("materialization_id"),
                        source_model="claude",
                    )
                    return [
                        TextContent(
                            type="text",
                            text=(
                                f"Committed: {result.get('bullets_added', 0)} added, "
                                f"{result.get('bullets_updated', 0)} updated, "
                                f"{result.get('bullets_merged', 0)} merged"
                            ),
                        )
                    ]

                case "engram_decide":
                    result = await client.record_decision(
                        context_id=arguments["context_id"],
                        decision=arguments["decision"],
                        rationale=arguments["rationale"],
                        alternatives_considered=arguments.get("alternatives_considered", []),
                        agent_id="claude-mcp",
                    )
                    return [
                        TextContent(
                            type="text",
                            text=f"Decision recorded (id: {result.get('decision_id', 'unknown')})",
                        )
                    ]

                case "engram_health":
                    health = await client.get_health(
                        context_id=arguments["context_id"],
                    )
                    cap = health.get("capacity") or {}
                    lines = [
                        f"Context Health: {arguments['context_id']}",
                        f"  Capacity: {cap.get('active_bullet_count', 0)}/{cap.get('max_active_bullets', 10000)} "
                        f"({cap.get('capacity_percent', 0):.0f}%) — {cap.get('pressure_level', 'unknown')}",
                        f"  Archived: {health.get('archived_bullets', 0)} bullets",
                        f"  Schemas: {health.get('schema_count', 0)}",
                        f"  Avg salience: {health.get('avg_salience', 0):.2f}",
                        f"  Stale bullets: {health.get('stale_bullet_count', 0)}",
                        f"  Last consolidation: {health.get('last_consolidation') or 'never'}",
                    ]
                    return [TextContent(type="text", text="\n".join(lines))]

                case "engram_create_context":
                    ctx = await client.create_context(
                        name=arguments["name"],
                        intent={
                            "objective": arguments["objective"],
                            "success_criteria": arguments["success_criteria"],
                            "constraints": arguments.get("constraints", []),
                        },
                    )
                    return [
                        TextContent(
                            type="text",
                            text=f"Created context '{ctx.name}' (id: {ctx.id})",
                        )
                    ]

                case _:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as exc:
            logger.error("MCP tool error (%s): %s", name, exc)
            return [TextContent(type="text", text=f"Error: {exc}")]

    return server


async def run_mcp_server(engram_url: str = "http://localhost:5820") -> None:
    """Run the MCP server over stdio."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install engram[mcp]"
        )

    server = create_mcp_server(engram_url)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
