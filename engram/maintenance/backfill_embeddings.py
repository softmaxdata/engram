"""Repair missing memory vectors without re-extracting or changing memory text.

The command defaults to a read-only preview. It never loads .env, initializes a
schema, or chooses a database implicitly. Provider calls require --apply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from engram.core.models import Bullet, LifecycleState
from engram.llm.adapter import LiteLLMAdapter, LLMAdapter
from engram.llm.explicit import explicit_embedding_runtime
from engram.storage.base import StorageBackend
from engram.storage.postgres import PostgresBackend
from engram.storage.sqlite import SQLiteBackend

EMBEDDING_DIMENSIONS = 1536  # Matches the current PostgreSQL vector columns.
MAX_LIMIT = 1000


def _eligible(bullet: Bullet, context_id: str) -> bool:
    return (
        bullet.context_id == context_id
        and bullet.is_active
        and not bullet.is_archived
        and bullet.lifecycle_state == LifecycleState.ACTIVE
        and bullet.embedding is None
    )


def _normalize_vector(vector: Any) -> list[float] | None:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        return None
    if len(vector) != EMBEDDING_DIMENSIONS:
        return None
    try:
        import struct

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            return None
        # PostgreSQL stores float32. Validate the representation that will be
        # persisted, including underflow-to-zero and overflow, on both backends.
        normalized = list(struct.unpack(f"{EMBEDDING_DIMENSIONS}f", struct.pack(
            f"{EMBEDDING_DIMENSIONS}f", *vector,
        )))
        # Cosine search also accumulates squared components in float32. A
        # representable nonzero component can still square to zero or infinity,
        # making the saved vector unusable for search. Preserve provider scale
        # but reject that representation before marking a memory repaired.
        squared_norm = 0.0
        for value in normalized:
            squared = struct.unpack("f", struct.pack("f", value * value))[0]
            squared_norm = struct.unpack("f", struct.pack("f", squared_norm + squared))[0]
        return normalized if math.isfinite(squared_norm) and squared_norm > 0 else None
    except (OverflowError, TypeError, ValueError, struct.error):
        return None


async def backfill_embeddings(
    storage: StorageBackend,
    context_id: uuid.UUID,
    *,
    llm: LLMAdapter | None = None,
    apply: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Preview or repair up to ``limit`` eligible memories in one context.

    Provider calls happen outside transactions. Each conditional vector write
    is atomic; successful earlier repairs remain useful if another item fails.
    Re-running skips repaired items. Error details intentionally omit provider
    messages, which can contain credentials or memory text.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    if apply and llm is None:
        raise ValueError("apply requires an embedding adapter")
    if await storage.get_context(context_id) is None:
        raise ValueError("context does not exist")
    ctx = str(context_id)
    eligible = sorted(
        (bullet for bullet in await storage.list_bullets(ctx) if _eligible(bullet, ctx)),
        key=lambda bullet: bullet.id,
    )
    selected = eligible[:limit]
    result: dict[str, Any] = {
        "context_id": ctx,
        "dry_run": not apply,
        "eligible": len(eligible),
        "selected": len(selected),
        "attempted": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    if not apply:
        return result

    assert llm is not None
    for original in selected:
        result["attempted"] += 1
        try:
            vector = await llm.embed(original.content)
        except Exception:
            result["failed"] += 1
            result["errors"].append({"bullet_id": original.id, "reason": "provider_error"})
            continue
        vector = _normalize_vector(vector)
        if vector is None:
            result["failed"] += 1
            result["errors"].append({"bullet_id": original.id, "reason": "invalid_vector"})
            continue
        try:
            async with storage.transaction(ctx) as tx:
                current = await tx.get_bullet(original.id)
                if (
                    current is None
                    or not _eligible(current, ctx)
                    or current.content != original.content
                ):
                    updated = False
                else:
                    updated = await tx.update_bullet_embedding_if_missing(
                        ctx, current.id, original.content, list(vector),
                    )
            result["updated" if updated else "skipped"] += 1
        except Exception:
            result["failed"] += 1
            result["errors"].append({"bullet_id": original.id, "reason": "storage_error"})
    return result


def _positive_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc
    if not 1 <= parsed <= MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {MAX_LIMIT}")
    return parsed


def _env_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise argparse.ArgumentTypeError("expected an environment-variable name")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    database = parser.add_mutually_exclusive_group(required=True)
    database.add_argument("--sqlite", metavar="EXISTING_PATH", help="existing SQLite database")
    database.add_argument(
        "--postgres-dsn-env", type=_env_name, metavar="ENV_NAME",
        help="environment variable containing the existing PostgreSQL database DSN",
    )
    parser.add_argument("--context-id", type=uuid.UUID, required=True)
    parser.add_argument("--limit", type=_positive_limit, default=100)
    parser.add_argument("--apply", action="store_true", help="call the provider and save vectors")
    parser.add_argument("--embedding-model", help="explicit embedding model for --apply")
    parser.add_argument(
        "--embedding-key-env", type=_env_name, metavar="ENV_NAME",
        help="environment variable containing the embedding provider API key",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.sqlite:
        storage = SQLiteBackend(args.sqlite, read_only=not args.apply, existing_only=True)
    else:
        storage = PostgresBackend(
            os.environ[args.postgres_dsn_env], read_only=not args.apply, existing_only=True,
        )
    try:
        with explicit_embedding_runtime() if args.apply else nullcontext():
            llm = None
            if args.apply:
                llm = LiteLLMAdapter(
                    embedding_model=args.embedding_model,
                    embedding_api_key=os.environ[args.embedding_key_env],
                )
            result = await backfill_embeddings(
                storage, args.context_id, llm=llm, apply=args.apply, limit=args.limit,
            )
        print(json.dumps(result, sort_keys=True))
        return 1 if result["failed"] else 0
    finally:
        await storage.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.postgres_dsn_env and not os.environ.get(args.postgres_dsn_env):
        parser.error("the PostgreSQL DSN environment variable must be set")
    if args.apply:
        if not args.embedding_model or not args.embedding_key_env:
            parser.error("--apply requires --embedding-model and --embedding-key-env")
        if not os.environ.get(args.embedding_key_env):
            parser.error("the embedding API key environment variable must be set")
    try:
        return asyncio.run(_run(args))
    except ValueError:
        print(json.dumps({"error": "invalid_context_or_options"}))
        return 1
    except Exception:
        # A raw driver error can include the DSN, SQL, or a provider credential.
        print(json.dumps({"error": "database_operation_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
