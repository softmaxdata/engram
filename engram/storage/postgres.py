"""PostgreSQL storage backend for production deployment — v0.3 with lifecycle management.

Uses asyncpg for async PostgreSQL access and pgvector for embedding similarity search.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from engram.core.models import (
    Activity,
    Bullet,
    CapacityStatus,
    ConceptEdge,
    ConceptNode,
    Context,
    DeltaBatch,
    IntentAnchor,
    LifecycleConfig,
    LifecycleState,
    MaterializationRecord,
    SchemaNode,
    SlotDefinition,
)
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_rowcount(status: str) -> int:
    """Extract affected row count from asyncpg status string (e.g., 'UPDATE 3')."""
    try:
        return int(status.split()[-1])
    except (IndexError, ValueError):
        return 0


class PostgresBackend(StorageBackend):
    """PostgreSQL-based storage for production deployment with pgvector similarity search."""
    
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=10,
                init=self._init_connection,
            )
        return self._pool

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS contexts (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        owner TEXT DEFAULT 'default',
                        version INTEGER DEFAULT 1,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        lifecycle_config TEXT DEFAULT '{}',
                        core_memory TEXT DEFAULT ''
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS intents (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL UNIQUE,
                        objective TEXT NOT NULL,
                        success_criteria TEXT DEFAULT '[]',
                        constraints TEXT DEFAULT '[]',
                        status TEXT DEFAULT 'active',
                        sub_intents TEXT DEFAULT '[]',
                        progress_notes TEXT DEFAULT '[]',
                        created_at TIMESTAMPTZ NOT NULL,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS concepts (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1536),
                        confidence REAL DEFAULT 0.8,
                        salience REAL DEFAULT 0.5,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ,
                        version INTEGER DEFAULT 1,
                        source_session TEXT,
                        source_agent TEXT,
                        domain_tags TEXT DEFAULT '[]',
                        metadata TEXT DEFAULT '{}',
                        is_valid BOOLEAN DEFAULT TRUE,
                        invalidated_at TIMESTAMPTZ,
                        invalidation_reason TEXT,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS bullets (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        section TEXT DEFAULT 'general',
                        content TEXT NOT NULL,
                        bullet_type TEXT DEFAULT 'fact',
                        source_type TEXT DEFAULT 'reflection',
                        embedding vector(1536),
                        hit_count INTEGER DEFAULT 0,
                        miss_count INTEGER DEFAULT 0,
                        recall_count INTEGER DEFAULT 0,
                        last_recalled_at TIMESTAMPTZ,
                        salience REAL DEFAULT 0.5,
                        confidence REAL DEFAULT 0.5,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        source_session TEXT,
                        source_agent TEXT,
                        parent_concept_id TEXT,
                        schema_id TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        is_archived BOOLEAN DEFAULT FALSE,
                        archived_at TIMESTAMPTZ,
                        lifecycle_state TEXT DEFAULT 'active',
                        archive_reason TEXT,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS schemas (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        slots TEXT DEFAULT '{}',
                        typical_values TEXT DEFAULT '{}',
                        exceptions TEXT DEFAULT '[]',
                        instance_count INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        bullet_ids TEXT DEFAULT '[]',
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS delta_batches (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        operations TEXT DEFAULT '[]',
                        trigger TEXT DEFAULT 'commit',
                        timestamp TIMESTAMPTZ NOT NULL,
                        bullets_added INTEGER DEFAULT 0,
                        bullets_updated INTEGER DEFAULT 0,
                        bullets_removed INTEGER DEFAULT 0,
                        bullets_merged INTEGER DEFAULT 0,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS materializations (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        bullets_included TEXT DEFAULT '[]',
                        token_count INTEGER DEFAULT 0,
                        target_model TEXT DEFAULT 'claude',
                        query TEXT,
                        timestamp TIMESTAMPTZ NOT NULL,
                        reconsolidated_at TIMESTAMPTZ,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS edges (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        from_node TEXT NOT NULL,
                        to_node TEXT NOT NULL,
                        type TEXT NOT NULL,
                        weight REAL DEFAULT 0.5,
                        rationale TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        source_session TEXT,
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS activities (
                        id TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        agent_id TEXT NOT NULL,
                        session_id TEXT,
                        action_type TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        concepts_created TEXT DEFAULT '[]',
                        concepts_updated TEXT DEFAULT '[]',
                        concepts_invalidated TEXT DEFAULT '[]',
                        delta_batch_id TEXT,
                        materialization_id TEXT,
                        raw_input TEXT,
                        raw_input_hash TEXT,
                        content_type TEXT,
                        source_agent_model TEXT,
                        feedback TEXT,
                        extraction_model TEXT,
                        extraction_prompt_version TEXT,
                        bullet_ids_produced TEXT DEFAULT '[]',
                        FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
                    )
                """)

                # Indexes
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_concepts_context ON concepts(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_concepts_valid ON concepts(is_valid)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bullets_context ON bullets(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bullets_section ON bullets(section)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bullets_active ON bullets(is_active)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_schemas_context ON schemas(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_delta_batches_context ON delta_batches(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_materializations_context ON materializations(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_context ON edges(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_activities_context ON activities(context_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON activities(timestamp)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bullets_lifecycle ON bullets(lifecycle_state)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bullets_archived_at ON bullets(archived_at)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_activities_raw_input_hash "
                    "ON activities(context_id, raw_input_hash)"
                )

        # Schema migrations for existing databases
        await self._migrate_v03(pool)
        await self._migrate_v04(pool)
        await self._migrate_v05(pool)

        safe_dsn = self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn
        logger.info("PostgreSQL storage initialized at %s", safe_dsn)

    async def _column_exists(
        self, pool: asyncpg.Pool, table: str, column: str
    ) -> bool:
        row = await pool.fetchrow(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name=$1 AND column_name=$2",
            table, column,
        )
        return row is not None

    async def _migrate_v03(self, pool: asyncpg.Pool) -> None:
        """Add v0.3 lifecycle columns if they don't exist."""
        if not await self._column_exists(pool, "bullets", "lifecycle_state"):
            await pool.execute(
                "ALTER TABLE bullets ADD COLUMN lifecycle_state TEXT DEFAULT 'active'"
            )
            logger.info("Added lifecycle_state column to bullets")

        if not await self._column_exists(pool, "bullets", "archive_reason"):
            await pool.execute(
                "ALTER TABLE bullets ADD COLUMN archive_reason TEXT"
            )
            logger.info("Added archive_reason column to bullets")

        if not await self._column_exists(pool, "contexts", "lifecycle_config"):
            await pool.execute(
                "ALTER TABLE contexts ADD COLUMN lifecycle_config TEXT DEFAULT '{}'"
            )
            logger.info("Added lifecycle_config column to contexts")

        await pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_bullets_lifecycle ON bullets(lifecycle_state)"
        )
        await pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_bullets_archived_at ON bullets(archived_at)"
        )

    async def _migrate_v04(self, pool: asyncpg.Pool) -> None:
        """Add v0.4 raw input preservation columns to activities."""
        new_cols = [
            ("raw_input", "TEXT"),
            ("raw_input_hash", "TEXT"),
            ("content_type", "TEXT"),
            ("source_agent_model", "TEXT"),
            ("feedback", "TEXT"),
            ("extraction_model", "TEXT"),
            ("extraction_prompt_version", "TEXT"),
            ("bullet_ids_produced", "TEXT DEFAULT '[]'"),
        ]
        for col_name, col_type in new_cols:
            if not await self._column_exists(pool, "activities", col_name):
                await pool.execute(
                    f"ALTER TABLE activities ADD COLUMN {col_name} {col_type}"  # noqa: S608
                )
                logger.info("Added %s column to activities", col_name)

        await pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_activities_raw_input_hash "
            "ON activities(context_id, raw_input_hash)"
        )

    async def _migrate_v05(self, pool: asyncpg.Pool) -> None:
        """v0.5: Mem-α core_memory on contexts, DC activity embedding column."""
        if not await self._column_exists(pool, "contexts", "core_memory"):
            await pool.execute(
                "ALTER TABLE contexts ADD COLUMN core_memory TEXT DEFAULT ''"
            )
            logger.info("Added core_memory column to contexts")
        if not await self._column_exists(pool, "activities", "raw_input_embedding"):
            await pool.execute(
                "ALTER TABLE activities ADD COLUMN raw_input_embedding TEXT"
            )
            logger.info("Added raw_input_embedding column to activities")
        if not await self._column_exists(pool, "materializations", "reconsolidated_at"):
            await pool.execute(
                "ALTER TABLE materializations ADD COLUMN reconsolidated_at TIMESTAMPTZ"
            )
            logger.info("Added reconsolidated_at column to materializations")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── Context CRUD ───────────────────────────────────────────────────

    async def create_context(self, context: Context) -> Context:
        pool = await self._get_pool()
        now = _utcnow()
        lifecycle_json = context.lifecycle_config.model_dump_json()
        await pool.execute(
            "INSERT INTO contexts (id, name, description, owner, version, "
            "created_at, updated_at, lifecycle_config, core_memory) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            str(context.id), context.name, context.description, context.owner,
            context.version, context.created_at, now, lifecycle_json,
            context.core_memory,
        )
        intent = context.intent
        intent.context_id = context.id
        await pool.execute(
            "INSERT INTO intents (id, context_id, objective, success_criteria, "
            "constraints, status, sub_intents, progress_notes, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            str(intent.id), str(context.id), intent.objective,
            json.dumps(intent.success_criteria), json.dumps(intent.constraints),
            intent.status.value, json.dumps([]), json.dumps(intent.progress_notes),
            intent.created_at,
        )
        return context

    async def get_context(self, context_id: uuid.UUID) -> Context | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM contexts WHERE id = $1", str(context_id)
        )
        if row is None:
            return None
        intent = await self.get_intent(context_id)
        if intent is None:
            return None
        lc_raw = row["lifecycle_config"] if "lifecycle_config" in row.keys() else "{}"
        lifecycle_config = LifecycleConfig(**json.loads(lc_raw or "{}"))
        core_memory = row["core_memory"] if "core_memory" in row.keys() else ""
        return Context(
            id=uuid.UUID(row["id"]), name=row["name"], description=row["description"],
            owner=row["owner"], intent=intent, version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lifecycle_config=lifecycle_config,
            core_memory=core_memory or "",
        )

    async def list_contexts(
        self, owner: str | None = None, status: str | None = None
    ) -> list[Context]:
        pool = await self._get_pool()
        query = "SELECT c.* FROM contexts c"
        params: list[Any] = []
        conditions: list[str] = []
        param_idx = 1
        if owner is not None:
            conditions.append(f"c.owner = ${param_idx}")
            params.append(owner)
            param_idx += 1
        if status is not None:
            query += " JOIN intents i ON i.context_id = c.id"
            conditions.append(f"i.status = ${param_idx}")
            params.append(status)
            param_idx += 1
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.updated_at DESC"
        rows = await pool.fetch(query, *params)
        contexts: list[Context] = []
        for row in rows:
            ctx_id = uuid.UUID(row["id"])
            intent = await self.get_intent(ctx_id)
            if intent is not None:
                lc_raw = row["lifecycle_config"] if "lifecycle_config" in row.keys() else "{}"
                lifecycle_config = LifecycleConfig(**json.loads(lc_raw or "{}"))
                core_memory = row["core_memory"] if "core_memory" in row.keys() else ""
                contexts.append(Context(
                    id=ctx_id, name=row["name"], description=row["description"],
                    owner=row["owner"], intent=intent, version=row["version"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    lifecycle_config=lifecycle_config,
                    core_memory=core_memory or "",
                ))
        return contexts

    async def update_context(self, context: Context) -> Context:
        pool = await self._get_pool()
        now = _utcnow()
        lifecycle_json = context.lifecycle_config.model_dump_json()
        await pool.execute(
            "UPDATE contexts SET name=$1, description=$2, owner=$3, version=$4, "
            "updated_at=$5, lifecycle_config=$6, core_memory=$7 WHERE id=$8",
            context.name, context.description, context.owner, context.version,
            now, lifecycle_json, context.core_memory, str(context.id),
        )
        context.updated_at = now
        return context

    async def update_core_memory(
        self, context_id: str, core_memory: str
    ) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE contexts SET core_memory = $1, updated_at = $2 WHERE id = $3",
            core_memory, _utcnow(), context_id,
        )

    async def delete_context(self, context_id: uuid.UUID) -> None:
        pool = await self._get_pool()
        await pool.execute("DELETE FROM contexts WHERE id = $1", str(context_id))

    # ── Intent ─────────────────────────────────────────────────────────

    async def get_intent(self, context_id: uuid.UUID) -> IntentAnchor | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM intents WHERE context_id = $1", str(context_id)
        )
        if row is None:
            return None
        return IntentAnchor(
            id=uuid.UUID(row["id"]), context_id=uuid.UUID(row["context_id"]),
            objective=row["objective"],
            success_criteria=json.loads(row["success_criteria"]),
            constraints=json.loads(row["constraints"]),
            status=row["status"], sub_intents=json.loads(row["sub_intents"]),
            progress_notes=json.loads(row["progress_notes"]),
            created_at=row["created_at"],
        )

    async def update_intent(self, intent: IntentAnchor) -> IntentAnchor:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE intents SET status=$1, progress_notes=$2 WHERE id=$3",
            intent.status.value, json.dumps(intent.progress_notes), str(intent.id),
        )
        return intent

    # ── Bullets (v0.2) ─────────────────────────────────────────────────

    async def add_bullet(self, context_id: str, bullet: Bullet) -> Bullet:
        pool = await self._get_pool()
        bullet.context_id = context_id
        vec = np.array(bullet.embedding, dtype=np.float32) if bullet.embedding else None
        await pool.execute(
            "INSERT INTO bullets (id, context_id, section, content, bullet_type, source_type, "
            "embedding, hit_count, miss_count, recall_count, last_recalled_at, salience, "
            "confidence, created_at, updated_at, source_session, source_agent, "
            "parent_concept_id, schema_id, is_active, is_archived, archived_at, "
            "lifecycle_state, archive_reason) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
            "$13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)",
            bullet.id, context_id, bullet.section, bullet.content,
            bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else bullet.bullet_type,
            bullet.source_type.value if hasattr(bullet.source_type, 'value') else bullet.source_type,
            vec,
            bullet.hit_count, bullet.miss_count, bullet.recall_count,
            bullet.last_recalled_at,
            bullet.salience, bullet.confidence,
            bullet.created_at, bullet.updated_at,
            bullet.source_session, bullet.source_agent,
            bullet.parent_concept_id, bullet.schema_id,
            bullet.is_active, bullet.is_archived,
            bullet.archived_at,
            bullet.lifecycle_state.value, bullet.archive_reason,
        )
        return bullet

    async def get_bullet(self, bullet_id: str) -> Bullet | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM bullets WHERE id = $1", bullet_id)
        return self._row_to_bullet(row) if row else None

    async def list_bullets(
        self, context_id: str, section: str | None = None,
        bullet_type: str | None = None, include_archived: bool = False,
        min_salience: float | None = None,
    ) -> list[Bullet]:
        pool = await self._get_pool()
        query = "SELECT * FROM bullets WHERE context_id = $1 AND is_active = TRUE"
        params: list[Any] = [context_id]
        param_idx = 2
        if not include_archived:
            query += " AND is_archived = FALSE"
        if section:
            query += f" AND section = ${param_idx}"
            params.append(section)
            param_idx += 1
        if bullet_type:
            query += f" AND bullet_type = ${param_idx}"
            params.append(bullet_type)
            param_idx += 1
        if min_salience is not None:
            query += f" AND salience >= ${param_idx}"
            params.append(min_salience)
            param_idx += 1
        query += " ORDER BY salience DESC, created_at DESC"
        rows = await pool.fetch(query, *params)
        return [self._row_to_bullet(row) for row in rows]

    async def update_bullet(self, bullet: Bullet) -> Bullet:
        pool = await self._get_pool()
        now = _utcnow()
        vec = np.array(bullet.embedding, dtype=np.float32) if bullet.embedding else None
        await pool.execute(
            "UPDATE bullets SET section=$1, content=$2, bullet_type=$3, source_type=$4, "
            "embedding=$5, hit_count=$6, miss_count=$7, recall_count=$8, last_recalled_at=$9, "
            "salience=$10, confidence=$11, updated_at=$12, schema_id=$13, is_active=$14, "
            "is_archived=$15, archived_at=$16, lifecycle_state=$17, archive_reason=$18 "
            "WHERE id=$19",
            bullet.section, bullet.content,
            bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else bullet.bullet_type,
            bullet.source_type.value if hasattr(bullet.source_type, 'value') else bullet.source_type,
            vec,
            bullet.hit_count, bullet.miss_count, bullet.recall_count,
            bullet.last_recalled_at,
            bullet.salience, bullet.confidence, now, bullet.schema_id,
            bullet.is_active, bullet.is_archived,
            bullet.archived_at,
            bullet.lifecycle_state.value, bullet.archive_reason,
            bullet.id,
        )
        bullet.updated_at = now
        return bullet

    async def remove_bullet(self, bullet_id: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE bullets SET is_active = FALSE WHERE id = $1", bullet_id
        )

    async def find_similar_bullets(
        self, context_id: str, embedding: list[float],
        limit: int = 10, threshold: float = 0.7,
    ) -> list[tuple[Bullet, float]]:
        pool = await self._get_pool()
        vec = np.array(embedding, dtype=np.float32)
        rows = await pool.fetch(
            """
            SELECT *, 1 - (embedding <=> $1::vector) AS similarity
            FROM bullets
            WHERE context_id = $2
              AND is_active = TRUE
              AND is_archived = FALSE
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> $1::vector) >= $3
            ORDER BY similarity DESC
            LIMIT $4
            """,
            vec, context_id, threshold, limit,
        )
        return [(self._row_to_bullet(row), row["similarity"]) for row in rows]

    async def count_bullets(self, context_id: str) -> int:
        pool = await self._get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM bullets "
            "WHERE context_id = $1 AND is_active = TRUE AND is_archived = FALSE",
            context_id,
        )
        return count or 0

    # ── Schemas (v0.2) ─────────────────────────────────────────────────

    async def add_schema(self, context_id: str, schema: SchemaNode) -> SchemaNode:
        pool = await self._get_pool()
        schema.context_id = context_id
        slots_json = json.dumps({
            k: v.model_dump() for k, v in schema.slots.items()
        })
        await pool.execute(
            "INSERT INTO schemas (id, context_id, name, description, slots, typical_values, "
            "exceptions, instance_count, confidence, bullet_ids, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
            schema.id, context_id, schema.name, schema.description, slots_json,
            json.dumps(schema.typical_values), json.dumps(schema.exceptions),
            schema.instance_count, schema.confidence, json.dumps(schema.bullet_ids),
            schema.created_at, schema.updated_at,
        )
        return schema

    async def get_schema(self, schema_id: str) -> SchemaNode | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM schemas WHERE id = $1", schema_id)
        return self._row_to_schema(row) if row else None

    async def list_schemas(self, context_id: str) -> list[SchemaNode]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM schemas WHERE context_id = $1 ORDER BY confidence DESC",
            context_id,
        )
        return [self._row_to_schema(row) for row in rows]

    async def update_schema(self, schema: SchemaNode) -> SchemaNode:
        pool = await self._get_pool()
        now = _utcnow()
        slots_json = json.dumps({
            k: v.model_dump() for k, v in schema.slots.items()
        })
        await pool.execute(
            "UPDATE schemas SET name=$1, description=$2, slots=$3, typical_values=$4, "
            "exceptions=$5, instance_count=$6, confidence=$7, bullet_ids=$8, "
            "updated_at=$9 WHERE id=$10",
            schema.name, schema.description, slots_json,
            json.dumps(schema.typical_values), json.dumps(schema.exceptions),
            schema.instance_count, schema.confidence, json.dumps(schema.bullet_ids),
            now, schema.id,
        )
        schema.updated_at = now
        return schema

    # ── Delta History (v0.2) ───────────────────────────────────────────

    async def save_delta_batch(self, delta_batch: DeltaBatch) -> DeltaBatch:
        pool = await self._get_pool()
        ops_json = json.dumps([op.model_dump(mode="json") for op in delta_batch.operations])
        await pool.execute(
            "INSERT INTO delta_batches (id, context_id, operations, trigger, timestamp, "
            "bullets_added, bullets_updated, bullets_removed, bullets_merged) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            delta_batch.id, delta_batch.context_id, ops_json, delta_batch.trigger,
            delta_batch.timestamp, delta_batch.bullets_added,
            delta_batch.bullets_updated, delta_batch.bullets_removed,
            delta_batch.bullets_merged,
        )
        return delta_batch

    async def get_delta_batch(self, delta_batch_id: str) -> DeltaBatch | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM delta_batches WHERE id = $1", delta_batch_id
        )
        return self._row_to_delta_batch(row) if row else None

    async def list_delta_batches(
        self, context_id: str, limit: int = 50, offset: int = 0
    ) -> list[DeltaBatch]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM delta_batches WHERE context_id = $1 "
            "ORDER BY timestamp DESC LIMIT $2 OFFSET $3",
            context_id, limit, offset,
        )
        return [self._row_to_delta_batch(row) for row in rows]

    # ── Materialization Tracking (v0.2) ────────────────────────────────

    async def save_materialization(
        self, record: MaterializationRecord
    ) -> MaterializationRecord:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO materializations (id, context_id, bullets_included, token_count, "
            "target_model, query, timestamp, reconsolidated_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            record.id, record.context_id, json.dumps(record.bullets_included),
            record.token_count, record.target_model, record.query,
            record.timestamp, record.reconsolidated_at,
        )
        return record

    async def get_materialization(
        self, materialization_id: str
    ) -> MaterializationRecord | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM materializations WHERE id = $1", materialization_id
        )
        if row is None:
            return None
        return MaterializationRecord(
            id=row["id"], context_id=row["context_id"],
            bullets_included=json.loads(row["bullets_included"]),
            token_count=row["token_count"], target_model=row["target_model"],
            query=row["query"],
            timestamp=row["timestamp"],
            reconsolidated_at=(
                row["reconsolidated_at"] if "reconsolidated_at" in row.keys() else None
            ),
        )

    async def mark_materialization_reconsolidated(
        self, materialization_id: str, when: datetime,
    ) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE materializations SET reconsolidated_at = $1 WHERE id = $2",
            when, materialization_id,
        )

    # ── Lifecycle Management (v0.3) ──────────────────────────────────

    async def archive_bullet(
        self, context_id: str, bullet_id: str, reason: str = "manual"
    ) -> bool:
        pool = await self._get_pool()
        now = _utcnow()
        status = await pool.execute(
            "UPDATE bullets SET lifecycle_state='archived', is_archived=TRUE, "
            "archived_at=$1, archive_reason=$2, updated_at=$3 "
            "WHERE id=$4 AND context_id=$5 AND lifecycle_state='active'",
            now, reason, now, bullet_id, context_id,
        )
        return _get_rowcount(status) > 0

    async def restore_bullet(self, context_id: str, bullet_id: str) -> Bullet | None:
        pool = await self._get_pool()
        now = _utcnow()
        status = await pool.execute(
            "UPDATE bullets SET lifecycle_state='active', is_archived=FALSE, "
            "archived_at=NULL, archive_reason=NULL, updated_at=$1 "
            "WHERE id=$2 AND context_id=$3 AND lifecycle_state='archived'",
            now, bullet_id, context_id,
        )
        if _get_rowcount(status) == 0:
            return None
        return await self.get_bullet(bullet_id)

    async def purge_bullet(self, context_id: str, bullet_id: str) -> bool:
        pool = await self._get_pool()
        # Delete connected edges first
        await pool.execute(
            "DELETE FROM edges WHERE context_id=$1 AND (from_node=$2 OR to_node=$3)",
            context_id, bullet_id, bullet_id,
        )
        status = await pool.execute(
            "DELETE FROM bullets WHERE id=$1 AND context_id=$2",
            bullet_id, context_id,
        )
        return _get_rowcount(status) > 0

    async def purge_expired_archives(
        self, context_id: str, purge_after_days: int = 180
    ) -> int:
        pool = await self._get_pool()
        cutoff = _utcnow() - timedelta(days=purge_after_days)

        # Get bullet IDs to purge (for edge cleanup)
        rows = await pool.fetch(
            "SELECT id FROM bullets WHERE context_id=$1 AND lifecycle_state='archived' "
            "AND archived_at IS NOT NULL AND archived_at < $2",
            context_id, cutoff,
        )
        bullet_ids = [row["id"] for row in rows]

        if not bullet_ids:
            return 0

        # Delete connected edges using ANY()
        await pool.execute(
            "DELETE FROM edges WHERE context_id=$1 AND "
            "(from_node = ANY($2) OR to_node = ANY($2))",
            context_id, bullet_ids,
        )

        # Delete the bullets
        status = await pool.execute(
            "DELETE FROM bullets WHERE context_id=$1 AND id = ANY($2)",
            context_id, bullet_ids,
        )
        return _get_rowcount(status)

    async def get_archived_bullets(
        self, context_id: str, offset: int = 0, limit: int = 50
    ) -> list[Bullet]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM bullets WHERE context_id=$1 AND lifecycle_state='archived' "
            "ORDER BY archived_at DESC LIMIT $2 OFFSET $3",
            context_id, limit, offset,
        )
        return [self._row_to_bullet(row) for row in rows]

    async def get_capacity_status(
        self, context_id: str, max_active_bullets: int = 10000
    ) -> CapacityStatus:
        pool = await self._get_pool()

        active_count = await pool.fetchval(
            "SELECT COUNT(*) FROM bullets "
            "WHERE context_id=$1 AND is_active=TRUE AND lifecycle_state='active'",
            context_id,
        ) or 0

        archived_count = await pool.fetchval(
            "SELECT COUNT(*) FROM bullets "
            "WHERE context_id=$1 AND lifecycle_state='archived'",
            context_id,
        ) or 0

        schema_count_val = await pool.fetchval(
            "SELECT COUNT(*) FROM schemas WHERE context_id=$1",
            context_id,
        ) or 0

        return CapacityStatus(
            active_bullet_count=active_count,
            max_active_bullets=max_active_bullets,
            archived_bullet_count=archived_count,
            schema_count=schema_count_val,
        )

    async def purge_context(self, context_id: str) -> bool:
        pool = await self._get_pool()
        # CASCADE foreign keys handle all child table deletions
        status = await pool.execute(
            "DELETE FROM contexts WHERE id=$1", context_id
        )
        return _get_rowcount(status) > 0

    async def purge_user(self, user_id: str) -> int:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT id FROM contexts WHERE owner=$1", user_id
        )
        count = 0
        for row in rows:
            if await self.purge_context(row["id"]):
                count += 1
        return count

    # ── Legacy Concepts ────────────────────────────────────────────────

    async def add_concept(
        self, context_id: uuid.UUID, concept: ConceptNode
    ) -> ConceptNode:
        pool = await self._get_pool()
        vec = np.array(concept.embedding, dtype=np.float32) if concept.embedding else None
        await pool.execute(
            "INSERT INTO concepts (id, context_id, type, content, embedding, confidence, "
            "salience, created_at, updated_at, expires_at, version, source_session, "
            "source_agent, domain_tags, metadata, is_valid, invalidated_at, "
            "invalidation_reason) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
            "$13, $14, $15, $16, $17, $18)",
            str(concept.id), str(context_id), concept.type.value, concept.content,
            vec,
            concept.confidence, concept.salience,
            concept.created_at, concept.updated_at,
            concept.expires_at,
            concept.version,
            str(concept.source_session) if concept.source_session else None,
            concept.source_agent, json.dumps(concept.domain_tags),
            json.dumps(concept.metadata), concept.is_valid,
            concept.invalidated_at,
            concept.invalidation_reason,
        )
        return concept

    async def get_concept(self, concept_id: uuid.UUID) -> ConceptNode | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM concepts WHERE id = $1", str(concept_id)
        )
        return self._row_to_concept(row) if row else None

    async def list_concepts(
        self, context_id: uuid.UUID, include_invalid: bool = False,
        type_filter: str | None = None, domain_tags: list[str] | None = None,
    ) -> list[ConceptNode]:
        pool = await self._get_pool()
        query = "SELECT * FROM concepts WHERE context_id = $1"
        params: list[Any] = [str(context_id)]
        param_idx = 2
        if not include_invalid:
            query += " AND is_valid = TRUE"
        if type_filter:
            query += f" AND type = ${param_idx}"
            params.append(type_filter)
            param_idx += 1
        query += " ORDER BY salience DESC, created_at DESC"
        rows = await pool.fetch(query, *params)
        concepts = [self._row_to_concept(row) for row in rows]
        if domain_tags:
            tag_set = set(domain_tags)
            concepts = [c for c in concepts if tag_set.intersection(set(c.domain_tags))]
        return concepts

    async def update_concept(self, concept: ConceptNode) -> ConceptNode:
        pool = await self._get_pool()
        now = _utcnow()
        vec = np.array(concept.embedding, dtype=np.float32) if concept.embedding else None
        await pool.execute(
            "UPDATE concepts SET content=$1, embedding=$2, confidence=$3, salience=$4, "
            "updated_at=$5, version=$6, domain_tags=$7, metadata=$8, is_valid=$9, "
            "invalidated_at=$10, invalidation_reason=$11 WHERE id=$12",
            concept.content, vec,
            concept.confidence, concept.salience, now, concept.version,
            json.dumps(concept.domain_tags), json.dumps(concept.metadata),
            concept.is_valid,
            concept.invalidated_at,
            concept.invalidation_reason, str(concept.id),
        )
        concept.updated_at = now
        return concept

    async def find_similar_concepts(
        self, context_id: uuid.UUID, embedding: list[float],
        limit: int = 10, threshold: float = 0.7,
    ) -> list[tuple[ConceptNode, float]]:
        pool = await self._get_pool()
        vec = np.array(embedding, dtype=np.float32)
        rows = await pool.fetch(
            """
            SELECT *, 1 - (embedding <=> $1::vector) AS similarity
            FROM concepts
            WHERE context_id = $2
              AND is_valid = TRUE
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> $1::vector) >= $3
            ORDER BY similarity DESC
            LIMIT $4
            """,
            vec, str(context_id), threshold, limit,
        )
        return [(self._row_to_concept(row), row["similarity"]) for row in rows]

    async def count_concepts(self, context_id: uuid.UUID) -> int:
        pool = await self._get_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM concepts WHERE context_id = $1 AND is_valid = TRUE",
            str(context_id),
        )
        return count or 0

    # ── Edges ──────────────────────────────────────────────────────────

    async def add_edge(
        self, context_id: uuid.UUID, edge: ConceptEdge
    ) -> ConceptEdge:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO edges (id, context_id, from_node, to_node, type, weight, "
            "rationale, created_at, source_session) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            str(edge.id), str(context_id), str(edge.from_node), str(edge.to_node),
            edge.type.value, edge.weight, edge.rationale, edge.created_at,
            str(edge.source_session) if edge.source_session else None,
        )
        return edge

    async def get_edges(
        self, context_id: uuid.UUID, node_id: uuid.UUID | None = None,
        edge_type: str | None = None,
    ) -> list[ConceptEdge]:
        pool = await self._get_pool()
        query = "SELECT * FROM edges WHERE context_id = $1"
        params: list[Any] = [str(context_id)]
        param_idx = 2
        if node_id is not None:
            query += f" AND (from_node = ${param_idx} OR to_node = ${param_idx + 1})"
            params.extend([str(node_id), str(node_id)])
            param_idx += 2
        if edge_type is not None:
            query += f" AND type = ${param_idx}"
            params.append(edge_type)
            param_idx += 1
        rows = await pool.fetch(query, *params)
        return [self._row_to_edge(row) for row in rows]

    async def delete_edge(self, edge_id: uuid.UUID) -> None:
        pool = await self._get_pool()
        await pool.execute("DELETE FROM edges WHERE id = $1", str(edge_id))

    # ── Activity Ledger ────────────────────────────────────────────────

    async def add_activity(
        self, context_id: uuid.UUID, activity: Activity
    ) -> Activity:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO activities (id, context_id, timestamp, agent_id, session_id, "
            "action_type, summary, concepts_created, concepts_updated, "
            "concepts_invalidated, delta_batch_id, materialization_id, "
            "raw_input, raw_input_hash, content_type, source_agent_model, feedback, "
            "extraction_model, extraction_prompt_version, bullet_ids_produced, "
            "raw_input_embedding"
            ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
            "$13, $14, $15, $16, $17, $18, $19, $20, $21)",
            str(activity.id), str(context_id), activity.timestamp,
            activity.agent_id,
            str(activity.session_id) if activity.session_id else None,
            activity.action_type.value, activity.summary,
            json.dumps([str(uid) for uid in activity.concepts_created]),
            json.dumps([str(uid) for uid in activity.concepts_updated]),
            json.dumps([str(uid) for uid in activity.concepts_invalidated]),
            activity.delta_batch_id, activity.materialization_id,
            # v0.4 fields
            activity.raw_input, activity.raw_input_hash,
            activity.content_type, activity.source_agent_model,
            json.dumps(activity.feedback) if activity.feedback else None,
            activity.extraction_model, activity.extraction_prompt_version,
            json.dumps(activity.bullet_ids_produced),
            # v0.5 worked-example retrieval
            json.dumps(activity.raw_input_embedding) if activity.raw_input_embedding else None,
        )
        return activity

    async def list_activities(
        self, context_id: uuid.UUID, limit: int = 50, offset: int = 0,
    ) -> list[Activity]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM activities WHERE context_id = $1 "
            "ORDER BY timestamp DESC LIMIT $2 OFFSET $3",
            str(context_id), limit, offset,
        )
        return [self._row_to_activity(row) for row in rows]

    # ── Raw Input Queries (v0.4) ───────────────────────────────────────

    async def get_activities_with_raw_input(
        self,
        context_id: str,
        since: datetime | None = None,
        content_type: str | None = None,
    ) -> list[Activity]:
        """Get activity records that have raw_input, for re-extraction."""
        pool = await self._get_pool()
        query = (
            "SELECT * FROM activities WHERE context_id=$1 "
            "AND raw_input IS NOT NULL AND raw_input != ''"
        )
        params: list[Any] = [context_id]
        param_idx = 2
        if since is not None:
            if isinstance(since, str):
                since = datetime.fromisoformat(since)
            query += f" AND timestamp >= ${param_idx}"
            params.append(since)
            param_idx += 1
        if content_type is not None:
            query += f" AND content_type = ${param_idx}"
            params.append(content_type)
            param_idx += 1
        query += " ORDER BY timestamp ASC"
        rows = await pool.fetch(query, *params)
        return [self._row_to_activity(row) for row in rows]

    async def get_raw_input_by_hash(
        self, context_id: str, raw_input_hash: str,
    ) -> Activity | None:
        """Check if this exact raw input has already been ingested (dedup)."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM activities WHERE context_id=$1 AND raw_input_hash=$2 LIMIT 1",
            context_id, raw_input_hash,
        )
        return self._row_to_activity(row) if row else None

    async def get_bullets_by_ids(
        self, context_id: str, bullet_ids: list[str],
    ) -> list[Bullet]:
        """Get multiple bullets by their IDs."""
        if not bullet_ids:
            return []
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM bullets WHERE context_id=$1 AND id = ANY($2)",
            context_id, bullet_ids,
        )
        return [self._row_to_bullet(row) for row in rows]

    async def find_similar_activities(
        self,
        context_id: str,
        embedding: list[float],
        limit: int = 3,
        threshold: float = 0.85,
        exclude_hash: str | None = None,
    ) -> list[tuple[Activity, float]]:
        """Cosine similarity over activity raw_input_embedding (JSON column).
        Loaded in-process — fine at typical activity-ledger scale. Switch to
        pgvector if a context routinely has >10k activities."""
        from engram.core.similarity import cosine_similarity

        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT * FROM activities WHERE context_id=$1 "
            "AND raw_input_embedding IS NOT NULL AND raw_input != ''",
            context_id,
        )

        scored: list[tuple[Activity, float]] = []
        for row in rows:
            activity = self._row_to_activity(row)
            if exclude_hash and activity.raw_input_hash == exclude_hash:
                continue
            if not activity.raw_input_embedding:
                continue
            sim = cosine_similarity(embedding, activity.raw_input_embedding)
            if sim >= threshold:
                scored.append((activity, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ── Row Converters ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_bullet(row: asyncpg.Record) -> Bullet:
        keys = row.keys()
        lifecycle_state_raw = row["lifecycle_state"] if "lifecycle_state" in keys else "active"
        archive_reason = row["archive_reason"] if "archive_reason" in keys else None
        emb = row["embedding"]
        return Bullet(
            id=row["id"], context_id=row["context_id"], section=row["section"],
            content=row["content"], bullet_type=row["bullet_type"],
            source_type=row["source_type"],
            embedding=emb.tolist() if emb is not None else None,
            hit_count=row["hit_count"], miss_count=row["miss_count"],
            recall_count=row["recall_count"],
            last_recalled_at=row["last_recalled_at"],
            salience=row["salience"], confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source_session=row["source_session"], source_agent=row["source_agent"],
            parent_concept_id=row["parent_concept_id"], schema_id=row["schema_id"],
            is_active=row["is_active"],
            is_archived=row["is_archived"],
            archived_at=row["archived_at"],
            lifecycle_state=LifecycleState(lifecycle_state_raw or "active"),
            archive_reason=archive_reason,
        )

    @staticmethod
    def _row_to_schema(row: asyncpg.Record) -> SchemaNode:
        raw_slots = json.loads(row["slots"])
        slots = {k: SlotDefinition(**v) for k, v in raw_slots.items()}
        return SchemaNode(
            id=row["id"], context_id=row["context_id"], name=row["name"],
            description=row["description"], slots=slots,
            typical_values=json.loads(row["typical_values"]),
            exceptions=json.loads(row["exceptions"]),
            instance_count=row["instance_count"], confidence=row["confidence"],
            bullet_ids=json.loads(row["bullet_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_delta_batch(row: asyncpg.Record) -> DeltaBatch:
        from engram.core.models import DeltaOperation
        ops_raw = json.loads(row["operations"])
        operations = [DeltaOperation(**op) for op in ops_raw]
        return DeltaBatch(
            id=row["id"], context_id=row["context_id"], operations=operations,
            trigger=row["trigger"],
            timestamp=row["timestamp"],
            bullets_added=row["bullets_added"], bullets_updated=row["bullets_updated"],
            bullets_removed=row["bullets_removed"], bullets_merged=row["bullets_merged"],
        )

    @staticmethod
    def _row_to_concept(row: asyncpg.Record) -> ConceptNode:
        emb = row["embedding"]
        return ConceptNode(
            id=uuid.UUID(row["id"]), type=row["type"], content=row["content"],
            embedding=emb.tolist() if emb is not None else None,
            confidence=row["confidence"], salience=row["salience"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            version=row["version"],
            source_session=uuid.UUID(row["source_session"]) if row["source_session"] else None,
            source_agent=row["source_agent"],
            domain_tags=json.loads(row["domain_tags"]),
            metadata=json.loads(row["metadata"]),
            is_valid=row["is_valid"],
            invalidated_at=row["invalidated_at"],
            invalidation_reason=row["invalidation_reason"],
        )

    @staticmethod
    def _row_to_edge(row: asyncpg.Record) -> ConceptEdge:
        return ConceptEdge(
            id=uuid.UUID(row["id"]), from_node=uuid.UUID(row["from_node"]),
            to_node=uuid.UUID(row["to_node"]), type=row["type"],
            weight=row["weight"], rationale=row["rationale"],
            created_at=row["created_at"],
            source_session=uuid.UUID(row["source_session"]) if row["source_session"] else None,
        )

    @staticmethod
    def _row_to_activity(row: asyncpg.Record) -> Activity:
        keys = row.keys()
        # v0.4 fields — backward compatible via key check
        raw_input = row["raw_input"] if "raw_input" in keys else None
        raw_input_hash = row["raw_input_hash"] if "raw_input_hash" in keys else None
        content_type = row["content_type"] if "content_type" in keys else None
        source_agent_model = row["source_agent_model"] if "source_agent_model" in keys else None
        feedback_raw = row["feedback"] if "feedback" in keys else None
        feedback = json.loads(feedback_raw) if feedback_raw else None
        extraction_model = row["extraction_model"] if "extraction_model" in keys else None
        extraction_prompt_version = (
            row["extraction_prompt_version"] if "extraction_prompt_version" in keys else None
        )
        bullet_ids_raw = row["bullet_ids_produced"] if "bullet_ids_produced" in keys else "[]"
        bullet_ids_produced = json.loads(bullet_ids_raw or "[]")
        emb_raw = row["raw_input_embedding"] if "raw_input_embedding" in keys else None
        raw_input_embedding = json.loads(emb_raw) if emb_raw else None

        return Activity(
            id=uuid.UUID(row["id"]),
            timestamp=row["timestamp"],
            agent_id=row["agent_id"],
            session_id=uuid.UUID(row["session_id"]) if row["session_id"] else None,
            action_type=row["action_type"], summary=row["summary"],
            concepts_created=[uuid.UUID(u) for u in json.loads(row["concepts_created"])],
            concepts_updated=[uuid.UUID(u) for u in json.loads(row["concepts_updated"])],
            concepts_invalidated=[uuid.UUID(u) for u in json.loads(row["concepts_invalidated"])],
            delta_batch_id=row["delta_batch_id"] if "delta_batch_id" in keys else None,
            materialization_id=row["materialization_id"] if "materialization_id" in keys else None,
            # v0.4 fields
            raw_input=raw_input,
            raw_input_hash=raw_input_hash,
            content_type=content_type,
            source_agent_model=source_agent_model,
            feedback=feedback,
            extraction_model=extraction_model,
            extraction_prompt_version=extraction_prompt_version,
            bullet_ids_produced=bullet_ids_produced,
            raw_input_embedding=raw_input_embedding,
        )
