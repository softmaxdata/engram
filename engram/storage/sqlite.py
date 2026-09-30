"""SQLite storage backend for local development — v0.3 with lifecycle management."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

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
from engram.core.similarity import cosine_similarity as _cosine_similarity
from engram.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteBackend(StorageBackend):
    """SQLite-based storage for local development without Docker dependencies."""

    def __init__(self, db_path: str = "./engram.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
        return self._db

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        db = await self._get_db()
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                owner TEXT DEFAULT 'default',
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                core_memory TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS intents (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                success_criteria TEXT DEFAULT '[]',
                constraints TEXT DEFAULT '[]',
                status TEXT DEFAULT 'active',
                sub_intents TEXT DEFAULT '[]',
                progress_notes TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                confidence REAL DEFAULT 0.8,
                salience REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                version INTEGER DEFAULT 1,
                source_session TEXT,
                source_agent TEXT,
                domain_tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                is_valid INTEGER DEFAULT 1,
                invalidated_at TEXT,
                invalidation_reason TEXT,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS bullets (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                section TEXT DEFAULT 'general',
                content TEXT NOT NULL,
                bullet_type TEXT DEFAULT 'fact',
                source_type TEXT DEFAULT 'reflection',
                embedding TEXT,
                hit_count INTEGER DEFAULT 0,
                miss_count INTEGER DEFAULT 0,
                recall_count INTEGER DEFAULT 0,
                last_recalled_at TEXT,
                salience REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_session TEXT,
                source_agent TEXT,
                parent_concept_id TEXT,
                schema_id TEXT,
                is_active INTEGER DEFAULT 1,
                is_archived INTEGER DEFAULT 0,
                archived_at TEXT,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS delta_batches (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                operations TEXT DEFAULT '[]',
                trigger TEXT DEFAULT 'commit',
                timestamp TEXT NOT NULL,
                bullets_added INTEGER DEFAULT 0,
                bullets_updated INTEGER DEFAULT 0,
                bullets_removed INTEGER DEFAULT 0,
                bullets_merged INTEGER DEFAULT 0,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS materializations (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                bullets_included TEXT DEFAULT '[]',
                token_count INTEGER DEFAULT 0,
                target_model TEXT DEFAULT 'claude',
                query TEXT,
                timestamp TEXT NOT NULL,
                reconsolidated_at TEXT,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                rationale TEXT,
                created_at TEXT NOT NULL,
                source_session TEXT,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS activities (
                id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT,
                action_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                concepts_created TEXT DEFAULT '[]',
                concepts_updated TEXT DEFAULT '[]',
                concepts_invalidated TEXT DEFAULT '[]',
                delta_batch_id TEXT,
                materialization_id TEXT,
                FOREIGN KEY (context_id) REFERENCES contexts(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_context ON concepts(context_id);
            CREATE INDEX IF NOT EXISTS idx_concepts_valid ON concepts(is_valid);
            CREATE INDEX IF NOT EXISTS idx_bullets_context ON bullets(context_id);
            CREATE INDEX IF NOT EXISTS idx_bullets_section ON bullets(section);
            CREATE INDEX IF NOT EXISTS idx_bullets_active ON bullets(is_active);
            CREATE INDEX IF NOT EXISTS idx_schemas_context ON schemas(context_id);
            CREATE INDEX IF NOT EXISTS idx_delta_batches_context ON delta_batches(context_id);
            CREATE INDEX IF NOT EXISTS idx_materializations_context ON materializations(context_id);
            CREATE INDEX IF NOT EXISTS idx_edges_context ON edges(context_id);
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node);
            CREATE INDEX IF NOT EXISTS idx_activities_context ON activities(context_id);
            CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON activities(timestamp);
            """
        )

        # v0.3 schema migration: add lifecycle columns
        await self._migrate_v03(db)

        # v0.4 schema migration: add raw input columns to activities
        await self._migrate_v04(db)

        # v0.5 schema migration: Mem-α core memory + DC worked-example activity embeddings
        await self._migrate_v05(db)

        await db.commit()
        logger.info("SQLite storage initialized at %s", self.db_path)

    async def _migrate_v03(self, db: aiosqlite.Connection) -> None:
        """Add v0.3 lifecycle columns if they don't exist."""
        # Check if lifecycle_state column exists on bullets
        cursor = await db.execute("PRAGMA table_info(bullets)")
        cols = {row[1] for row in await cursor.fetchall()}

        if "lifecycle_state" not in cols:
            await db.execute(
                "ALTER TABLE bullets ADD COLUMN lifecycle_state TEXT DEFAULT 'active'"
            )
            logger.info("Added lifecycle_state column to bullets")

        if "archive_reason" not in cols:
            await db.execute(
                "ALTER TABLE bullets ADD COLUMN archive_reason TEXT"
            )
            logger.info("Added archive_reason column to bullets")

        # Check if lifecycle_config column exists on contexts
        cursor = await db.execute("PRAGMA table_info(contexts)")
        ctx_cols = {row[1] for row in await cursor.fetchall()}

        if "lifecycle_config" not in ctx_cols:
            await db.execute(
                "ALTER TABLE contexts ADD COLUMN lifecycle_config TEXT DEFAULT '{}'"
            )
            logger.info("Added lifecycle_config column to contexts")

        # Add indexes for lifecycle queries
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_bullets_lifecycle ON bullets(lifecycle_state)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_bullets_archived_at ON bullets(archived_at)"
        )

    async def _migrate_v04(self, db: aiosqlite.Connection) -> None:
        """Add v0.4 raw input preservation columns to activities."""
        cursor = await db.execute("PRAGMA table_info(activities)")
        cols = {row[1] for row in await cursor.fetchall()}

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
            if col_name not in cols:
                await db.execute(
                    f"ALTER TABLE activities ADD COLUMN {col_name} {col_type}"
                )
                logger.info("Added %s column to activities", col_name)

        # Add index for hash-based dedup lookups
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_activities_raw_input_hash "
            "ON activities(context_id, raw_input_hash)"
        )

    async def _migrate_v05(self, db: aiosqlite.Connection) -> None:
        """Add v0.5 columns: core_memory on contexts, raw_input_embedding on activities."""
        cursor = await db.execute("PRAGMA table_info(contexts)")
        ctx_cols = {row[1] for row in await cursor.fetchall()}
        if "core_memory" not in ctx_cols:
            await db.execute(
                "ALTER TABLE contexts ADD COLUMN core_memory TEXT DEFAULT ''"
            )
            logger.info("Added core_memory column to contexts")

        cursor = await db.execute("PRAGMA table_info(activities)")
        act_cols = {row[1] for row in await cursor.fetchall()}
        if "raw_input_embedding" not in act_cols:
            await db.execute(
                "ALTER TABLE activities ADD COLUMN raw_input_embedding TEXT"
            )
            logger.info("Added raw_input_embedding column to activities")

        cursor = await db.execute("PRAGMA table_info(materializations)")
        mat_cols = {row[1] for row in await cursor.fetchall()}
        if "reconsolidated_at" not in mat_cols:
            await db.execute(
                "ALTER TABLE materializations ADD COLUMN reconsolidated_at TEXT"
            )
            logger.info("Added reconsolidated_at column to materializations")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── Context CRUD ───────────────────────────────────────────────────

    async def create_context(self, context: Context) -> Context:
        db = await self._get_db()
        now = _utcnow().isoformat()
        lifecycle_json = context.lifecycle_config.model_dump_json()
        await db.execute(
            "INSERT INTO contexts (id, name, description, owner, version, created_at, updated_at, "
            "lifecycle_config, core_memory) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(context.id), context.name, context.description, context.owner,
             context.version, context.created_at.isoformat(), now, lifecycle_json,
             context.core_memory),
        )
        intent = context.intent
        intent.context_id = context.id
        await db.execute(
            "INSERT INTO intents (id, context_id, objective, success_criteria, constraints, "
            "status, sub_intents, progress_notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(intent.id), str(context.id), intent.objective,
             json.dumps(intent.success_criteria), json.dumps(intent.constraints),
             intent.status.value, json.dumps([]), json.dumps(intent.progress_notes),
             intent.created_at.isoformat()),
        )
        await db.commit()
        return context

    async def get_context(self, context_id: uuid.UUID) -> Context | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM contexts WHERE id = ?", (str(context_id),))
        row = await cursor.fetchone()
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
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            lifecycle_config=lifecycle_config,
            core_memory=core_memory or "",
        )

    async def list_contexts(
        self, owner: str | None = None, status: str | None = None
    ) -> list[Context]:
        db = await self._get_db()
        query = "SELECT c.* FROM contexts c"
        params: list[Any] = []
        conditions: list[str] = []
        if owner is not None:
            conditions.append("c.owner = ?")
            params.append(owner)
        if status is not None:
            query += " JOIN intents i ON i.context_id = c.id"
            conditions.append("i.status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.updated_at DESC"
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
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
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    lifecycle_config=lifecycle_config,
                    core_memory=core_memory or "",
                ))
        return contexts

    async def update_context(self, context: Context) -> Context:
        db = await self._get_db()
        now = _utcnow().isoformat()
        lifecycle_json = context.lifecycle_config.model_dump_json()
        await db.execute(
            "UPDATE contexts SET name=?, description=?, owner=?, version=?, "
            "updated_at=?, lifecycle_config=?, core_memory=? WHERE id=?",
            (context.name, context.description, context.owner, context.version,
             now, lifecycle_json, context.core_memory, str(context.id)),
        )
        await db.commit()
        context.updated_at = datetime.fromisoformat(now)
        return context

    async def delete_context(self, context_id: uuid.UUID) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM contexts WHERE id = ?", (str(context_id),))
        await db.commit()

    async def update_core_memory(
        self, context_id: str, core_memory: str
    ) -> None:
        db = await self._get_db()
        await db.execute(
            "UPDATE contexts SET core_memory = ?, updated_at = ? WHERE id = ?",
            (core_memory, _utcnow().isoformat(), context_id),
        )
        await db.commit()

    # ── Intent ─────────────────────────────────────────────────────────

    async def get_intent(self, context_id: uuid.UUID) -> IntentAnchor | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM intents WHERE context_id = ?", (str(context_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return IntentAnchor(
            id=uuid.UUID(row["id"]), context_id=uuid.UUID(row["context_id"]),
            objective=row["objective"],
            success_criteria=json.loads(row["success_criteria"]),
            constraints=json.loads(row["constraints"]),
            status=row["status"], sub_intents=json.loads(row["sub_intents"]),
            progress_notes=json.loads(row["progress_notes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def update_intent(self, intent: IntentAnchor) -> IntentAnchor:
        db = await self._get_db()
        await db.execute(
            "UPDATE intents SET status=?, progress_notes=? WHERE id=?",
            (intent.status.value, json.dumps(intent.progress_notes), str(intent.id)),
        )
        await db.commit()
        return intent

    # ── Bullets (v0.2) ─────────────────────────────────────────────────

    async def add_bullet(self, context_id: str, bullet: Bullet) -> Bullet:
        db = await self._get_db()
        bullet.context_id = context_id
        await db.execute(
            "INSERT INTO bullets (id, context_id, section, content, bullet_type, source_type, "
            "embedding, hit_count, miss_count, recall_count, last_recalled_at, salience, "
            "confidence, created_at, updated_at, source_session, source_agent, "
            "parent_concept_id, schema_id, is_active, is_archived, archived_at, "
            "lifecycle_state, archive_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (bullet.id, context_id, bullet.section, bullet.content,
             bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else bullet.bullet_type,
             bullet.source_type.value if hasattr(bullet.source_type, 'value') else bullet.source_type,
             json.dumps(bullet.embedding) if bullet.embedding else None,
             bullet.hit_count, bullet.miss_count, bullet.recall_count,
             bullet.last_recalled_at.isoformat() if bullet.last_recalled_at else None,
             bullet.salience, bullet.confidence,
             bullet.created_at.isoformat(), bullet.updated_at.isoformat(),
             bullet.source_session, bullet.source_agent,
             bullet.parent_concept_id, bullet.schema_id,
             1 if bullet.is_active else 0, 1 if bullet.is_archived else 0,
             bullet.archived_at.isoformat() if bullet.archived_at else None,
             bullet.lifecycle_state.value, bullet.archive_reason),
        )
        await db.commit()
        return bullet

    async def get_bullet(self, bullet_id: str) -> Bullet | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM bullets WHERE id = ?", (bullet_id,))
        row = await cursor.fetchone()
        return self._row_to_bullet(row) if row else None

    async def list_bullets(
        self, context_id: str, section: str | None = None,
        bullet_type: str | None = None, include_archived: bool = False,
        min_salience: float | None = None,
    ) -> list[Bullet]:
        db = await self._get_db()
        query = "SELECT * FROM bullets WHERE context_id = ? AND is_active = 1"
        params: list[Any] = [context_id]
        if not include_archived:
            query += " AND is_archived = 0"
        if section:
            query += " AND section = ?"
            params.append(section)
        if bullet_type:
            query += " AND bullet_type = ?"
            params.append(bullet_type)
        if min_salience is not None:
            query += " AND salience >= ?"
            params.append(min_salience)
        query += " ORDER BY salience DESC, created_at DESC"
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_bullet(row) for row in rows]

    async def update_bullet(self, bullet: Bullet) -> Bullet:
        db = await self._get_db()
        now = _utcnow().isoformat()
        await db.execute(
            "UPDATE bullets SET section=?, content=?, bullet_type=?, source_type=?, "
            "embedding=?, hit_count=?, miss_count=?, recall_count=?, last_recalled_at=?, "
            "salience=?, confidence=?, updated_at=?, schema_id=?, is_active=?, "
            "is_archived=?, archived_at=?, lifecycle_state=?, archive_reason=? WHERE id=?",
            (bullet.section, bullet.content,
             bullet.bullet_type.value if hasattr(bullet.bullet_type, 'value') else bullet.bullet_type,
             bullet.source_type.value if hasattr(bullet.source_type, 'value') else bullet.source_type,
             json.dumps(bullet.embedding) if bullet.embedding else None,
             bullet.hit_count, bullet.miss_count, bullet.recall_count,
             bullet.last_recalled_at.isoformat() if bullet.last_recalled_at else None,
             bullet.salience, bullet.confidence, now, bullet.schema_id,
             1 if bullet.is_active else 0, 1 if bullet.is_archived else 0,
             bullet.archived_at.isoformat() if bullet.archived_at else None,
             bullet.lifecycle_state.value, bullet.archive_reason,
             bullet.id),
        )
        await db.commit()
        bullet.updated_at = datetime.fromisoformat(now)
        return bullet

    async def remove_bullet(self, bullet_id: str) -> None:
        db = await self._get_db()
        await db.execute("UPDATE bullets SET is_active = 0 WHERE id = ?", (bullet_id,))
        await db.commit()

    async def find_similar_bullets(
        self, context_id: str, embedding: list[float],
        limit: int = 10, threshold: float = 0.7,
    ) -> list[tuple[Bullet, float]]:
        bullets = await self.list_bullets(context_id)
        scored: list[tuple[Bullet, float]] = []
        for bullet in bullets:
            if bullet.embedding is None:
                continue
            sim = _cosine_similarity(embedding, bullet.embedding)
            if sim >= threshold:
                scored.append((bullet, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def count_bullets(self, context_id: str) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM bullets WHERE context_id = ? AND is_active = 1 AND is_archived = 0",
            (context_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    # ── Schemas (v0.2) ─────────────────────────────────────────────────

    async def add_schema(self, context_id: str, schema: SchemaNode) -> SchemaNode:
        db = await self._get_db()
        schema.context_id = context_id
        slots_json = json.dumps({
            k: v.model_dump() for k, v in schema.slots.items()
        })
        await db.execute(
            "INSERT INTO schemas (id, context_id, name, description, slots, typical_values, "
            "exceptions, instance_count, confidence, bullet_ids, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (schema.id, context_id, schema.name, schema.description, slots_json,
             json.dumps(schema.typical_values), json.dumps(schema.exceptions),
             schema.instance_count, schema.confidence, json.dumps(schema.bullet_ids),
             schema.created_at.isoformat(), schema.updated_at.isoformat()),
        )
        await db.commit()
        return schema

    async def get_schema(self, schema_id: str) -> SchemaNode | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM schemas WHERE id = ?", (schema_id,))
        row = await cursor.fetchone()
        return self._row_to_schema(row) if row else None

    async def list_schemas(self, context_id: str) -> list[SchemaNode]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM schemas WHERE context_id = ? ORDER BY confidence DESC", (context_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_schema(row) for row in rows]

    async def update_schema(self, schema: SchemaNode) -> SchemaNode:
        db = await self._get_db()
        now = _utcnow().isoformat()
        slots_json = json.dumps({
            k: v.model_dump() for k, v in schema.slots.items()
        })
        await db.execute(
            "UPDATE schemas SET name=?, description=?, slots=?, typical_values=?, "
            "exceptions=?, instance_count=?, confidence=?, bullet_ids=?, updated_at=? WHERE id=?",
            (schema.name, schema.description, slots_json,
             json.dumps(schema.typical_values), json.dumps(schema.exceptions),
             schema.instance_count, schema.confidence, json.dumps(schema.bullet_ids),
             now, schema.id),
        )
        await db.commit()
        schema.updated_at = datetime.fromisoformat(now)
        return schema

    # ── Delta History (v0.2) ───────────────────────────────────────────

    async def save_delta_batch(self, delta_batch: DeltaBatch) -> DeltaBatch:
        db = await self._get_db()
        ops_json = json.dumps([op.model_dump(mode="json") for op in delta_batch.operations])
        await db.execute(
            "INSERT INTO delta_batches (id, context_id, operations, trigger, timestamp, "
            "bullets_added, bullets_updated, bullets_removed, bullets_merged) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (delta_batch.id, delta_batch.context_id, ops_json, delta_batch.trigger,
             delta_batch.timestamp.isoformat(), delta_batch.bullets_added,
             delta_batch.bullets_updated, delta_batch.bullets_removed,
             delta_batch.bullets_merged),
        )
        await db.commit()
        return delta_batch

    async def get_delta_batch(self, delta_batch_id: str) -> DeltaBatch | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM delta_batches WHERE id = ?", (delta_batch_id,))
        row = await cursor.fetchone()
        return self._row_to_delta_batch(row) if row else None

    async def list_delta_batches(
        self, context_id: str, limit: int = 50, offset: int = 0
    ) -> list[DeltaBatch]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM delta_batches WHERE context_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (context_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_delta_batch(row) for row in rows]

    # ── Materialization Tracking (v0.2) ────────────────────────────────

    async def save_materialization(self, record: MaterializationRecord) -> MaterializationRecord:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO materializations (id, context_id, bullets_included, token_count, "
            "target_model, query, timestamp, reconsolidated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record.id, record.context_id, json.dumps(record.bullets_included),
             record.token_count, record.target_model, record.query,
             record.timestamp.isoformat(),
             record.reconsolidated_at.isoformat() if record.reconsolidated_at else None),
        )
        await db.commit()
        return record

    async def get_materialization(self, materialization_id: str) -> MaterializationRecord | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM materializations WHERE id = ?", (materialization_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        keys = row.keys()
        rec_at = row["reconsolidated_at"] if "reconsolidated_at" in keys else None
        return MaterializationRecord(
            id=row["id"], context_id=row["context_id"],
            bullets_included=json.loads(row["bullets_included"]),
            token_count=row["token_count"], target_model=row["target_model"],
            query=row["query"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            reconsolidated_at=datetime.fromisoformat(rec_at) if rec_at else None,
        )

    async def mark_materialization_reconsolidated(
        self, materialization_id: str, when: datetime,
    ) -> None:
        db = await self._get_db()
        await db.execute(
            "UPDATE materializations SET reconsolidated_at = ? WHERE id = ?",
            (when.isoformat(), materialization_id),
        )
        await db.commit()

    # ── Lifecycle Management (v0.3) ──────────────────────────────────

    async def archive_bullet(
        self, context_id: str, bullet_id: str, reason: str = "manual"
    ) -> bool:
        db = await self._get_db()
        now = _utcnow().isoformat()
        result = await db.execute(
            "UPDATE bullets SET lifecycle_state='archived', is_archived=1, "
            "archived_at=?, archive_reason=?, updated_at=? "
            "WHERE id=? AND context_id=? AND lifecycle_state='active'",
            (now, reason, now, bullet_id, context_id),
        )
        await db.commit()
        return result.rowcount > 0  # type: ignore[union-attr]

    async def restore_bullet(self, context_id: str, bullet_id: str) -> Bullet | None:
        db = await self._get_db()
        now = _utcnow().isoformat()
        result = await db.execute(
            "UPDATE bullets SET lifecycle_state='active', is_archived=0, "
            "archived_at=NULL, archive_reason=NULL, updated_at=? "
            "WHERE id=? AND context_id=? AND lifecycle_state='archived'",
            (now, bullet_id, context_id),
        )
        await db.commit()
        if result.rowcount == 0:  # type: ignore[union-attr]
            return None
        return await self.get_bullet(bullet_id)

    async def purge_bullet(self, context_id: str, bullet_id: str) -> bool:
        db = await self._get_db()
        # Delete connected edges first
        await db.execute(
            "DELETE FROM edges WHERE context_id=? AND (from_node=? OR to_node=?)",
            (context_id, bullet_id, bullet_id),
        )
        result = await db.execute(
            "DELETE FROM bullets WHERE id=? AND context_id=?",
            (bullet_id, context_id),
        )
        await db.commit()
        return result.rowcount > 0  # type: ignore[union-attr]

    async def purge_expired_archives(
        self, context_id: str, purge_after_days: int = 180
    ) -> int:
        db = await self._get_db()
        from datetime import timedelta
        cutoff = (_utcnow() - timedelta(days=purge_after_days)).isoformat()

        # Get bullet IDs to purge (for edge cleanup)
        cursor = await db.execute(
            "SELECT id FROM bullets WHERE context_id=? AND lifecycle_state='archived' "
            "AND archived_at IS NOT NULL AND archived_at < ?",
            (context_id, cutoff),
        )
        bullet_ids = [row[0] for row in await cursor.fetchall()]

        if not bullet_ids:
            return 0

        # Delete connected edges
        placeholders = ",".join("?" * len(bullet_ids))
        await db.execute(
            f"DELETE FROM edges WHERE context_id=? AND "
            f"(from_node IN ({placeholders}) OR to_node IN ({placeholders}))",
            [context_id] + bullet_ids + bullet_ids,
        )

        # Delete the bullets
        result = await db.execute(
            f"DELETE FROM bullets WHERE context_id=? AND id IN ({placeholders})",
            [context_id] + bullet_ids,
        )
        await db.commit()
        return result.rowcount or 0  # type: ignore[union-attr]

    async def get_archived_bullets(
        self, context_id: str, offset: int = 0, limit: int = 50
    ) -> list[Bullet]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM bullets WHERE context_id=? AND lifecycle_state='archived' "
            "ORDER BY archived_at DESC LIMIT ? OFFSET ?",
            (context_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_bullet(row) for row in rows]

    async def get_capacity_status(
        self, context_id: str, max_active_bullets: int = 10000
    ) -> CapacityStatus:
        db = await self._get_db()

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM bullets "
            "WHERE context_id=? AND is_active=1 AND lifecycle_state='active'",
            (context_id,),
        )
        active_row = await cursor.fetchone()
        active_count = active_row["cnt"] if active_row else 0

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM bullets "
            "WHERE context_id=? AND lifecycle_state='archived'",
            (context_id,),
        )
        archived_row = await cursor.fetchone()
        archived_count = archived_row["cnt"] if archived_row else 0

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM schemas WHERE context_id=?",
            (context_id,),
        )
        schema_row = await cursor.fetchone()
        schema_count_val = schema_row["cnt"] if schema_row else 0

        return CapacityStatus(
            active_bullet_count=active_count,
            max_active_bullets=max_active_bullets,
            archived_bullet_count=archived_count,
            schema_count=schema_count_val,
        )

    async def purge_context(self, context_id: str) -> bool:
        db = await self._get_db()
        # Delete from all tables for this context
        tables = [
            "activities", "edges", "materializations", "delta_batches",
            "schemas", "bullets", "concepts", "intents", "contexts",
        ]
        found = False
        for table in tables:
            col = "context_id" if table != "contexts" else "id"
            result = await db.execute(
                f"DELETE FROM {table} WHERE {col}=?", (context_id,)
            )
            if table == "contexts" and result.rowcount:  # type: ignore[union-attr]
                found = True
        await db.commit()
        return found

    async def purge_user(self, user_id: str) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT id FROM contexts WHERE owner=?", (user_id,),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            if await self.purge_context(row["id"]):
                count += 1
        return count

    # ── Legacy Concepts ────────────────────────────────────────────────

    async def add_concept(self, context_id: uuid.UUID, concept: ConceptNode) -> ConceptNode:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO concepts (id, context_id, type, content, embedding, confidence, "
            "salience, created_at, updated_at, expires_at, version, source_session, "
            "source_agent, domain_tags, metadata, is_valid, invalidated_at, invalidation_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(concept.id), str(context_id), concept.type.value, concept.content,
             json.dumps(concept.embedding) if concept.embedding else None,
             concept.confidence, concept.salience,
             concept.created_at.isoformat(), concept.updated_at.isoformat(),
             concept.expires_at.isoformat() if concept.expires_at else None,
             concept.version,
             str(concept.source_session) if concept.source_session else None,
             concept.source_agent, json.dumps(concept.domain_tags),
             json.dumps(concept.metadata), 1 if concept.is_valid else 0,
             concept.invalidated_at.isoformat() if concept.invalidated_at else None,
             concept.invalidation_reason),
        )
        await db.commit()
        return concept

    async def get_concept(self, concept_id: uuid.UUID) -> ConceptNode | None:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM concepts WHERE id = ?", (str(concept_id),))
        row = await cursor.fetchone()
        return self._row_to_concept(row) if row else None

    async def list_concepts(
        self, context_id: uuid.UUID, include_invalid: bool = False,
        type_filter: str | None = None, domain_tags: list[str] | None = None,
    ) -> list[ConceptNode]:
        db = await self._get_db()
        query = "SELECT * FROM concepts WHERE context_id = ?"
        params: list[Any] = [str(context_id)]
        if not include_invalid:
            query += " AND is_valid = 1"
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        query += " ORDER BY salience DESC, created_at DESC"
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        concepts = [self._row_to_concept(row) for row in rows]
        if domain_tags:
            tag_set = set(domain_tags)
            concepts = [c for c in concepts if tag_set.intersection(set(c.domain_tags))]
        return concepts

    async def update_concept(self, concept: ConceptNode) -> ConceptNode:
        db = await self._get_db()
        now = _utcnow().isoformat()
        await db.execute(
            "UPDATE concepts SET content=?, embedding=?, confidence=?, salience=?, "
            "updated_at=?, version=?, domain_tags=?, metadata=?, is_valid=?, "
            "invalidated_at=?, invalidation_reason=? WHERE id=?",
            (concept.content, json.dumps(concept.embedding) if concept.embedding else None,
             concept.confidence, concept.salience, now, concept.version,
             json.dumps(concept.domain_tags), json.dumps(concept.metadata),
             1 if concept.is_valid else 0,
             concept.invalidated_at.isoformat() if concept.invalidated_at else None,
             concept.invalidation_reason, str(concept.id)),
        )
        await db.commit()
        concept.updated_at = datetime.fromisoformat(now)
        return concept

    async def find_similar_concepts(
        self, context_id: uuid.UUID, embedding: list[float],
        limit: int = 10, threshold: float = 0.7,
    ) -> list[tuple[ConceptNode, float]]:
        concepts = await self.list_concepts(context_id, include_invalid=False)
        scored: list[tuple[ConceptNode, float]] = []
        for concept in concepts:
            if concept.embedding is None:
                continue
            sim = _cosine_similarity(embedding, concept.embedding)
            if sim >= threshold:
                scored.append((concept, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def count_concepts(self, context_id: uuid.UUID) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM concepts WHERE context_id = ? AND is_valid = 1",
            (str(context_id),),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    # ── Edges ──────────────────────────────────────────────────────────

    async def add_edge(self, context_id: uuid.UUID, edge: ConceptEdge) -> ConceptEdge:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO edges (id, context_id, from_node, to_node, type, weight, "
            "rationale, created_at, source_session) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(edge.id), str(context_id), str(edge.from_node), str(edge.to_node),
             edge.type.value, edge.weight, edge.rationale, edge.created_at.isoformat(),
             str(edge.source_session) if edge.source_session else None),
        )
        await db.commit()
        return edge

    async def get_edges(
        self, context_id: uuid.UUID, node_id: uuid.UUID | None = None,
        edge_type: str | None = None,
    ) -> list[ConceptEdge]:
        db = await self._get_db()
        query = "SELECT * FROM edges WHERE context_id = ?"
        params: list[Any] = [str(context_id)]
        if node_id is not None:
            query += " AND (from_node = ? OR to_node = ?)"
            params.extend([str(node_id), str(node_id)])
        if edge_type is not None:
            query += " AND type = ?"
            params.append(edge_type)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_edge(row) for row in rows]

    async def delete_edge(self, edge_id: uuid.UUID) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM edges WHERE id = ?", (str(edge_id),))
        await db.commit()

    # ── Activity Ledger ────────────────────────────────────────────────

    async def add_activity(self, context_id: uuid.UUID, activity: Activity) -> Activity:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO activities (id, context_id, timestamp, agent_id, session_id, "
            "action_type, summary, concepts_created, concepts_updated, concepts_invalidated, "
            "delta_batch_id, materialization_id, "
            "raw_input, raw_input_hash, content_type, source_agent_model, feedback, "
            "extraction_model, extraction_prompt_version, bullet_ids_produced, "
            "raw_input_embedding"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(activity.id), str(context_id), activity.timestamp.isoformat(),
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
             # v0.5 — DC worked-example retrieval
             json.dumps(activity.raw_input_embedding) if activity.raw_input_embedding else None),
        )
        await db.commit()
        return activity

    async def list_activities(
        self, context_id: uuid.UUID, limit: int = 50, offset: int = 0,
    ) -> list[Activity]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM activities WHERE context_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (str(context_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_activity(row) for row in rows]

    # ── Raw Input Queries (v0.4) ───────────────────────────────────────

    async def get_activities_with_raw_input(
        self,
        context_id: str,
        since: datetime | None = None,
        content_type: str | None = None,
    ) -> list[Activity]:
        """Get activity records that have raw_input, for re-extraction."""
        db = await self._get_db()
        query = (
            "SELECT * FROM activities WHERE context_id=? "
            "AND raw_input IS NOT NULL AND raw_input != ''"
        )
        params: list[Any] = [context_id]
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if content_type is not None:
            query += " AND content_type = ?"
            params.append(content_type)
        query += " ORDER BY timestamp ASC"
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_activity(row) for row in rows]

    async def get_raw_input_by_hash(
        self, context_id: str, raw_input_hash: str,
    ) -> Activity | None:
        """Check if this exact raw input has already been ingested (dedup)."""
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM activities WHERE context_id=? AND raw_input_hash=? LIMIT 1",
            (context_id, raw_input_hash),
        )
        row = await cursor.fetchone()
        return self._row_to_activity(row) if row else None

    async def get_bullets_by_ids(
        self, context_id: str, bullet_ids: list[str],
    ) -> list[Bullet]:
        """Get multiple bullets by their IDs."""
        if not bullet_ids:
            return []
        db = await self._get_db()
        placeholders = ",".join("?" * len(bullet_ids))
        cursor = await db.execute(
            f"SELECT * FROM bullets WHERE context_id=? AND id IN ({placeholders})",
            [context_id] + bullet_ids,
        )
        rows = await cursor.fetchall()
        return [self._row_to_bullet(row) for row in rows]

    async def find_similar_activities(
        self,
        context_id: str,
        embedding: list[float],
        limit: int = 3,
        threshold: float = 0.85,
        exclude_hash: str | None = None,
    ) -> list[tuple[Activity, float]]:
        """Brute-force cosine similarity against all activities with embeddings.
        Fine for SQLite — production-scale uses Postgres pgvector path."""
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM activities WHERE context_id=? "
            "AND raw_input_embedding IS NOT NULL AND raw_input != ''",
            (context_id,),
        )
        rows = await cursor.fetchall()
        scored: list[tuple[Activity, float]] = []
        for row in rows:
            activity = self._row_to_activity(row)
            if exclude_hash and activity.raw_input_hash == exclude_hash:
                continue
            if not activity.raw_input_embedding:
                continue
            sim = _cosine_similarity(embedding, activity.raw_input_embedding)
            if sim >= threshold:
                scored.append((activity, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ── Row Converters ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_bullet(row: aiosqlite.Row) -> Bullet:
        keys = row.keys()
        lifecycle_state_raw = row["lifecycle_state"] if "lifecycle_state" in keys else "active"
        archive_reason = row["archive_reason"] if "archive_reason" in keys else None
        return Bullet(
            id=row["id"], context_id=row["context_id"], section=row["section"],
            content=row["content"], bullet_type=row["bullet_type"],
            source_type=row["source_type"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            hit_count=row["hit_count"], miss_count=row["miss_count"],
            recall_count=row["recall_count"],
            last_recalled_at=datetime.fromisoformat(row["last_recalled_at"]) if row["last_recalled_at"] else None,
            salience=row["salience"], confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_session=row["source_session"], source_agent=row["source_agent"],
            parent_concept_id=row["parent_concept_id"], schema_id=row["schema_id"],
            is_active=bool(row["is_active"]), is_archived=bool(row["is_archived"]),
            archived_at=datetime.fromisoformat(row["archived_at"]) if row["archived_at"] else None,
            lifecycle_state=LifecycleState(lifecycle_state_raw or "active"),
            archive_reason=archive_reason,
        )

    @staticmethod
    def _row_to_schema(row: aiosqlite.Row) -> SchemaNode:
        raw_slots = json.loads(row["slots"])
        slots = {k: SlotDefinition(**v) for k, v in raw_slots.items()}
        return SchemaNode(
            id=row["id"], context_id=row["context_id"], name=row["name"],
            description=row["description"], slots=slots,
            typical_values=json.loads(row["typical_values"]),
            exceptions=json.loads(row["exceptions"]),
            instance_count=row["instance_count"], confidence=row["confidence"],
            bullet_ids=json.loads(row["bullet_ids"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_delta_batch(row: aiosqlite.Row) -> DeltaBatch:
        from engram.core.models import DeltaOperation
        ops_raw = json.loads(row["operations"])
        operations = [DeltaOperation(**op) for op in ops_raw]
        return DeltaBatch(
            id=row["id"], context_id=row["context_id"], operations=operations,
            trigger=row["trigger"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bullets_added=row["bullets_added"], bullets_updated=row["bullets_updated"],
            bullets_removed=row["bullets_removed"], bullets_merged=row["bullets_merged"],
        )

    @staticmethod
    def _row_to_concept(row: aiosqlite.Row) -> ConceptNode:
        return ConceptNode(
            id=uuid.UUID(row["id"]), type=row["type"], content=row["content"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            confidence=row["confidence"], salience=row["salience"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            version=row["version"],
            source_session=uuid.UUID(row["source_session"]) if row["source_session"] else None,
            source_agent=row["source_agent"],
            domain_tags=json.loads(row["domain_tags"]),
            metadata=json.loads(row["metadata"]),
            is_valid=bool(row["is_valid"]),
            invalidated_at=datetime.fromisoformat(row["invalidated_at"]) if row["invalidated_at"] else None,
            invalidation_reason=row["invalidation_reason"],
        )

    @staticmethod
    def _row_to_edge(row: aiosqlite.Row) -> ConceptEdge:
        return ConceptEdge(
            id=uuid.UUID(row["id"]), from_node=uuid.UUID(row["from_node"]),
            to_node=uuid.UUID(row["to_node"]), type=row["type"],
            weight=row["weight"], rationale=row["rationale"],
            created_at=datetime.fromisoformat(row["created_at"]),
            source_session=uuid.UUID(row["source_session"]) if row["source_session"] else None,
        )

    @staticmethod
    def _row_to_activity(row: aiosqlite.Row) -> Activity:
        keys = row.keys()
        # v0.4 fields — backward compatible via key check
        raw_input = row["raw_input"] if "raw_input" in keys else None
        raw_input_hash = row["raw_input_hash"] if "raw_input_hash" in keys else None
        content_type = row["content_type"] if "content_type" in keys else None
        source_agent_model = row["source_agent_model"] if "source_agent_model" in keys else None
        feedback_raw = row["feedback"] if "feedback" in keys else None
        feedback = json.loads(feedback_raw) if feedback_raw else None
        extraction_model = row["extraction_model"] if "extraction_model" in keys else None
        extraction_prompt_version = row["extraction_prompt_version"] if "extraction_prompt_version" in keys else None
        bullet_ids_raw = row["bullet_ids_produced"] if "bullet_ids_produced" in keys else "[]"
        bullet_ids_produced = json.loads(bullet_ids_raw or "[]")
        emb_raw = row["raw_input_embedding"] if "raw_input_embedding" in keys else None
        raw_input_embedding = json.loads(emb_raw) if emb_raw else None

        return Activity(
            id=uuid.UUID(row["id"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
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
