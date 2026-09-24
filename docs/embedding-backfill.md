# Repair missing embeddings

Use the corrected source checkout to repair existing active memories that have no embedding. This does not repeat extraction, change text, or create new memories. The current database schema uses 1536-dimensional embeddings; choose the same embedding model used for the existing vectors.

Preview one context in an existing SQLite database:

```bash
.venv/bin/python -m engram.maintenance.backfill_embeddings \
  --sqlite /absolute/path/to/engram.db \
  --context-id CONTEXT_UUID --limit 100
```

For PostgreSQL, put the existing database DSN in an environment variable and pass its **name**:

```bash
.venv/bin/python -m engram.maintenance.backfill_embeddings \
  --postgres-dsn-env ENGRAM_MAINTENANCE_DSN \
  --context-id CONTEXT_UUID --limit 100
```

Preview opens the existing database read-only. It does not initialize schemas, create the vector extension, change SQLite journal mode, load `.env`, or call an embedding provider. SQLite may create WAL coordination sidecars when opening an existing WAL database; its database content and journal mode remain unchanged. There is no default database target. A missing database or context fails instead of creating one.

To save embeddings, add the explicit apply and provider options to either command:

```bash
  --apply --embedding-model text-embedding-3-small \
  --embedding-key-env ENGRAM_EMBEDDING_API_KEY
```

Set the named provider-key environment variable before running apply. Apply makes provider requests and may incur usage charges. The limit is between 1 and 1000 and bounds attempted memories per invocation; it defaults to 100. The command is an operator tool using database access, not a new unauthenticated HTTP endpoint.

The JSON summary reports eligible, selected, attempted, updated, skipped, and failed counts. Exit status is 0 for a successful preview/apply, 1 for an operational or per-memory failure, and 2 for invalid command options. Errors identify a memory and a reason without printing provider messages, credentials, or memory text.

Each vector is computed outside a database transaction. Before saving, the command rechecks context, content, lifecycle, and whether a vector is still missing. Concurrent edits or an already-completed repair are skipped. Only the embedding column changes; IDs, text, timestamps, metadata, counters, and audit history remain intact. Archived and inactive memories are excluded. Both backends validate the float32 representation used by PostgreSQL, rejecting empty, nonfinite, zero, underflow-to-zero, overflowing, and wrong-dimension vectors. Squared norms must also remain finite and nonzero in float32 so repaired vectors work with cosine search; vector scale is preserved.

Repairs commit one memory at a time. If a later provider request fails, earlier successful repairs remain saved. Re-run the same command to repair remaining items; completed items are skipped. This tool cannot determine the model used for historical vectors, so model compatibility must be selected by the operator.

Validation is isolated: `pytest tests/test_embedding_backfill.py` uses temporary SQLite databases and fake providers. Setting `ENGRAM_TEST_POSTGRES_DSN` to a disposable test admin database also exercises PostgreSQL in newly created random databases. No repair was run against an existing developer or customer database during implementation.
