# Compatibility and upgrade notes

The supported upgrade target is the existing REST, MCP and Python SDK API with
populated SQLite or PostgreSQL databases. The compatibility audit compares the
previous Git revision with the corrected source: all 34 existing HTTP operations,
40 schema definitions and 33 public SDK methods retain their existing fields,
enum values, documented responses and method signatures. All 353 existing public Python definitions remain present. Database upgrades add
columns and indexes without replacing account-independent context, memory,
activity or receipt IDs. No developer or customer database was upgraded during
validation.

Built-in SQLite and PostgreSQL support explicit transaction-scoped storage.
Out-of-tree storage implementations must implement `StorageBackend.transaction`
before adopting these atomic mutation paths; unsupported implementations fail
clearly. This is an extension-interface requirement. Existing renderers and
completion adapters without a per-call model argument remain supported. OSS v0.5
core memory, worked examples, validity checks and reconsolidation remain intact.

Incorrect or unsafe behavior is intentionally corrected: cross-context writes
are rejected, active capacity is checked against the actual committed state,
failed mutations cannot leave partial audit/activity records, and unavailable or
invalid similarity comparisons cannot justify destructive matching. Configured
reflector models and deduplication thresholds are now honored.

Historical audit rows that lack a generated target ID or required inverse state
cannot be reliably undone. Rollback refuses those batches before a partial write
rather than claiming success. An old content update without a saved embedding
clears the mismatched vector on rollback; it cannot recreate that historical
embedding. Old receipt rows cannot recover consumption history never recorded.
See [embedding-backfill.md](embedding-backfill.md) for the optional, explicit,
bounded repair of missing vectors using the correct historical embedding model.

The distribution is `engram-contextdb`; the Python import and CLI remain `engram`.
MCP 1.x is the supported dependency range. Installed-package checks build from an
sdist, install the resulting wheel outside the checkout, run the CLI, and exercise
the stdio protocol. These checks do not publish a corrected package or validate
native third-party sign-in. All fixes remain local until released.
