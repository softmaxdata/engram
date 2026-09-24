# Shared-core port and release checks

The OSS distribution (`engram-contextdb`, GitHub) and paid distribution (`engram`,
GitLab) share storage and retrieval contracts. Their complete core sources are not
identical: OSS v0.5 has reconsolidation deltas, rollback metadata, and an optional
validity gate; paid v0.4 has account/authentication, API-key, tenant, and hosted MCP
extensions. Port behavior deliberately instead of copying entire core directories.

`shared-core-manifest.json` records a small set of byte-identical helpers/tests and
named regression contracts. It also enumerates intentionally distinct modules.
Hashes enforce recorded ports within each repository's CI. The two-checkout check
additionally verifies that selected files and manifests actually agree. Contract
names ensure regression cases are retained; passing names/hashes does **not**
replace executing both editions' tests.

## Run locally

Use an isolated environment for each edition and Python version (3.11 and 3.13).
Install `.[dev,mcp]` for OSS and `.[dev]` for paid, plus `build`. CI uses those same
extras, so absent MCP dependencies cannot turn the package gate into a skip.

```bash
python -m pip install -e '.[dev,mcp]' build  # OSS
# python -m pip install -e '.[dev]' build   # paid
python scripts/check-shared-core.py --checkout .
python scripts/check-ci.py --backend sqlite --output reports/sqlite.xml
python -m evals.retrieval --output reports/retrieval.json
python scripts/check-package.py --output-dir reports/package
```

The suite wrapper runs from a temporary directory with a temporary pytest base,
ignoring checkout `.env` and inherited application/provider configuration. The
SQLite gate permits missing-DSN skips only for PostgreSQL cases. Both gates also
permit a narrow list of exact test-name/reason pairs for inapplicable scenarios
(independent `:memory:` databases, SQLite worker-queue tests on PostgreSQL, and
OSS-only extensions in paid). Each is printed separately. The package check
builds an sdist, builds a wheel from that sdist, installs the wheel into a fresh
venv outside the checkout, checks installed-module provenance, starts the actual
`engram` console script against temporary SQLite and calls `/health`, then runs
all six MCP stdio tests. Those tests include a real subprocess initialization,
tool discovery, and tool invocation against a loopback HTTP fixture. All six must
run with zero skips. No embedding/LLM provider is called. Retained wheel artifacts
are verification outputs; these workflows contain no publishing jobs.

PostgreSQL verification requires a **disposable** pgvector server with a test role
allowed to create/drop databases and roles. Every database integration fixture
creates its own random test database; never supply a customer/production DSN.

```bash
docker run --detach --rm --name engram-ci-pg \
  -e POSTGRES_DB=engram_test -e POSTGRES_USER=engram_test \
  -e POSTGRES_PASSWORD=engram-test-only -p 127.0.0.1:55432:5432 \
  pgvector/pgvector:pg16
ENGRAM_TEST_POSTGRES_DSN='postgresql://engram_test:engram-test-only@127.0.0.1:55432/engram_test' \
  python scripts/check-ci.py --backend postgres --output reports/postgres.xml
docker stop engram-ci-pg
```

The wrapper fails immediately when the DSN is missing, waits for connectivity,
checks pgvector availability and test privileges, and rejects any unexpected skip,
including a missing-DSN or missing-MCP skip in the PostgreSQL run. It also requires the vector-roundtrip/restart case to
execute. A broken service cannot produce a green pipeline by skipping coverage.

## Port discipline

Run the following from either checkout, with OSS first and paid second:

```bash
python scripts/check-shared-core.py /path/to/oss /path/to/paid
python scripts/check-shared-core.py --self-test /path/to/oss /path/to/paid
```

The self-test copies only selected files into disposable directories, introduces
real file drift and removes a required test name, and confirms both fail. It does
not edit either checkout. A single-repo CI run cannot prove the other repository
was updated; the two-checkout command is a required port/release check.

For each shared behavior change:

1. Record the source edition, source commit or patch identifier, counterpart
   commit/patch, affected contract, and any intentionally different implementation.
2. Port the change and its regression cases. Add new shared helpers/corpus files
   to `identical_files` only when both implementations are ready and intentionally
   identical. Use a `contracts` entry for shared test names when implementation or
   assertions must differ. Update the two manifests together.
3. Execute both editions on SQLite and PostgreSQL, both supported CI Python
   versions, retrieval evaluations, and installed-wheel/MCP checks. Record actual
   counts, versions, report paths, and failures; do not describe configured hosted
   jobs as executed pipelines.
4. Review the diff, then refresh recorded hashes only after the port is complete:

   ```bash
   python scripts/check-shared-core.py --refresh /path/to/oss /path/to/paid
   python scripts/check-shared-core.py --self-test /path/to/oss /path/to/paid
   ```

`--refresh` refuses missing contracts, different manifests, or unequal selected
files. It updates both manifests, never source files. Review hash changes as part
of the port; updating hashes is not evidence that tests passed.

## Release evidence checklist

Before approving a release, record the following in the release review:

- Source revisions/patches and port provenance for both editions; two-checkout
  shared-core result and deliberate-drift self-test result.
- Both Python versions' SQLite and PostgreSQL reports; pgvector/PostgreSQL version
  and the fact that only disposable databases were used; any failed checks.
- Both controlled retrieval reports and corpus/threshold revision. These are
  deterministic regression measurements, not real-provider quality claims.
- Both sdist-to-wheel artifacts, installed CLI/import checks, and six-case MCP
  smoke reports; actual interpreter/dependency versions used.
- Migration/backward-compatibility review (nullable historical receipt markers,
  transaction behavior, missing-embedding repair) and rollback/retry behavior.
- README/SDK/MCP examples and release snippets checked against the installed
  package; edition names/versions and intentional differences checked.
- Hosted pipeline URLs and results **when actually run**; record public release
  and deployment separately. Local equivalents do not establish
  hosted runner/service availability or branch-protection settings.

Workflow service configuration follows the official
[GitHub PostgreSQL service-container guide](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers)
and [GitLab PostgreSQL service guide](https://docs.gitlab.com/ci/services/postgres/).
GitHub runner jobs use the published loopback port; GitLab container jobs use the
`postgres` service alias. Both run the same readiness/suite wrapper and a Python
3.11/3.13 matrix. Jobs upload reports and package artifacts only.
