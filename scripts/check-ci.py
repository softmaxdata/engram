#!/usr/bin/env python3
"""Run the full regression suite with an explicit SQLite or pgvector CI gate."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


def inapplicable_cases(project: str) -> set[tuple[str, str, str]]:
    """Exact exceptions for tests whose scenario cannot exist on that backend."""
    cases = {
        ("test_atomic_mutations",
         "test_independent_engines_preserve_distinct_receipt_increments[:memory:]",
         "independent in-memory instances intentionally have separate databases"),
        ("test_atomic_mutations", "test_two_processes_consume_one_receipt_once[:memory:]",
         "process race requires persistent shared storage"),
    }
    for phase in ("begin", "ordinary_write", "savepoint"):
        cases.add(("test_storage_transactions",
                   "test_sqlite_cancellation_during_queued_sql_does_not_leak_transaction"
                   f"[postgres-{phase}]", "SQLite worker queue regression"))
    if project == "engram":
        for renderer in ("claude", "gpt", "generic"):
            cases.add(("test_retrieval_evaluation",
                       f"test_duplicate_content_keeps_selected_bullets_usage_stats[{renderer}]",
                       "Usage annotations are an OSS v0.5 extension"))
        for kind in ("bullets", "legacy", "empty"):
            for renderer in ("claude", "gpt", "generic"):
                cases.add(("test_retrieval_evaluation",
                           "test_oss_core_memory_worked_examples_fit_tight_budget"
                           f"[{kind}-{renderer}]",
                           "OSS v0.5 extensions are intentionally edition-specific"))
    return cases


async def wait_for_postgres(dsn: str) -> None:
    import asyncpg

    connection = None
    for attempt in range(45):
        try:
            connection = await asyncpg.connect(dsn, timeout=2)
            break
        except (TimeoutError, OSError, asyncpg.PostgresError):
            if attempt == 44:
                raise RuntimeError(
                    "Disposable PostgreSQL did not become ready; DSN redacted"
                ) from None
            await asyncio.sleep(1)
    assert connection is not None
    try:
        available = await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
        )
        capable = await connection.fetchval(
            "SELECT rolsuper OR (rolcreatedb AND rolcreaterole) "
            "FROM pg_roles WHERE rolname = current_user"
        )
        if not available:
            raise RuntimeError("PostgreSQL service must include the pgvector extension")
        if not capable:
            raise RuntimeError("Disposable test DSN must permit CREATE DATABASE and CREATE ROLE")
    finally:
        await connection.close()
    print("Disposable PostgreSQL is ready; pgvector and test privileges verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="JUnit XML output path")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    report = args.output.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN", "")
    if args.backend == "postgres":
        if not dsn:
            parser.error("--backend postgres requires ENGRAM_TEST_POSTGRES_DSN; never skip CI")
        asyncio.run(wait_for_postgres(dsn))
    # Tests run away from .env; discard application/provider configuration too.
    keep = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "LC_ALL")
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update({
        "PYTHONPATH": str(checkout), "PYTHONNOUSERSITE": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DO_NOT_TRACK": "1",
        "OTEL_SDK_DISABLED": "true",
    })
    if args.backend == "postgres":
        env["ENGRAM_TEST_POSTGRES_DSN"] = dsn
    with tempfile.TemporaryDirectory(prefix="engram-ci-tests-") as temporary:
        work = Path(temporary).resolve()
        if work.is_relative_to(checkout):
            raise RuntimeError("Use a TMPDIR outside the source checkout")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-q", "-c", str(checkout / "pyproject.toml"),
            "-o", "cache_dir=" + str(work / "pytest-cache"),
            "--basetemp", str(work / "pytest-temp"), "--junitxml", str(report),
            str(checkout / "tests"),
        ], cwd=work, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
    cases = ET.parse(report).findall(".//testcase")
    if not cases:
        raise SystemExit("CI must execute tests")
    project = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]["name"]
    allowed = inapplicable_cases(project)
    unexpected_skips = []
    postgres_unconfigured = 0
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None:
            continue
        reason = skipped.get("message", "")
        name = case.get("name", "unknown")
        key = (case.get("classname", "").rsplit(".", 1)[-1], name, reason)
        if key in allowed:
            print(f"Explicitly inapplicable: {name}: {reason}", flush=True)
        elif (args.backend == "sqlite" and reason in {
            "requires a disposable PostgreSQL admin DSN",
            "requires disposable PostgreSQL admin DSN",
        } and ("[postgres" in name or name == "test_fresh_postgres_vector_roundtrip_and_restart")):
            postgres_unconfigured += 1
        else:
            unexpected_skips.append(name)
    if unexpected_skips:
        raise SystemExit("Unexpected skipped tests: " + ", ".join(unexpected_skips))
    if args.backend == "postgres" and not any(
        case.get("name") == "test_fresh_postgres_vector_roundtrip_and_restart"
        and case.find("skipped") is None for case in cases
    ):
        raise SystemExit("Required PostgreSQL vector/restart integration test did not execute")
    if postgres_unconfigured:
        print(f"SQLite-only run: {postgres_unconfigured} PostgreSQL cases require the separate "
              "PostgreSQL gate", flush=True)
    print(f"{args.backend} gate passed: {len(cases)} cases; no unexpected skips", flush=True)


if __name__ == "__main__":
    main()
