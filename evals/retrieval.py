"""Controlled retrieval regression benchmark using real SQLite and materialization.

The default uses offline embedding fixtures. Synthetic vectors control ranking
independently of provider changes; the text judgments also support live embeddings.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engram.core.materialization import MaterializationEngine
from engram.core.models import Bullet, ConceptNode, ConceptType, Context, IntentAnchor, SchemaNode
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend

CORPUS = Path(__file__).with_name("corpus.json")


class FixtureEmbeddings(LLMAdapter):
    """Exact text lookup: unknown text fails rather than silently inventing vectors."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    async def embed(self, text: str) -> list[float]:
        return self.vectors[text]

    async def complete(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("Retrieval evaluations never request completions")


def ranking_metrics(ranked: list[str], relevance: dict[str, int], k: int) -> dict:
    """Binary recall and graded nDCG; duplicate IDs earn no additional credit.

    Empty judgments return null, and are excluded from aggregate averages.
    Empty behavior is checked separately so an all-empty corpus cannot pass.
    """
    if k <= 0 or any(grade < 0 for grade in relevance.values()):
        raise ValueError("k must be positive and relevance grades nonnegative")
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        return {"recall_at_k": None, "ndcg_at_k": None}
    seen = set()
    dcg = 0.0
    for rank, key in enumerate(ranked[:k]):
        grade = relevance.get(key, 0) if key not in seen else 0
        dcg += (2**grade - 1) / math.log2(rank + 2)
        seen.add(key)
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {"recall_at_k": len(seen & relevant) / len(relevant), "ndcg_at_k": dcg / ideal}


def assess_result(
    case: dict,
    result: dict,
    receipt: Any,
    renderer: Any,
    id_map: dict[str, str],
    schema_map: dict[str, str],
) -> dict:
    """Assess the actual packed order, full rendered texts, and persisted receipt."""
    raw_ids = [str(key) for key in result["bullets_included"] + result["concepts_included"]]
    reverse_ids = {value: key for key, value in id_map.items()}
    ranked = [reverse_ids.get(key, key) for key in raw_ids]
    text = result["rendered_text"]
    rendered_ids = [item["id"] for item in case["items"] if item["text"] in text]
    forbidden = set(case.get("forbidden_ids", []))
    leakage = sorted(forbidden & (set(ranked) | set(rendered_ids)))
    metrics = ranking_metrics(ranked, case["relevance"], case["k"])
    errors = []
    for metric, threshold in case["thresholds"].items():
        if metrics[metric] is not None and metrics[metric] + 1e-12 < threshold:
            errors.append(metric)
    if leakage:
        errors.append("forbidden_id_leakage")
    if len(raw_ids) != len(set(raw_ids)) or set(ranked) != set(rendered_ids):
        errors.append("rendered_id_consistency")
    expected_schemas = {key for key, content in schema_map.items() if content in text}
    if set(result["schemas_included"]) != expected_schemas:
        errors.append("schema_consistency")
    if (
        receipt is None
        or receipt.bullets_included != result["bullets_included"]
        or receipt.token_count != result["token_count"]
    ):
        errors.append("persisted_receipt_consistency")
    actual_tokens = renderer.estimate_tokens(text)
    if result["token_count"] != actual_tokens or actual_tokens > case["token_budget"]:
        errors.append("token_budget_consistency")
    if case.get("expect_empty") and (ranked or rendered_ids):
        errors.append("expected_empty")
    return {
        "case": case["id"],
        "k": case["k"],
        **metrics,
        "ranked_ids": ranked,
        "rendered_ids": rendered_ids,
        "forbidden_id_leakage": leakage,
        "token_count": actual_tokens,
        "token_budget": case["token_budget"],
        "token_budget_consistent": "token_budget_consistency" not in errors,
        "receipt_consistent": not any(
            error in errors
            for error in (
                "rendered_id_consistency",
                "schema_consistency",
                "persisted_receipt_consistency",
            )
        ),
        "passed": not errors,
        "failures": errors,
    }


async def run_case(
    case: dict, target_model: str, storage: SQLiteBackend, provider: LLMAdapter | None = None
) -> dict:
    vectors = {case["query"]: case["query_vector"]}
    vectors.update({item["text"]: item["vector"] for item in case["items"]})
    if provider is not None:
        # Prefetch all provider vectors, including the query. Provider errors
        # must fail the benchmark, not trigger the engine's salience fallback.
        vectors = {text: await provider.embed(text) for text in vectors}
        dimensions = {len(vector) for vector in vectors.values()}
        if (
            len(dimensions) != 1
            or not next(iter(dimensions))
            or any(
                not all(math.isfinite(value) for value in vector) or not any(vector)
                for vector in vectors.values()
            )
        ):
            raise ValueError("Provider returned invalid or inconsistent embedding vectors")
    adapter = FixtureEmbeddings(vectors)
    context = Context(name="retrieval-eval", intent=IntentAnchor(objective="Evaluate retrieval"))
    foreign = Context(name="foreign-eval", intent=IntentAnchor(objective="Tenant isolation"))
    await storage.create_context(context)
    await storage.create_context(foreign)
    id_map = {}
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    for item in case["items"]:
        target = foreign.id if item.get("foreign") else context.id
        key = str(uuid.uuid4())
        id_map[item["id"]] = key
        if case.get("kind") == "legacy":
            await storage.add_concept(
                target,
                ConceptNode(
                    id=uuid.UUID(key),
                    type=ConceptType.FACT,
                    content=item["text"],
                    embedding=vectors[item["text"]],
                    salience=item["salience"],
                    created_at=timestamp,
                ),
            )
        else:
            await storage.add_bullet(
                str(target),
                Bullet(
                    id=key,
                    content=item["text"],
                    embedding=vectors[item["text"]],
                    salience=item["salience"],
                    created_at=timestamp,
                    is_active=item.get("is_active", True),
                    is_archived=item.get("is_archived", False),
                ),
            )
    schema_map = {}
    for item in case.get("schemas", []):
        schema = SchemaNode(name=item["name"], description=item["description"])
        await storage.add_schema(str(context.id), schema)
        schema_map[schema.id] = f"[Pattern: {schema.name}] {schema.description}"
    engine = MaterializationEngine(storage, adapter)
    options = {}
    if "include_worked_examples" in inspect.signature(engine.materialize).parameters:
        options["include_worked_examples"] = False
    result = await engine.materialize(
        context.id,
        query=case["query"],
        target_model=target_model,
        token_budget=case["token_budget"],
        include_intent=False,
        recency_weight=0,
        **options,
    )
    receipt = await storage.get_materialization(result["materialization_id"])
    assessment = assess_result(
        case, result, receipt, engine._get_renderer(target_model), id_map, schema_map
    )
    assessment["renderer"] = target_model
    return assessment


def summarize(results: list[dict], expected_count: int) -> dict:
    judged = [row for row in results if row["recall_at_k"] is not None]
    return {
        "passed": bool(judged)
        and len(results) == expected_count
        and all(row["passed"] for row in results),
        "scenario_renderer_count": len(results),
        "judged_count": len(judged),
        "mean_recall_at_k": sum(row["recall_at_k"] for row in judged) / len(judged)
        if judged
        else None,
        "mean_ndcg_at_k": sum(row["ndcg_at_k"] for row in judged) / len(judged) if judged else None,
        "forbidden_id_leakage_count": sum(len(row["forbidden_id_leakage"]) for row in results),
    }


async def evaluate(corpus: dict, provider: LLMAdapter | None = None) -> dict:
    # No configured paths, environment loading, or application startup: this
    # runner can only touch a brand-new scratch database, including in live mode.
    with tempfile.TemporaryDirectory(prefix="engram-retrieval-") as directory:
        storage = SQLiteBackend(str(Path(directory) / "evaluation.db"))
        await storage.initialize()
        try:
            live = provider is not None
            if provider is not None:
                texts = dict.fromkeys(
                    text
                    for case in corpus["cases"]
                    for text in [case["query"], *(item["text"] for item in case["items"])]
                )
                provider = FixtureEmbeddings({text: await provider.embed(text) for text in texts})
            results = [
                await run_case(case, renderer, storage, provider)
                for case in corpus["cases"]
                for renderer in corpus["renderers"]
            ]
        finally:
            await storage.close()
    summary = summarize(results, len(corpus["cases"]) * len(corpus["renderers"]))
    return {
        "benchmark": "live_provider_labeled_corpus" if live else "controlled_regression",
        "corpus_version": corpus["version"],
        "quality_claim": "Curated scratch data; offline vectors do not measure provider quality",
        "summary": summary,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write machine-readable JSON (also printed)")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--live-embeddings",
        action="store_true",
        help="Explicitly allow billable provider embeddings of curated texts only",
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--api-key-env", help="Environment variable holding the provider key")
    args = parser.parse_args(argv)
    if args.live_embeddings:
        if not args.embedding_model or not args.api_key_env or not os.environ.get(args.api_key_env):
            parser.error("live mode requires --embedding-model and a populated --api-key-env")
    elif args.embedding_model or args.api_key_env:
        parser.error("provider options require --live-embeddings")
    try:
        corpus = json.loads(args.corpus.read_text())
        if args.live_embeddings:
            from engram.llm.adapter import LiteLLMAdapter
            from engram.llm.explicit import explicit_embedding_runtime

            with explicit_embedding_runtime():
                provider = LiteLLMAdapter(
                    embedding_model=args.embedding_model,
                    embedding_api_key=os.environ[args.api_key_env],
                )
                report = asyncio.run(evaluate(corpus, provider))
            report["embedding_model"] = args.embedding_model
        else:
            report = asyncio.run(evaluate(corpus))
    except Exception as exc:
        # Provider exception messages can contain request credentials. Never
        # print them; the exception type is enough to diagnose a failed gate.
        report = {"summary": {"passed": False}, "error_type": type(exc).__name__}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
