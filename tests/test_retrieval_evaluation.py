"""Real-storage retrieval gates and exact renderer packing regressions."""

import copy
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engram.core.materialization import MaterializationEngine
from engram.core.models import (
    ActionType,
    Activity,
    Bullet,
    ConceptNode,
    ConceptType,
    Context,
    IntentAnchor,
    SchemaNode,
)
from engram.storage.sqlite import SQLiteBackend
from evals.retrieval import (
    CORPUS,
    FixtureEmbeddings,
    assess_result,
    evaluate,
    ranking_metrics,
    summarize,
)


def test_metric_math_and_duplicate_credit():
    relevance = {"best": 3, "next": 1}
    assert ranking_metrics(["best", "next"], relevance, 2) == {"recall_at_k": 1, "ndcg_at_k": 1}
    actual = ranking_metrics(["next", "best"], relevance, 2)
    assert actual["ndcg_at_k"] == pytest.approx((1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3)))
    assert ranking_metrics(["best", "best"], relevance, 2)["recall_at_k"] == 0.5
    assert ranking_metrics([], relevance, 2) == {"recall_at_k": 0, "ndcg_at_k": 0}
    assert ranking_metrics([], {}, 2) == {"recall_at_k": None, "ndcg_at_k": None}
    with pytest.raises(ValueError):
        ranking_metrics([], relevance, 0)


def test_gate_cannot_pass_without_judgments_or_missing_results():
    assert not summarize([], 0)["passed"]
    empty = dict(passed=True, recall_at_k=None, ndcg_at_k=None, forbidden_id_leakage=[])
    assert not summarize([empty], 1)["passed"]
    judged = dict(empty, recall_at_k=1, ndcg_at_k=1)
    assert not summarize([judged], 2)["passed"]


@pytest.mark.asyncio
async def test_controlled_corpus_gate():
    report = await evaluate(json.loads(CORPUS.read_text()))
    assert report["benchmark"] == "controlled_regression"
    assert report["summary"]["scenario_renderer_count"] >= 24
    assert report["summary"]["judged_count"] > 0
    assert report["summary"]["passed"], [row for row in report["results"] if not row["passed"]]


@pytest.mark.asyncio
async def test_opt_in_provider_vectors_are_cached_and_validated():
    corpus = json.loads(CORPUS.read_text())
    corpus["cases"] = corpus["cases"][:1]
    case = corpus["cases"][0]
    vectors = {case["query"]: case["query_vector"]}
    vectors.update({item["text"]: item["vector"] for item in case["items"]})

    class RecordingProvider(FixtureEmbeddings):
        calls = []

        async def embed(self, text):
            self.calls.append(text)
            return await super().embed(text)

    provider = RecordingProvider(vectors)
    report = await evaluate(corpus, provider)
    assert report["benchmark"] == "live_provider_labeled_corpus"
    assert report["summary"]["passed"]
    assert sorted(provider.calls) == sorted(vectors)
    vectors[case["query"]] = [float("nan"), 0, 0]
    with pytest.raises(ValueError, match="invalid"):
        await evaluate(corpus, provider)


@pytest.mark.parametrize("degradation", ["ranking", "leakage", "empty"])
def test_cli_failure_gate(tmp_path, degradation):
    corpus = json.loads(CORPUS.read_text())
    case = copy.deepcopy(corpus["cases"][0])
    if degradation == "ranking":
        case["items"][0]["vector"], case["items"][1]["vector"] = (
            case["items"][1]["vector"],
            case["items"][0]["vector"],
        )
    elif degradation == "leakage":
        # Deliberately put a forbidden fact in the eligible context. This
        # simulates a storage isolation regression; the gate must catch it.
        case["forbidden_ids"] = [case["items"][0]["id"]]
    else:
        case = next(row for row in corpus["cases"] if row.get("expect_empty"))
    corpus["cases"] = [case]
    corpus["renderers"] = ["generic"]
    path = tmp_path / "degraded.json"
    path.write_text(json.dumps(corpus))
    output = tmp_path / "result.json"
    run = subprocess.run(
        [sys.executable, "-m", "evals.retrieval", "--corpus", str(path), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    report = json.loads(output.read_text())
    assert not report["summary"]["passed"]
    if degradation != "empty":
        failure = "recall_at_k" if degradation == "ranking" else "forbidden_id_leakage"
        assert failure in report["results"][0]["failures"]


def test_live_cli_redacts_real_adapter_errors(tmp_path):
    canary = "retrieval-fake-provider-key-canary"
    # Execute the real LiteLLMAdapter against a fake SDK in a separate process.
    # Import output, SDK output and the adapter's exception log must all stay
    # out of the machine-readable command output. No external call is possible.
    (tmp_path / "litellm.py").write_text(
        "import sys\n"
        "suppress_debug_info = False\n"
        "set_verbose = True\n"
        f"print({canary!r})\n"
        "async def aembedding(**kwargs):\n"
        "    print(kwargs['api_key'])\n"
        "    print(kwargs['api_key'], file=sys.stderr)\n"
        "    raise RuntimeError(kwargs['api_key'])\n"
    )
    output = tmp_path / "live-failure.json"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.retrieval",
            "--live-embeddings",
            "--embedding-model",
            "fake-model",
            "--api-key-env",
            "EVAL_TEST_API_KEY",
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path), "EVAL_TEST_API_KEY": canary},
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert run.returncode == 1
    assert canary not in run.stdout + run.stderr
    report = json.loads(run.stdout)
    assert report == json.loads(output.read_text())
    assert report["summary"]["passed"] is False
    assert report["error_type"] == "LLMAdapterError"


def test_offline_cli_does_not_import_provider_sdk(tmp_path):
    (tmp_path / "litellm.py").write_text("raise AssertionError('Provider SDK imported offline')\n")
    run = subprocess.run(
        [sys.executable, "-m", "evals.retrieval"],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["summary"]["passed"]


@pytest.mark.parametrize("target", ["claude", "gpt", "generic"])
@pytest.mark.parametrize("kind", ["bullets", "legacy", "schema"])
@pytest.mark.asyncio
async def test_receipts_match_complete_rendered_content(tmp_path, target, kind):
    storage = SQLiteBackend(str(tmp_path / "receipt.db"))
    await storage.initialize()
    try:
        context = Context(name="packing", intent=IntentAnchor(objective="Pack memories"))
        await storage.create_context(context)
        engine = MaterializationEngine(storage, FixtureEmbeddings({"query": [1, 0]}))
        texts = {}
        for index in range(3):
            content = f"Unique evidence {index}."
            if kind == "legacy":
                item = ConceptNode(
                    type=ConceptType.FACT,
                    content=content,
                    embedding=[1, 0],
                    salience=1 - index * 0.1,
                )
                await storage.add_concept(context.id, item)
            else:
                item = Bullet(content=content, embedding=[1, 0], salience=1 - index * 0.1)
                await storage.add_bullet(str(context.id), item)
            texts[str(item.id)] = content
        schemas = {}
        if kind == "schema":
            item = SchemaNode(name="Packing", description="Preserve all selected evidence.")
            await storage.add_schema(str(context.id), item)
            schemas[item.id] = f"[Pattern: {item.name}] {item.description}"
        options = (
            {"include_worked_examples": False}
            if "include_worked_examples" in (inspect.signature(engine.materialize).parameters)
            else {}
        )
        renderer = engine._get_renderer(target)
        # Sweep boundaries around section wrappers and multiple candidate sizes.
        for budget in [1, 10, 25, 30, 35, 40, 45, 55, 70, 100]:
            result = await engine.materialize(
                context.id,
                query="query",
                target_model=target,
                token_budget=budget,
                include_intent=False,
                **options,
            )
            included = [
                str(key) for key in result["bullets_included"] + result["concepts_included"]
            ]
            assert set(included) == {
                key for key, text in texts.items() if text in result["rendered_text"]
            }
            assert set(result["schemas_included"]) == {
                key for key, text in schemas.items() if text in result["rendered_text"]
            }
            receipt = await storage.get_materialization(result["materialization_id"])
            assert receipt.bullets_included == result["bullets_included"]
            assert receipt.token_count == result["token_count"] <= budget
            assert result["token_count"] == renderer.estimate_tokens(result["rendered_text"])
    finally:
        await storage.close()


@pytest.mark.parametrize("target", ["claude", "gpt", "generic"])
@pytest.mark.parametrize("kind", ["bullets", "legacy", "empty"])
@pytest.mark.asyncio
async def test_oss_core_memory_worked_examples_fit_tight_budget(tmp_path, target, kind):
    if "core_memory" not in Context.model_fields:
        pytest.skip("OSS v0.5 extensions are intentionally edition-specific")
    storage = SQLiteBackend(str(tmp_path / "extensions.db"))
    await storage.initialize()
    try:
        context = Context(
            name="extensions",
            core_memory="Persistent core summary. " * 20,
            intent=IntentAnchor(objective="Important objective. " * 20),
        )
        await storage.create_context(context)
        bullet = Bullet(content="Short actionable fact.", embedding=[1, 0])
        if kind == "bullets":
            await storage.add_bullet(str(context.id), bullet)
        elif kind == "legacy":
            await storage.add_concept(
                context.id,
                ConceptNode(type=ConceptType.FACT, content=bullet.content, embedding=[1, 0]),
            )
        activity = Activity(
            agent_id="eval",
            action_type=ActionType.MATERIALIZATION_OCCURRED,
            summary="Prior useful input",
            raw_input="Prior worked input. " * 40,
            raw_input_embedding=[1, 0],
            bullet_ids_produced=[bullet.id],
        )
        await storage.add_activity(context.id, activity)
        engine = MaterializationEngine(storage, FixtureEmbeddings({"query": [1, 0]}))
        # Real retrieval, including worked-example fetching from stored activity.
        for budget in [1, 30, 80, 200, 1000]:
            result = await engine.materialize(
                context.id,
                query="query",
                token_budget=budget,
                target_model=target,
                include_usage_stats=True,
            )
            assert result["token_count"] <= budget
            if bullet.id in result["bullets_included"]:
                assert bullet.content in result["rendered_text"]
        assert "Prior worked input." in result["rendered_text"]
        assert "Persistent core summary." in result["rendered_text"]
    finally:
        await storage.close()


@pytest.mark.parametrize("target", ["claude", "gpt", "generic"])
def test_complete_duplicate_candidates_and_oversized_extensions(target):
    renderer = MaterializationEngine._get_renderer(target)
    large = ConceptNode(type=ConceptType.DECISION, content="Large decision. " * 200)
    small = ConceptNode(type=ConceptType.FACT, content="Small complete fact.")
    duplicate = small.model_copy(update={"id": large.id})
    single_text = renderer.render([small], None, 1000)
    budget = renderer.estimate_tokens(single_text)
    options = {}
    if "core_memory" in inspect.signature(renderer.render).parameters:
        options = {"worked_examples": [{"input": "Oversized worked example. " * 200}]}
    text, selected = renderer.render_with_selection(
        [large, small, duplicate], None, budget, **options
    )
    assert [item.id for item in selected] == [small.id]
    assert small.content in text
    assert "Oversized worked example." not in text
    assert renderer.estimate_tokens(text) <= budget


@pytest.mark.parametrize("target", ["claude", "gpt", "generic"])
@pytest.mark.asyncio
async def test_duplicate_content_keeps_selected_bullets_usage_stats(tmp_path, target):
    if "include_usage_stats" not in inspect.signature(MaterializationEngine.materialize).parameters:
        pytest.skip("Usage annotations are an OSS v0.5 extension")
    storage = SQLiteBackend(str(tmp_path / "duplicate-usage.db"))
    await storage.initialize()
    try:
        context = Context(name="duplicate usage", intent=IntentAnchor(objective="Keep provenance"))
        await storage.create_context(context)
        first = Bullet(content="Identical fact.", salience=1, recall_count=2, hit_count=2)
        second = Bullet(content=first.content, salience=0.1, recall_count=99, hit_count=99)
        for bullet in (first, second):
            await storage.add_bullet(str(context.id), bullet)
        renderer = MaterializationEngine._get_renderer(target)
        concept = ConceptNode(type=ConceptType.FACT, content=first.content, confidence=0.5)
        # Existing callers can still supply annotations keyed by content.
        legacy_text = renderer.render(
            [concept], None, 1000, usage_stats={first.content: "(used 99×, success 99/99)"}
        )
        assert "(used 99×, success 99/99)" in legacy_text
        budget = 19 if target == "generic" else renderer.estimate_tokens(legacy_text)
        engine = MaterializationEngine(storage, FixtureEmbeddings({}))
        result = await engine.materialize(
            context.id,
            target_model=target,
            token_budget=budget,
            include_intent=False,
            include_worked_examples=False,
            include_usage_stats=True,
            recency_weight=0,
        )
        assert result["bullets_included"] == [first.id]
        assert "(used 2×, success 2/2)" in result["rendered_text"]
        assert "99" not in result["rendered_text"]
        assert result["token_count"] <= budget
        receipt = await storage.get_materialization(result["materialization_id"])
        assert receipt.bullets_included == [first.id]
    finally:
        await storage.close()


def test_inconsistent_receipt_is_detected():
    case = json.loads(CORPUS.read_text())["cases"][0]
    result = dict(
        bullets_included=["real-id"],
        concepts_included=[],
        schemas_included=[],
        rendered_text="",
        token_count=0,
    )
    renderer = MaterializationEngine._get_renderer("generic")
    assessment = assess_result(case, result, None, renderer, {"retry": "real-id"}, {})
    assert not assessment["passed"]
    assert "rendered_id_consistency" in assessment["failures"]
    assert "persisted_receipt_consistency" in assessment["failures"]
