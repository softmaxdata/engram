# Retrieval regression evaluation

Run from a source checkout after installing the project dependencies:

```sh
python -m evals.retrieval --output retrieval-results.json
pytest tests/test_retrieval_evaluation.py tests/test_materialization.py
```

The command prints JSON, optionally writes the same JSON file, and exits **1**
when any scenario fails. It creates and deletes a fresh temporary SQLite database.
It does not load `.env`, application configuration, or existing customer data.
Default embeddings are deterministic text-to-vector fixtures; no embedding or
completion provider is contacted. GPT uses its normal tiktoken estimator when
available (precache its tokenizer assets for a fully disconnected environment),
otherwise its documented character-count fallback.

This is a **controlled regression benchmark**, not measured real-provider
retrieval quality. The committed `evals/corpus.json` contains curated queries,
texts, graded relevance judgments, synthetic vectors, and per-case thresholds.
Both editions share the corpus, runner and test contract. The engine, scoring,
SQLite vector search, renderer and persisted materialization receipts are real.
The OSS edition retains its MMR ranking and v0.5 features; paid ranking is unchanged.

## Cases and interpretation

Each of ten scenarios runs against Claude, GPT and generic renderers:

| Case | What it checks |
| --- | --- |
| Semantic over salience | A relevant low-salience fact ranks ahead of an unrelated high-salience fact. |
| Irrelevant distractors | Two credential-management facts outrank unrelated facts. |
| Duplicate and diverse | Graded judgments cover a paraphrase and a distinct rollback fact. |
| Lifecycle exclusion | Archived and inactive facts never appear. |
| Tenant isolation | Another context's highly similar fact never appears. |
| Oversized candidate | A long leading candidate does not prevent packing a smaller useful fact. |
| Tight budget receipt | Complete rendered memories match receipt IDs within the budget. |
| Empty context | No IDs are invented; empty judgments do not inflate metric averages. |
| Legacy oversized | Legacy concepts follow the same packing guarantees. |
| Schema summary | Schema IDs match complete rendered summaries. |

The JSON contains per-scenario/per-renderer `recall_at_k`, `ndcg_at_k`, packed
`ranked_ids`, `rendered_ids`, forbidden-ID leakage, estimated token usage,
receipt/budget consistency flags, and named failures. Recall counts unique
relevant IDs in the first k selections divided by all positively judged IDs.
nDCG uses graded gain `2^grade - 1` and logarithmic rank discount. Repeated IDs
earn credit once. Order is the packer's selection order; renderers may group
the resulting text by concept type. Irrelevant facts may appear after k; this
corpus does not claim that retrieval filters every irrelevant candidate.

Recall must be 1.0 on every judged case. nDCG must be 1.0 except the duplicate/
diverse case, whose 0.9 floor permits the existing edition-specific ranking
tradeoff. Every case requires zero forbidden leakage, consistent persisted
receipts, and output within its own renderer's token estimator. Empty judgments
have null ranking metrics and are excluded from averages. A report with no
judged results or missing results cannot pass. Aggregate means are descriptive;
one failed scenario still fails the gate.

The tests deliberately degrade vectors, introduce a forbidden eligible fact,
and supply only empty judgments, then verify a nonzero CLI exit. They also check
metric arithmetic, duplicate credit, schema/legacy/bullet packing across small
budget boundaries, and OSS core memory, annotations and worked examples.
OSS-only tests skip on paid without changing its public API.

## Packing behavior

Selection accounts for the final renderer format, including headers, closing
tags, separators and annotations. Oversized candidates are skipped while
smaller successors are considered. Receipts record only selected complete
memories; duplicate text is tracked through IDs, not substring matching.
XML, Markdown and plain-text formats remain intact.

Core memory takes priority, then intent, worked examples and ranked memories.
Oversized core memory or intent text is shortened with a truncation marker;
examples, schemas and facts are included whole or omitted. A budget too small
for even an empty format returns an empty string. Tokens are estimates, not a
guarantee about an external model's tokenizer; empty text is zero tokens.

## Optional provider evaluation

Provider use requires explicit opt-in, a model and a named credential variable:

```sh
python -m evals.retrieval --live-embeddings \
  --embedding-model text-embedding-3-small --api-key-env OPENAI_API_KEY \
  --output live-retrieval-results.json
```

This makes billable embedding calls for distinct curated corpus texts and queries
once each, caches them for the run, and still uses only a new scratch database.
It never requests completions. Provider failures or invalid vectors fail the
command rather than silently measuring the engine's salience fallback. Results
are labeled `live_provider_labeled_corpus` and identify the requested embedding
model. The same small corpus and thresholds apply; a provider result is not a
general quality estimate or evidence about customer workloads. Live mode is
never part of the default test or CI command.

The standalone live command prevents implicit SDK `.env` loading and suppresses
raw provider diagnostic output. Failures report only their exception type in
JSON, so credentials included in a provider exception cannot leak through logs.
