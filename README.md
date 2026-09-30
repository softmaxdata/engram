# Engram

**A brain-inspired, portable context database for AI agents.**

Engram stores agent context as atomic knowledge **bullets** in a concept graph — not raw text. Any AI agent can connect to it regardless of which LLM or framework it uses. Context persists across sessions, transfers between models, and gets smarter with every use through reinforcement learning on context quality.

Inspired by how the human brain stores and retrieves memory — associative recall, schema formation, reconsolidation, active forgetting, and consolidation — and by three lines of recent agent-memory research: [Agentic Context Engineering (ACE)](https://arxiv.org/abs/2510.04618), [Dynamic Cheatsheet (DC)](https://arxiv.org/abs/2504.07952), and [Mem-α](https://arxiv.org/abs/2509.25911). Every other agent memory product is building a filing cabinet. Engram is building something that learns.

## Why bother and why Engram?

Current AI agent frameworks store context as raw text, summaries, or vector chunks. This leads to:

- **Context decay** — details lost through repeated summarization
- **Context isolation** — Claude can't share context with GPT or Gemini
- **Context-as-text** — no structure, no relationships, no intent tracking
- **No learning** — context doesn't improve based on what actually worked

Engram solves these with:

- **Agnostic and portable** — We support all LLM: Claude, ChatGPT, Gemini, DeepSeek; all Agentic Frameworks: Langgraph, CrewAI, AG2; and all cloud platforms: AWS, GCP, Azure...
- **Atomic bullets** — discrete, individually-addressable knowledge units with usage tracking
- **Delta operations** — every mutation (including reconsolidation, core-memory edits) is an atomic delta op, never a wholesale rewrite, preventing context collapse and giving full audit + rollback
- **Cross-platform** — store once, materialize for any LLM (Claude, GPT, Gemini, local models)
- **Reinforcement loop** — bullets that prove useful get stronger; unhelpful ones fade away
- **Core memory slot** — Mem-α-inspired always-in-context running summary (≤512 tokens) per context, rewritten by the Reflector and prepended at every recall so the agent never loses the high-level frame
- **Worked-example retrieval** — Dynamic-Cheatsheet-inspired: when a recall query is highly similar to a prior raw input, attach that prior input + the bullets it produced as a worked example next to the question
- **MMR diversity at recall** — Maximal Marginal Relevance ranking stops the token budget from filling with near-duplicate bullets
- **Optional validity gate** — Mem-α-inspired LM-judge filter at write time, dropping malformed / trivial candidates before they hit the graph
- **Multi-agent safe** — per-context advisory locks serialize delta application across concurrent agents
- **Bounded storage** — three-tier lifecycle (active → archived → purged) with capacity management
- **Canonical extraction** — server-level Reflector model ensures consistent bullets regardless of which agent committed
- **Future-proofing with raw input preservation** — every commit stores the original text (like git commits), enabling re-extraction with better models

## How It Works

Engram is a **server** that sits between your AI agents and a knowledge graph. Your agents talk to Engram over HTTP — they send raw text in and get structured context back out. The server does all the heavy lifting.

### The Core Loop

There are only two operations that matter: **commit** (write) and **materialize** (read).

![The Core Loop](https://raw.githubusercontent.com/softmaxdata/engram/main/diagrams/core-loop.png)

### What's a Bullet?

Instead of storing your agent's output as a blob of text, Engram breaks it into **bullets** — atomic, individually-trackable knowledge units. Each bullet has a type (`fact`, `decision`, `strategy`, `warning`, `procedure`, `exception`, `principle`, `episodic`), a salience score, and usage stats that track how often it's been recalled and whether it helped. `episodic` bullets (Mem-α inspired) are timestamped events — formatted `At {timestamp}, {actor} {did X}` — and are merged with a looser similarity threshold because multiple agents often paraphrase the same event.

![What's a Bullet?](https://raw.githubusercontent.com/softmaxdata/engram/main/diagrams/bullet-anatomy.png)

### What's Reconsolidation?

This is Engram's learning mechanism, borrowed from neuroscience. When you recall memories and then use them, Engram tracks the outcome:

![Reconsolidation Loop](https://raw.githubusercontent.com/softmaxdata/engram/main/diagrams/reconsolidation.png)

### What's the Reflector?

Your agents send raw text. They don't extract bullets themselves. The Engram **server** runs a canonical LLM (the "Reflector," configured in `.env`) that processes ALL raw input from ALL agents.

![Canonical Reflector](https://raw.githubusercontent.com/softmaxdata/engram/main/diagrams/reflector.png)

### What's a Context?

A **context** is a container for related knowledge — like a project or workspace. Each context has:

- An **intent anchor** (immutable objective that prevents drift)
- A **core memory** (Mem-α-inspired always-in-context summary, ≤512 tokens, edited by the Reflector and prepended at every recall)
- A **concept graph** (bullets + edges + schemas)
- **Capacity limits** (bounded storage with automatic lifecycle management)
- Its own **activity ledger** (permanent record of every raw input, with embeddings used to retrieve nearest-prior-input worked examples at recall time)

You can have as many contexts as you need. Agents specify which context to read from and write to.

### What gets returned at recall time?

A materialization call assembles, in this order:

1. **Core memory** — the always-in-context summary
2. **Intent anchor** — objective, success criteria, constraints
3. **Schemas** — abstract patterns (token-efficient compressions of repeated bullets)
4. **Bullets** — ranked by relevance × effective_salience × recency, then re-ordered with **MMR** (Maximal Marginal Relevance) so duplicates don't crowd the token budget
5. **Worked examples** — if the query is highly similar to a prior raw input, the prior input + bullets it produced are attached *last* (closest to the question), with a "verify before copying" caveat — borrowed from Dynamic Cheatsheet

The whole pack is rendered for the target model (Claude XML, GPT Markdown, etc.) and a `materialization_id` receipt is returned so the next commit can close the reconsolidation loop.

## Quick Start

### 0. Setting up storage

There are two storage options:
- SQLite (local) — zero setup, tables auto-created on first run
- PostgreSQL (local/remote) — requires creating the database first

For PostgreSQL, create the database and enable pgvector before starting the server:
```sql
CREATE DATABASE engram;
\c engram
CREATE EXTENSION IF NOT EXISTS vector;
```

Then set the environment variables:
```bash
ENGRAM_STORAGE_BACKEND=postgres
ENGRAM_POSTGRES_DSN=postgresql://user:password@host:5432/engram
```

All tables are auto-created on first server start for both backends.

### 1. Configure the Server

Engram needs LLM API keys to extract knowledge from raw text. The server handles all LLM calls — agents never need to provide keys.

Create a `.env` file in the project root:

```bash
# .env

# ─── Database ───────────────────────────────────────────────
# SQLite for local development (zero setup)
ENGRAM_STORAGE_BACKEND=sqlite
ENGRAM_SQLITE_PATH=./engram.db

# PostgreSQL for production (uncomment to use instead of SQLite)
# ENGRAM_STORAGE_BACKEND=postgres
# ENGRAM_POSTGRES_DSN=postgresql://engram:password@localhost:5432/engram

# ─── Server ─────────────────────────────────────────────────
ENGRAM_HOST=0.0.0.0
ENGRAM_PORT=5820

# ─── LLM API Keys (required) ───────────────────────────────
# These are used SERVER-SIDE by the Reflector and Curator.
# Agents never see or need these keys.
ANTHROPIC_API_KEY=sk-ant-...           # For Reflector (concept extraction)
OPENAI_API_KEY=sk-...                  # For embeddings (similarity search)

# ─── Canonical Reflector ────────────────────────────────────
# The Reflector model processes ALL raw input from ALL agents.
# Using one model guarantees consistent bullet quality regardless
# of which agent (Claude, GPT, Gemini) committed the data.
ENGRAM_REFLECTOR_MODEL=claude-haiku-4-5
ENGRAM_REFLECTOR_PROMPT_VERSION=v1
ENGRAM_MAX_REFLECTION_ROUNDS=2         # Self-refinement iterations

# ─── Curator ────────────────────────────────────────────────
ENGRAM_CURATOR_DEDUP_THRESHOLD=0.92    # Cosine similarity threshold for merging
ENGRAM_CURATOR_SLOW_PATH_MODEL=claude-haiku-4-5  # For complex conflicts

# ─── Embedding ──────────────────────────────────────────────
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small

# ─── Consolidation ──────────────────────────────────────────
ENGRAM_CONSOLIDATION_TRIGGER=every_10_commits  # or "daily", "manual"
ENGRAM_FAST_DECAY_RATE=0.97            # Per-day salience decay for normal bullets
ENGRAM_SLOW_DECAY_RATE=0.995           # Per-day decay for decisions/schemas (slower)

# ─── Auth (disabled for local dev, enable for production) ───
ENGRAM_AUTH_ENABLED=false
# COGNITO_USER_POOL_ID=us-west-2_xxx
# COGNITO_APP_CLIENT_ID=xxx
```

### 2. Start the Server

**Local (SQLite):**

```bash
pip install engram-contextdb
engram
# Server running at http://localhost:5820
# Using .env for configuration
```

**Docker (PostgreSQL + pgvector):**

```bash
docker compose up
# Starts: Engram API + PostgreSQL + pgvector
# Reads .env for all configuration
```

### 3. Use the Python SDK

```python
import asyncio
from engram.sdk.client import Engram

async def main():
    # ──────────────────────────────────────────────────────────
    # Connect to the Engram server.
    # The SDK only needs the server URL — no LLM keys needed.
    # All concept extraction happens server-side using the
    # canonical Reflector model configured in .env.
    # ──────────────────────────────────────────────────────────
    async with Engram(url="http://localhost:5820") as engram:

        # ──────────────────────────────────────────────────────
        # CREATE CONTEXT
        #
        # A context is a container for related knowledge.
        # Think of it as a "project" or "workspace."
        #
        # The intent anchor defines the objective and prevents
        # drift over long-running sessions. It's immutable once
        # created and always included in materialized context.
        # ──────────────────────────────────────────────────────
        ctx = await engram.create_context(
            name="My Project",                    # Human-readable name
            intent={
                "objective": "Build a PDF form field extractor",
                "success_criteria": [             # How do we know we're done?
                    "95% accuracy",
                    "Under 30s per doc",
                ],
                "constraints": [                  # Boundaries and limitations
                    "Must use AWS infrastructure",
                ],
            },
        )
        # ctx.id is the context_id — a permanent identifier for this
        # knowledge container. Use it in all subsequent calls.

        # ──────────────────────────────────────────────────────
        # COMMIT (Write Knowledge)
        #
        # Send raw text to Engram. The server processes it through:
        #   1. Canonical Reflector (LLM) — extracts structured insights
        #   2. Curator — decides what to store as bullets
        #   3. Delta engine — applies atomic changes to the graph
        #   4. Activity ledger — stores the raw text permanently
        #
        # The agent provides raw text. The server does all the work.
        # ──────────────────────────────────────────────────────
        await engram.commit(
            ctx.id,                               # Which context to write to
            agent_id="claude-agent",              # Identifies WHO is writing (for provenance
                                                  # and multi-agent coordination)
            content="""
                PaddleOCR gives the best results on complex multi-column layouts.
                It's 40% faster than Textract with comparable accuracy.
                We decided to go with PaddleOCR for field detection.
            """,                                  # Raw text — the "source code"
                                                  # This is stored permanently in the
                                                  # activity ledger AND processed by the
                                                  # Reflector into bullets
            content_type="conversation",          # Tells the Reflector what kind of input:
                                                  #   "conversation" — chat messages
                                                  #   "tool_output"  — code/API results
                                                  #   "document"     — uploaded files
            source_model="claude-sonnet-4",       # Which LLM generated this text (metadata
                                                  # only — does NOT affect extraction)
        )
        # After this call, the concept graph now contains bullets like:
        #   [FACT]     "PaddleOCR is 40% faster than Textract with comparable accuracy"
        #   [DECISION] "Using PaddleOCR for field detection"
        #   [STRATEGY] "PaddleOCR works best for complex multi-column layouts"
        #
        # The raw text is also preserved in the activity ledger with a SHA-256
        # hash for dedup. If you commit the same text again, it's skipped.

        # ──────────────────────────────────────────────────────
        # MATERIALIZE (Read Knowledge)
        #
        # Retrieve relevant context from the concept graph.
        # This is NOT a simple database query — it does:
        #   1. Embeds your query
        #   2. Finds seed bullets by semantic similarity
        #   3. Traverses edges (spreading activation)
        #   4. Ranks by effective_salience (usage-weighted)
        #   5. Packs into token budget
        #   6. Renders for target model (Claude=XML, GPT=Markdown)
        #   7. Records which bullets were recalled (for reconsolidation)
        #
        # Returns a materialization_id — a receipt that tracks
        # exactly which bullets were included in this recall.
        # ──────────────────────────────────────────────────────
        result = await engram.materialize(
            ctx.id,                               # Which context to read from
            query="What OCR approach was decided?",  # Semantic search query — find
                                                  # bullets relevant to this question
            token_budget=2000,                    # Max tokens in returned context.
                                                  # The engine packs the highest-value
                                                  # bullets that fit within this budget.
            target_model="gpt-4o",                # Render for this model:
                                                  #   "claude" → XML structure
                                                  #   "gpt-4o" → Markdown with headers
                                                  #   "gemini" → Natural prose
                                                  #   "generic" → Plain text
        )
        print(result["rendered_text"])            # The materialized context, formatted
                                                  # for the target model
        print(result["materialization_id"])        # Receipt ID — pass this back in the
                                                  # next commit for reconsolidation

        # ──────────────────────────────────────────────────────
        # COMMIT WITH RECONSOLIDATION
        #
        # When you commit results AND include the materialization_id
        # from a previous recall, the reconsolidation engine fires:
        #
        #   - Looks up which bullets were in that materialization
        #   - If outcome=success: boosts hit_count + salience
        #   - If outcome=failure: boosts miss_count + decays salience
        #
        # This creates a reinforcement loop: bullets that consistently
        # prove useful get stronger. Unhelpful ones fade away.
        # The concept graph gets smarter with every use.
        # ──────────────────────────────────────────────────────
        await engram.commit(
            ctx.id,                               # Same context
            agent_id="gpt-evaluator",             # A different agent writing this time —
                                                  # demonstrating cross-model portability
            content="Re-evaluated PaddleOCR — confirmed excellent on forms.",
            materialization_id=result["materialization_id"],
                                                  # Links this commit to the previous
                                                  # recall. Tells Engram: "those bullets
                                                  # you gave me? Here's how they performed."
                                                  # Without this, the commit still saves
                                                  # new knowledge, but the reinforcement
                                                  # loop is broken.
            feedback={                            # Structured execution feedback.
                "outcome": "success",             # "success" | "failure" | "partial"
                                                  # Drives reconsolidation:
                                                  #   success → recalled bullets get +hit
                                                  #   failure → recalled bullets get +miss
            },
        )

        # ──────────────────────────────────────────────────────
        # LIFECYCLE STATUS
        #
        # Check capacity and health of a context.
        # ──────────────────────────────────────────────────────
        lifecycle = await engram.get_lifecycle(ctx.id)
        print(f"Active bullets:  {lifecycle['capacity']['active_bullets']}")
        print(f"Max bullets:     {lifecycle['capacity']['max_bullets']}")
        print(f"Capacity:        {lifecycle['capacity']['capacity_percent']}%")
        print(f"Pressure level:  {lifecycle['capacity']['pressure_level']}")
        # Pressure levels: normal (<80%), high (80-95%), critical (95-100%), full (100%)

        # ──────────────────────────────────────────────────────
        # SERVER CONFIGURATION
        #
        # Check which Reflector model the server uses.
        # This is configured in .env, NOT per-agent.
        # ──────────────────────────────────────────────────────
        config = await engram.get_ingestion_config()
        print(f"Reflector model:   {config['reflector_model']}")
        print(f"Prompt version:    {config['reflector_prompt_version']}")
        print(f"Dedup threshold:   {config['curator_dedup_threshold']}")

        # ──────────────────────────────────────────────────────
        # RE-EXTRACTION (dry run)
        #
        # Recompile bullets from stored raw text using a better
        # model. The raw text is the "source code" — bullets are
        # the "compiled output." This is like upgrading your
        # compiler and recompiling.
        # ──────────────────────────────────────────────────────
        preview = await engram.re_extract(
            ctx.id,
            reflector_model="claude-sonnet-4-20250514",  # New model to try
            dry_run=True,                         # Preview only — don't apply changes
        )
        print(f"Would process:   {preview['activities_to_process']} activities")
        print(f"Bullets to add:  {preview['estimated_bullets_added']}")
        print(f"Bullets to drop: {preview['estimated_bullets_removed']}")

asyncio.run(main())
```

## LangGraph Integration Example

A complete example showing Engram as the memory backend for a LangGraph agent. The agent recalls context before responding and commits results after — creating a persistent memory that survives across sessions and improves over time.

```python
import asyncio
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from engram.sdk.client import Engram


# ──────────────────────────────────────────────────────────
# SETUP
#
# The LLM is what the agent uses to respond to the user.
# The Engram client connects to the Engram server for memory.
# These are separate — the agent's LLM does NOT process memory.
# The server's canonical Reflector (configured in .env) does that.
# ──────────────────────────────────────────────────────────

CONTEXT_ID = None  # Will be created on first run, reused after

llm = ChatAnthropic(model="claude-sonnet-4-20250514")  # Agent's LLM
engram = Engram(url="http://localhost:5820")            # Memory server


# ──────────────────────────────────────────────────────────
# STATE
#
# LangGraph passes this state dict between nodes. Each node
# receives the full state and returns a partial dict of updates.
#
# Annotated[list, operator.add] means returned messages are
# APPENDED to the list, not replaced. So after setup → recall →
# respond, the messages list contains:
#   [HumanMessage, SystemMessage(memories), AIMessage(response)]
# ──────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, operator.add]   # Chat messages — appended by each node
    context_id: str                           # Which Engram context to use
    materialization_id: str | None            # Receipt from last recall (for reconsolidation)


# ──────────────────────────────────────────────────────────
# NODE 1: SETUP
#
# Create an Engram context on first run, reuse it after.
# The context persists on the server — even if you restart
# this script, the knowledge is still there.
# ──────────────────────────────────────────────────────────

async def setup_context(state: State) -> dict:
    global CONTEXT_ID
    if CONTEXT_ID is None:
        ctx = await engram.create_context(
            name="Research Assistant",
            intent={
                "objective": "Help user with research tasks",
                "success_criteria": ["Remember key findings across sessions"],
            },
        )
        CONTEXT_ID = str(ctx.id)
    return {"context_id": CONTEXT_ID}


# ──────────────────────────────────────────────────────────
# NODE 2: RECALL
#
# Before the LLM responds, pull relevant memories from Engram.
# The user's message is used as the search query.
#
# What happens inside engram.materialize():
#   1. Embed the query
#   2. Find semantically similar bullets (seed selection)
#   3. Traverse graph edges (spreading activation)
#   4. Rank by effective_salience = salience × (0.5 + hit_rate)
#   5. Pack top bullets into token_budget
#   6. Render as text for Claude (XML format)
#   7. Record which bullets were recalled → materialization_id
#
# The materialization_id is saved in state so the commit node
# can link the outcome back for reconsolidation.
# ──────────────────────────────────────────────────────────

async def recall(state: State) -> dict:
    last_msg = state["messages"][-1].content     # The user's message

    mat = await engram.materialize(
        context_id=state["context_id"],           # Which knowledge graph to search
        query=last_msg,                           # Semantic search using this text
        token_budget=2000,                        # Max tokens to return
    )

    # Inject memories as a system message. Because messages uses
    # operator.add, this gets APPENDED — the user's message stays.
    # Claude will see: [HumanMessage, SystemMessage(memories)]
    system = SystemMessage(content=f"Your memory:\n{mat['rendered_text']}")

    return {
        "messages": [system],                     # Appended to messages list
        "materialization_id": mat["materialization_id"],
            # Receipt: "these specific bullets were recalled."
            # Passed to commit for reconsolidation feedback.
    }


# ──────────────────────────────────────────────────────────
# NODE 3: RESPOND
#
# The LLM generates a response. At this point, state["messages"]
# contains:
#   [HumanMessage("user's question"), SystemMessage("Your memory:\n...")]
#
# Claude sees the user's question AND the relevant memories,
# so it can reference prior knowledge in its answer.
# ──────────────────────────────────────────────────────────

async def respond(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])  # Claude generates response
    return {"messages": [response]}                  # AIMessage appended to list


# ──────────────────────────────────────────────────────────
# NODE 4: COMMIT
#
# Save the conversation exchange back to Engram.
#
# What happens server-side when commit() is called:
#   1. Raw text stored in activity ledger (permanent, immutable)
#   2. SHA-256 hash computed for dedup (skip if identical)
#   3. Canonical Reflector extracts insights into bullets
#   4. Curator decides delta operations (add/update/merge)
#   5. Deltas applied under per-context advisory lock
#   6. If materialization_id provided → reconsolidation fires:
#      - Looks up which bullets were in that recall
#      - Outcome="success" → hit_count++, salience boosted
#      - Outcome="failure" → miss_count++, salience decayed
#   7. Events emitted to any subscribed agents
#
# This example commits just the last human + AI exchange,
# because it commits after EVERY turn. If you committed the
# full message history every time, earlier messages would be
# reprocessed (the hash dedup catches exact duplicates, but
# slightly different formatting would bypass it).
# ──────────────────────────────────────────────────────────

async def commit(state: State) -> dict:
    msgs = state["messages"]
    human = [m for m in msgs if isinstance(m, HumanMessage)][-1]
    ai = [m for m in msgs if isinstance(m, AIMessage)][-1]

    await engram.commit(
        context_id=state["context_id"],           # Which knowledge graph to write to
        agent_id="langgraph-research",            # Identifies this agent for provenance.
                                                  # Shows up in the activity ledger and
                                                  # agent dashboard. Also used for event
                                                  # notifications in multi-agent setups.
        content=f"User: {human.content}\nAssistant: {ai.content}",
                                                  # Raw text — stored permanently AND
                                                  # processed by the Reflector into bullets.
                                                  # This is the "source code" that can be
                                                  # re-extracted with a better model later.
        content_type="conversation",              # Tells the Reflector what extraction
                                                  # strategy to use:
                                                  #   "conversation" → extract insights,
                                                  #     decisions, strategies from dialogue
                                                  #   "tool_output" → extract success/fail
                                                  #     signals, metrics, error patterns
                                                  #   "document" → extract facts, procedures,
                                                  #     constraints from structured text
        materialization_id=state.get("materialization_id"),
                                                  # Links to the previous recall. This is
                                                  # what enables reconsolidation:
                                                  #   "The bullets you gave me in that
                                                  #    recall? They helped (or didn't)."
                                                  # If None (no prior recall), reconsolidation
                                                  # is skipped but knowledge is still saved.
        source_model="claude-sonnet-4",           # Which LLM generated this text.
                                                  # Metadata only — does NOT affect which
                                                  # model processes the extraction. That's
                                                  # always the server's canonical Reflector.
    )
    return {}


# ──────────────────────────────────────────────────────────
# BUILD THE GRAPH
#
# Four nodes in sequence:
#   setup → recall → respond → commit → END
#
# Every chat message flows through all four steps.
# The only state that persists between calls is the Engram
# context (on the server). The LangGraph state is ephemeral.
# ──────────────────────────────────────────────────────────

graph = StateGraph(State)
graph.add_node("setup", setup_context)
graph.add_node("recall", recall)
graph.add_node("respond", respond)
graph.add_node("commit", commit)

graph.set_entry_point("setup")
graph.add_edge("setup", "recall")
graph.add_edge("recall", "respond")
graph.add_edge("respond", "commit")
graph.add_edge("commit", END)

app = graph.compile()


# ──────────────────────────────────────────────────────────
# CHAT FUNCTION
#
# Each call starts fresh — no conversation history is carried
# between calls. The ONLY continuity comes from Engram.
# ──────────────────────────────────────────────────────────

async def chat(message: str):
    result = await app.ainvoke({
        "messages": [HumanMessage(content=message)],  # Fresh message list each time
        "materialization_id": None,                   # No prior recall yet
    })
    return result["messages"][-1].content             # Return Claude's response


# ──────────────────────────────────────────────────────────
# DEMO: Two "sessions" showing persistent memory
# ──────────────────────────────────────────────────────────

async def main():
    # Session 1: teach the agent some facts
    print(await chat("BM25 outperforms embeddings for keyword-heavy queries"))
    # → Engram stores: [FACT] "BM25 outperforms embedding-based search
    #   for keyword-heavy queries"

    print(await chat("FlashAttention gives 2x speedup with minimal accuracy loss"))
    # → Engram stores: [FACT] "FlashAttention provides ~2x speedup with
    #   minimal accuracy loss"
    # → Reconsolidation updates BM25 bullet's recall stats (it was
    #   materialized for this query and may/may not have been relevant)

    # Session 2: the agent remembers without any conversation history
    print(await chat("What do I know about search and attention?"))
    # → recall() finds both bullets via semantic similarity to
    #   "search and attention"
    # → Claude sees them as memories and synthesizes an answer
    # → commit() saves the summary + reconsolidation boosts both
    #   bullets (they were recalled and the task succeeded)

asyncio.run(main())
```

## Architecture

Engram's architecture maps directly to how human memory works:

```
┌──────────────────────────────────────────────────────────┐
│                    BRAIN ANALOGY                          │
│                                                          │
│  Prefrontal Cortex  =  LLM Context Window                │
│  Working Memory     =  Core Memory slot (always loaded)  │
│  Hippocampus        =  Ingestion (Reflector + Curator)   │
│  Neocortex          =  Concept Graph + Schemas           │
│  Amygdala           =  Salience Scorer                   │
│  Episodic Memory    =  Episodic Bullets (timestamped)    │
│  Sleep/Dreams       =  Consolidation Engine              │
│  Recall             =  Materialization Engine             │
│  Pattern Completion =  Worked-Example Retrieval          │
│  Reconsolidation    =  Post-Materialization Feedback     │
│  Forgetting Curve   =  Salience Decay                    │
│  Schema Formation   =  Schema Induction                  │
│  Synaptic Pruning   =  Data Lifecycle (Archive/Purge)    │
│  Source Memory      =  Raw Input Preservation            │
│  Brain Transplant   =  Re-Extraction                     │
└──────────────────────────────────────────────────────────┘
```

### System Overview

![System Architecture](diagrams/architecture.png)

### Key Concepts

- **Bullet** — Atomic unit of knowledge (fact, decision, strategy, warning, principle, exception, procedure, episodic) with usage tracking. Bullets that prove useful get stronger over time.
- **Core Memory** — Mem-α-inspired ≤512-token blob per context, always loaded at recall time. The Reflector decides whether to rewrite it on each commit. Edits are themselves delta ops (`UPDATE_CORE_MEMORY`), so they're auditable and rollback-able.
- **Worked-Example Retrieval** — Dynamic-Cheatsheet-inspired. At recall time, the activity ledger is searched for the nearest prior raw input by cosine similarity (default ≥ 0.85). Hits are attached as last-position worked examples in the rendered prompt so the consuming LLM sees a precedent next to the question.
- **MMR Diversity Ranking** — Maximal Marginal Relevance (`λ * relevance − (1 − λ) * max_sim_to_picked`) replaces the greedy-by-score pack so the token budget doesn't fill with near-duplicates. λ is tunable per call.
- **Validity Gate** *(opt-in)* — Mem-α-inspired batched LM-judge run by the Curator before committing `ADD_BULLET` ops. Rejects empty / trivial / malformed candidates. One extra LLM call per commit; off by default.
- **Usage Stats in Render** *(opt-in)* — Dynamic-Cheatsheet-inspired. Each bullet's `(used N×, success Y/Z)` is surfaced in the rendered prompt so the consuming LLM can weight proven items.
- **SchemaNode** — Abstract pattern derived from repeated experiences. Enables efficient context encoding — only deltas from the expected pattern need storage.
- **DeltaOperation** — Every mutation is an atomic delta op in a DeltaBatch — including bullet add/update/remove/merge, core-memory rewrites, schema changes, and reconsolidation. Context is never regenerated wholesale, preventing context collapse and giving full audit + rollback.
- **IntentAnchor** — Immutable project objective that prevents context drift across sessions.
- **Effective Salience** — `salience × (0.5 + hit_rate)` once a bullet has been recalled enough times — bullets are ranked by proven usefulness, not just raw importance.
- **Reconsolidation** — When recalled context leads to success/failure, bullet usage stats update automatically via a `RECONSOLIDATE_BULLET` delta batch. Context gets smarter with every use.
- **Consolidation** — Background "sleep cycle" that decays unused knowledge, deduplicates, induces schemas, and archives stale bullets.
- **Canonical Reflector** — Server-level model config ensures ALL agents' raw input is processed by the SAME Reflector model. No cross-model extraction variance.
- **Raw Input Preservation** — Every commit stores the original text in the activity ledger (like git commits), along with its embedding (used by worked-example retrieval). Bullets can be regenerated from source.
- **Re-Extraction** — Recompile bullets from raw history with a new/better Reflector model. Like upgrading a compiler and recompiling from source.

### Concurrency Model

Multiple agents can compute in parallel — the lock only serializes delta application:

1. **Parallel phase** — Reflector + Curator run concurrently across agents
2. **Serialize phase** — `ContextLockManager` ensures only one `apply_batch()` runs at a time per context
3. **Revalidation** — Inside the lock, delta operations are checked for stale targets
4. **Event emission** — After commit, an event is pushed to all subscribers

### Data Lifecycle

Bullets follow a three-tier lifecycle: **active → archived → purged**.

- **Active** — Default state. Included in materialization, subject to forgetting curve.
- **Archived** — Low-salience or stale bullets moved to cold storage. Recoverable via restore.
- **Purged** — Permanently deleted after `purge_after_days`. Irrecoverable.

Capacity management with pressure levels:
- **Normal** (<80%) — Standard consolidation
- **High** (80-95%) — Aggressive consolidation (lower dedup threshold, higher archive salience)
- **Critical** (95-100%) — Very aggressive consolidation
- **Full** (100%) — New bullet additions blocked until space is freed

## API Reference

### Context Management

| Endpoint | Method | Description |
|---|---|---|
| `/contexts` | POST | Create a new context with intent anchor |
| `/contexts` | GET | List contexts (with bullet/schema counts) |
| `/contexts/{id}` | GET | Get context metadata |

### Ingestion (Writing)

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/commit` | POST | Ingest content via Reflector/Curator pipeline |
| `/contexts/{id}/bullets` | POST | Add a bullet directly (bypass Reflector) |
| `/contexts/{id}/bullets` | GET | List bullets (filter by section, type) |
| `/contexts/{id}/bullets/{bid}` | GET | Get a single bullet |
| `/contexts/{id}/decisions` | POST | Record a decision |

### Materialization (Reading)

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/materialize` | POST | Materialize context (returns materialization_id) |
| `/contexts/{id}/recall` | POST | Quick recall (simplified materialize) |

### Delta History & Rollback

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/deltas` | GET | View delta operation audit trail |
| `/contexts/{id}/deltas/{did}` | GET | Get a specific delta batch |
| `/contexts/{id}/deltas/{did}/rollback` | POST | Roll back a delta batch |

### Lifecycle Management

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/lifecycle` | GET | Capacity metrics + lifecycle config |
| `/contexts/{id}/bullets/{bid}/archive` | POST | Archive a bullet |
| `/contexts/{id}/bullets/{bid}/restore` | POST | Restore an archived bullet |
| `/contexts/{id}/archived-bullets` | GET | Browse archived bullets (paginated) |
| `/contexts/{id}/purge` | DELETE | GDPR erasure — permanently delete all context data |
| `/contexts/{id}/subscribe` | GET | SSE stream for real-time events |
| `/contexts/{id}/sync` | POST | Polling alternative (delta batches since timestamp) |
| `/users/{uid}/purge` | DELETE | Delete all contexts owned by a user |

### Re-Extraction & Config

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/re-extract` | POST | Re-extract bullets from raw history with new model |
| `/config/ingestion` | GET | Get server-level ingestion configuration |
| `/config/ingestion` | PUT | Update ingestion configuration (admin) |

### Schemas & Health

| Endpoint | Method | Description |
|---|---|---|
| `/contexts/{id}/schemas` | GET | List induced schemas |
| `/contexts/{id}/schemas/{sid}` | GET | Get a specific schema |
| `/contexts/{id}/consolidate` | POST | Trigger consolidation ("sleep cycle") |
| `/contexts/{id}/health` | GET | Health metrics (avg salience, stale count, schemas, capacity) |
| `/contexts/{id}/activity` | GET | View activity ledger |
| `/health` | GET | Server health check |

## Integrations

### MCP Server (Claude)

```bash
pip install engram[mcp]
python -m engram.integrations.mcp_server
```

Configure in Claude Code:
```bash
claude mcp add engram -- python -m engram.integrations.mcp_server
```

Or in `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["-m", "engram.integrations.mcp_server"],
      "env": {
        "ENGRAM_API_URL": "http://localhost:5820",
        "ENGRAM_API_KEY": "eng_sk_..."
      }
    }
  }
}
```

Tools: `engram_list_contexts`, `engram_recall`, `engram_commit`, `engram_decide`, `engram_health`, `engram_create_context`

### OpenAI Function Calling

```python
from engram.integrations.openai_tools import get_engram_tools, EngramToolHandler

tools = get_engram_tools()
handler = EngramToolHandler(engram, default_agent_id="my-gpt-agent")

# Pass tools to your OpenAI agent, handle calls with:
result = await handler.handle_tool_call(function_name, arguments)
```

## Data Model

### Bullet (Atomic Knowledge Unit)

Each bullet is a single actionable insight with full usage and lifecycle tracking:

```
Bullet {
  id, section, content, bullet_type, source_type,
  hit_count, miss_count, recall_count,        # Usage tracking
  salience, confidence,                        # Lifecycle
  effective_salience = salience x (0.5 + hit_rate),  # Computed
  lifecycle_state,                             # active | archived | purged
  archive_reason,                              # Why it was archived
}
```

**Bullet types:** `fact`, `decision`, `strategy`, `warning`, `procedure`, `exception`, `principle`, `episodic` (timestamped events; merged with a looser similarity threshold and excluded from contradiction detection)

### SchemaNode (Abstract Pattern)

Derived from repeated experiences. Enables efficient context encoding:

```
SchemaNode {
  id, name, description,
  instance_count, confidence,
  bullet_ids                                   # Associated bullets
}
```

### DeltaOperation (Atomic Mutation)

Every graph change is tracked:

```
DeltaBatch {
  operations: [
    { op_type: ADD_BULLET | UPDATE_BULLET | REMOVE_BULLET | MERGE_BULLETS | ... }
  ],
  bullets_added, bullets_updated, bullets_removed, bullets_merged
}
```

## Configuration Reference

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| **Server** | | |
| `ENGRAM_HOST` | `0.0.0.0` | Server bind address |
| `ENGRAM_PORT` | `5820` | Server port |
| **Database** | | |
| `ENGRAM_STORAGE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `ENGRAM_SQLITE_PATH` | `./engram.db` | SQLite database path |
| `ENGRAM_POSTGRES_DSN` | — | PostgreSQL connection string (DSN) |
| **LLM API Keys** | | |
| `ANTHROPIC_API_KEY` | — | For Reflector + Curator (server-side only) |
| `OPENAI_API_KEY` | — | For embeddings (server-side only) |
| **Canonical Reflector** | | |
| `ENGRAM_REFLECTOR_MODEL` | `claude-haiku-4-5` | Model that processes ALL raw input |
| `ENGRAM_REFLECTOR_PROMPT_VERSION` | `v1` | Reflector prompt iteration |
| `ENGRAM_MAX_REFLECTION_ROUNDS` | `2` | Self-refinement iterations |
| **Curator** | | |
| `ENGRAM_CURATOR_DEDUP_THRESHOLD` | `0.92` | Cosine similarity for merging bullets (episodic uses a looser 0.85, same-type only) |
| `ENGRAM_CURATOR_SLOW_PATH_MODEL` | `claude-haiku-4-5` | LLM for complex conflicts |
| `ENGRAM_ENABLE_VALIDITY_GATE` | `false` | Mem-α-inspired batched LM-judge that drops malformed / trivial `ADD_BULLET` candidates before commit |
| `ENGRAM_VALIDITY_GATE_MODEL` | `claude-haiku-4-5` | Model used by the validity gate when enabled |
| **Embedding** | | |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | For bullet similarity search **and** worked-example retrieval over the activity ledger |
| **Consolidation** | | |
| `ENGRAM_CONSOLIDATION_TRIGGER` | `every_10_commits` | When to run consolidation |
| `ENGRAM_FAST_DECAY_RATE` | `0.97` | Per-day salience decay (normal bullets) |
| `ENGRAM_SLOW_DECAY_RATE` | `0.995` | Per-day decay (decisions/schemas) |
| **Auth** | | |
| `ENGRAM_AUTH_ENABLED` | `false` | Enable multi-tenant auth |
| `COGNITO_USER_POOL_ID` | — | AWS Cognito user pool |
| `COGNITO_APP_CLIENT_ID` | — | AWS Cognito app client |

## Development

```bash
git clone https://github.com/softmaxdata/engram.git
cd engram
cp .env.example .env           # Edit with your API keys
pip install -e ".[dev]"
pytest
```

## Acknowledgments & Research

Engram's architecture draws from four primary sources:

### Agentic Context Engineering (ACE)

The Reflector/Curator ingestion pipeline, bullet-point storage structure, delta update mechanism, and execution feedback integration are directly inspired by the ACE framework:

> **Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**
> Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun
> Stanford University, SambaNova Systems
> [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)

Key ACE insights incorporated into Engram:
- **Itemized bullet-point context** rather than free-form text, enabling granular updates
- **Delta operations (ADD/UPDATE/REMOVE)** instead of full context rewrites, preventing context collapse
- **Split Reflector/Curator pipeline** — separating analysis from storage decisions improves extraction quality
- **Execution feedback signals** — structured success/failure signals produce better reflections than parsing raw conversation text
- **Lightweight Curator fast path** — embedding-based dedup handles ~70% of operations without LLM calls

### Dynamic Cheatsheet (DC)

The recall-time retrieval of nearest prior raw inputs as **worked examples**, the optional surfacing of per-bullet usage counters in the rendered prompt, and the explicit "verify before copying" caveat next to retrieved precedents come from Dynamic Cheatsheet:

> **Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory**
> Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, James Zou
> EACL 2026
> [arXiv:2504.07952](https://arxiv.org/abs/2504.07952) · [reference code](https://github.com/suzgunmirac/dynamic-cheatsheet)

Key DC insights incorporated into Engram:
- **Adjacent worked-example injection at recall** — top-k similar prior (input, output) pairs placed last in the rendered context (closest to the question) materially helps generators on code/math-style tasks (the `<worked_examples>` block in Engram)
- **Per-item usage counts surfaced in the prompt** — `(used N×, success Y/Z)` lets the consuming LLM weight proven items
- **Bounded recall blob with "verify, don't copy" framing** — retrieved precedents are semantically close, not necessarily correct

Engram deliberately does **not** adopt DC's wholesale-rewrite update policy or its LLM-incremented usage counters without outcome signal — both are weaknesses DC inherits from lacking structured storage. Engram's delta ops + reconsolidation tied to a real outcome strictly dominate that pattern.

### Mem-α: Learning Memory Construction via Reinforcement Learning

The **core-memory slot**, the **episodic / semantic memory distinction**, and the optional **content-validity gate** at write time come from Mem-α:

> **Mem-α: Learning Memory Construction via Reinforcement Learning**
> Yu Wang, Wen Wang, Ricardo Henao, Lawrence Carin et al.
> [arXiv:2509.25911](https://arxiv.org/abs/2509.25911) · [reference code](https://github.com/wangyu-ustc/Mem-alpha)

Key Mem-α insights incorporated into Engram:
- **Core Memory as an always-in-context slot** — Mem-α's strongest architectural lesson is that an explicit, bounded running summary survives compaction even when retrieval misses. Engram implements this as a ≤512-token blob per context, rewritten by the Reflector and prepended at every recall.
- **Episodic vs semantic memory split** — episodic bullets are timestamped events formatted `At {timestamp}, {actor} {did X}`, merged with a looser threshold (paraphrased event reports are common across agents), and exempted from contradiction detection (events don't contradict — both can be true at different timestamps).
- **Content-validity signal at write time** — Mem-α's γ-reward (an LLM-judge over each candidate write) was indispensable in their ablation. Engram exposes this as an opt-in `enable_validity_gate` in the Curator, batched into a single LLM call per commit.

Because Engram is a runtime/server (not a training framework), it does **not** implement Mem-α's RL training loop. The mechanisms above are the training-free distillation of what the learned policy converges to.

### Cognitive Neuroscience

The reconsolidation loop, schema formation, active forgetting (Ebbinghaus decay), consolidation ("sleep cycle"), and salience-based retrieval are inspired by established neuroscience research on human memory systems, including:

- **Memory reconsolidation** — recalled memories become malleable and are re-stored in modified form (Nader, Schiller & Phelps, 2010)
- **Schema theory** — abstract frameworks built from repeated experience that change how new experiences are encoded (Bartlett, 1932; Piaget; van Kesteren et al., 2012)
- **Forgetting curve** — memory strength decays exponentially without rehearsal, and spaced recall resets the decay (Ebbinghaus, 1885)
- **Consolidation** — background reorganization during sleep that abstracts patterns and strengthens important connections (Born & Wilhelm, 2012)

## License

This project is MIT-licensed.

If you use this in production at a company, we’d love to hear about it:
please email logos@engram.so (company + use case).

For large-scale deployments (>20,000 external users), we offer paid support
and an enterprise agreement — contact logos@engram.so.

