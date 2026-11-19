# 🌍 AI Trip Planner — Multi-Agent Edition

An agentic travel planner built with LangGraph. Originally a single ReAct
agent with 4 tools; now a **supervisor + specialist multi-agent** system with
short & long-term memory, a RAG knowledge base, structured itinerary output,
an eval suite, streaming UX, and Docker/CI deployment.

## Architecture

```
START -> supervisor -> research_agent  ─┐
              │                          ├─(tools)─┐
              ├──────> budget_agent  ────┘         │
              │                                     │
              ├──────> itinerary_agent -> critique_agent
              │                                     │
              └───────────────< back to supervisor ─┘
              │
              └──> FINISH
```

- **supervisor** — routes to the right specialist based on conversation state, caps loops.
- **research_agent** — weather, place search (Google Places + Tavily fallback), and the RAG knowledge base.
- **budget_agent** — hotel/expense calculator + currency conversion.
- **itinerary_agent** — emits a structured `Itinerary` (Pydantic) object, not just prose.
- **critique_agent** — reviews the draft itinerary and either approves it or sends it back for one revision.

## Setup

```bash
pip install -r requirements.txt
cp .env.name .env   # then fill in your API keys
python -m rag.build_knowledge_base   # build the RAG index (run once, or whenever rag/data/*.md changes)
```

Run locally:
```bash
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py
```

Or with Docker:
```bash
docker compose up --build
```

## What's new, and how it works

### 1. Multi-agent architecture
See `agent/agentic_workflow.py`. Each specialist has its own system prompt
(`prompt_library/prompt.py`) and its own scoped tool list — the budget agent
literally cannot call place-search tools, and vice versa. The supervisor is
a small LLM call that only does routing; it never touches tools directly.

### 2. Memory & personalization
- **Short-term (per-conversation):** `utils/checkpointer.py` wires a
  `SqliteSaver` checkpointer into the compiled graph (`checkpoints.db`), so
  a `thread_id` passed to `/query` remembers everything said in that thread —
  even across API restarts. Falls back to an in-memory checkpointer if
  `langgraph-checkpoint-sqlite` isn't installed.
- **Long-term (per-user):** `utils/memory_store.py` is a separate SQLite
  store (`user_profiles.db`) holding durable facts (budget level, dietary
  needs, etc.) and a lightweight trip history. A small dedicated LLM call
  (`extract_memory_facts`) pulls out durable preferences after each message,
  rather than trusting the main agent to manage the profile inline.

### 3. RAG layer
`utils/knowledge_base.py` + `rag/data/*.md`. Ships with a TF-IDF backend
(scikit-learn, zero external services) for stable facts like visa rules and
best-time-to-visit guides — things you don't want re-fetched or hallucinated
on every request. Auto-upgrades to a Chroma + OpenAI-embeddings backend if
`OPENAI_API_KEY` is set and `langchain-chroma`/`chromadb` are installed —
swap to that once your corpus grows past a few hundred chunks. Rebuild the
index offline with `python -m rag.build_knowledge_base`, not on every request.

### 4. Structured output
`prompt_library/schema.py` defines the `Itinerary` Pydantic model.
`itinerary_agent_node` calls `llm.with_structured_output(Itinerary)` and then
**deterministically recomputes** `daily_budget` / `total_estimated_cost` from
the per-activity costs (`GraphBuilder._recompute_costs`) instead of trusting
LLM arithmetic. `utils/itinerary_formatter.py` renders it to markdown for
display; the structured dict is also returned directly in the API response
for downstream use (PDF export, custom UI rendering, etc.).

### 5. Evaluation & observability
- **Observability:** set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` in
  `.env` to get full LangSmith tracing of every agent hop and tool call, no
  code changes needed.
- **Evaluation:** `eval/test_cases.py` is a small golden set checked on
  *properties* (which agents ran, itinerary shape) rather than exact text,
  since LLM output varies. Run with `python -m eval.run_eval --fail-under 0.8`.
  Wired into CI at `.github/workflows/eval.yml` on every PR.

### 6. Better UX
`POST /query/stream` (see `main.py`) emits Server-Sent Events per graph step,
so the Streamlit UI (`streamlit_app.py`) can show live progress
("🔍 Researching...", "💰 Working out costs...") instead of one long spinner,
plus a persistent chat history and thread-based memory across turns.

### 7. Deployment maturity
`Dockerfile.api`, `Dockerfile.streamlit`, and `docker-compose.yml` containerize
both services with health checks. `.github/workflows/eval.yml` runs the eval
suite on every PR. `GET /health` reports which API keys are configured.

## API

```bash
# Blocking, full result:
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan a 3-day trip to Goa", "user_id": "alice"}'

# Streaming with live agent progress:
curl -N -X POST localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan a 3-day trip to Goa", "user_id": "alice"}'

# Health / configured API keys:
curl localhost:8000/health
```

Pass the `thread_id` returned from a previous call back in to continue that
conversation with full memory of what was already discussed.

## Original setup notes (uv / Windows)

```
uv --version
pip install uv
uv init AI_Travel_Planner
uv python install cpython-3.10.18-windows-x86_64-none
uv venv env --python cpython-3.10.18-windows-x86_64-none
# if you have conda active: conda deactivate  first
C:\Users\sunny\AI_Trip_Planner\env\Scripts\activate.bat
uv add pandas
```
