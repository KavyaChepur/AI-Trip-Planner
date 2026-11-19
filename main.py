import asyncio
import json
import os
import queue
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse

from agent.agentic_workflow import GraphBuilder
from prompt_library.schema import Itinerary
from utils.itinerary_formatter import itinerary_to_markdown
from utils.checkpointer import get_checkpointer
from utils.memory_store import UserProfileStore, extract_memory_facts

load_dotenv()

app = FastAPI(title="AI Trip Planner - Multi-Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at startup and reused across requests - this is what makes
# short-term (within-conversation) memory work at all.
checkpointer = get_checkpointer()
profile_store = UserProfileStore()

# Friendly labels shown to the UI while streaming agent progress.
AGENT_LABELS = {
    "supervisor": "🧭 Deciding next step...",
    "research_agent": "🔍 Researching weather, places & travel guides...",
    "budget_agent": "💰 Working out costs...",
    "itinerary_agent": "🗓️ Assembling the itinerary...",
    "critique_agent": "🕵️ Reviewing the draft plan...",
    "tools": "🛠️ Calling a tool...",
}


class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    thread_id: str | None = None  # omit to start a new conversation


def _build_graph(model_provider: str = "groq"):
    builder = GraphBuilder(model_provider=model_provider)
    return builder.build_graph(checkpointer=checkpointer)


def _initial_state(question: str, user_id: str) -> dict:
    profile = profile_store.get_profile(user_id)
    return {
        "messages": [question],
        "next_agent": "",
        "user_profile": profile,
        "itinerary_draft": None,
        "critique_feedback": None,
        "agent_call_counts": {},
    }


def _final_answer(state: dict) -> str:
    """The user-facing answer is the itinerary, not the last message: after a
    successful run the last message is the critique agent's "APPROVED" verdict."""
    itinerary = state.get("itinerary_draft")
    if itinerary:
        try:
            return itinerary_to_markdown(Itinerary(**itinerary))
        except Exception:  # noqa: BLE001
            pass
    messages = state.get("messages") or []
    return messages[-1].content if messages else ""


def _remember_async(user_id: str, question: str, model_provider: str):
    """Best-effort long-term memory extraction. Never let this break the request."""
    try:
        loader_graph = GraphBuilder(model_provider=model_provider)
        facts = extract_memory_facts(loader_graph.llm, question)
        if facts:
            profile_store.upsert_facts(user_id, facts)
    except Exception as e:  # noqa: BLE001
        print(f"Memory extraction skipped: {e}")


@app.get("/health")
async def health():
    required_keys = [
        "GROQ_API_KEY", "OPENAI_API_KEY", "GPLACES_API_KEY",
        "TAVILY_API_KEY", "OPENWEATHERMAP_API_KEY", "EXCHANGE_RATE_API_KEY",
    ]
    status = {k: bool(os.environ.get(k)) for k in required_keys}
    return {"status": "ok", "api_keys_present": status}


def _run_graph_sync(question: str, user_id: str, thread_id: str) -> dict:
    """Blocking work, executed off the event loop via run_in_executor."""
    react_app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    output = react_app.invoke(_initial_state(question, user_id), config=config)

    final_message = _final_answer(output)
    itinerary = output.get("itinerary_draft")

    _remember_async(user_id, question, "groq")
    if itinerary and itinerary.get("destination"):
        profile_store.log_trip(user_id, itinerary["destination"], thread_id)

    return {
        "answer": final_message,
        "itinerary": itinerary,
        "thread_id": thread_id,
        "agent_trace": output.get("agent_call_counts", {}),
    }


@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        thread_id = query.thread_id or str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        # Run the blocking graph invocation in a worker thread so this request
        # doesn't freeze the event loop (and every other concurrent request)
        # while waiting on LLM/tool network calls.
        result = await loop.run_in_executor(None, _run_graph_sync, query.question, query.user_id, thread_id)
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


def _run_graph_blocking(question: str, user_id: str, thread_id: str, out_queue: "queue.Queue"):
    """Runs entirely in a worker thread. Every call in here (LLM calls, tool
    calls, sqlite writes) is blocking I/O - this MUST NOT run directly inside
    an `async def`, or it freezes the whole server's single event loop
    (every other request, including /health, would hang behind it)."""
    try:
        react_app = _build_graph()
        config = {"configurable": {"thread_id": thread_id}}

        for step in react_app.stream(
            _initial_state(question, user_id), config=config, stream_mode="updates"
        ):
            for node_name, _node_output in step.items():
                label = AGENT_LABELS.get(node_name, node_name)
                out_queue.put({"type": "step", "node": node_name, "label": label})

        state = react_app.get_state(config).values
        final_message = _final_answer(state)
        out_queue.put({
            "type": "final",
            "answer": final_message,
            "itinerary": state.get("itinerary_draft"),
            "thread_id": thread_id,
        })

        _remember_async(user_id, question, "groq")
    except Exception as e:  # noqa: BLE001
        out_queue.put({"type": "error", "error": str(e)})
    finally:
        out_queue.put(None)  # sentinel: tells the async side we're done


@app.post("/query/stream")
async def stream_travel_agent(query: QueryRequest):
    """Server-Sent Events stream: emits one event per graph step so the UI
    can show live agent progress instead of one long blocking spinner.

    The actual graph run happens in a background thread (run_in_executor);
    this coroutine only ever does non-blocking queue reads, so it never
    blocks the event loop / other concurrent requests."""

    thread_id = query.thread_id or str(uuid.uuid4())
    out_queue: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    loop.run_in_executor(
        None, _run_graph_blocking, query.question, query.user_id, thread_id, out_queue
    )

    async def event_generator():
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"
        while True:
            # poll the thread-safe queue without blocking the event loop
            try:
                item = out_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
