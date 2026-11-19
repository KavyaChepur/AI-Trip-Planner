from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TripState(TypedDict):
    """Shared state passed between every node in the multi-agent graph."""

    # Full running conversation, including tool calls/results from every sub-agent.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Which agent the supervisor wants to run next ("research_agent", "FINISH", ...).
    next_agent: str

    # Long-term user profile (loaded once at the start of a run, from utils/memory_store.py).
    user_profile: Optional[dict]

    # Structured itinerary draft, filled in by itinerary_agent, checked by critique_agent.
    itinerary_draft: Optional[dict]

    # Feedback from critique_agent, e.g. "REVISE: add indoor backup activities for day 2".
    critique_feedback: Optional[str]

    # Simple loop guard so the supervisor can't ping-pong forever.
    agent_call_counts: dict
