import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import TripState
from prompt_library.prompt import (
    BUDGET_AGENT_PROMPT,
    CRITIQUE_AGENT_PROMPT,
    ITINERARY_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from prompt_library.schema import Itinerary
from tools.currency_conversion_tool import CurrencyConverterTool
from tools.expense_calculator_tool import CalculatorTool
from tools.knowledge_base_tool import KnowledgeBaseTool
from tools.place_search_tool import PlaceSearchTool
from tools.weather_info_tool import WeatherInfoTool
from utils.itinerary_formatter import itinerary_to_markdown
from utils.model_loader import ModelLoader

AGENT_NAMES = ["research_agent", "budget_agent", "itinerary_agent", "critique_agent"]
MAX_CALLS_PER_AGENT = 3
MAX_TOTAL_HOPS = 14
MAX_REVISIONS = 2  # critique -> itinerary rewrites before handing back to the supervisor

# Character caps applied to message contents before each LLM call. Without
# them the full history (raw web-search results etc.) is resent on every hop
# and quickly exceeds small per-request token limits (e.g. Groq free tier).
SUPERVISOR_MSG_CHARS = 400
AGENT_MSG_CHARS = 1500


def _compact_messages(messages, max_chars: int):
    """Return a copy of `messages` with long string contents truncated."""
    compacted = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str) and len(content) > max_chars:
            m = m.model_copy(update={"content": content[:max_chars] + " ...[truncated]"})
        compacted.append(m)
    return compacted


class GraphBuilder:
    """
    Supervisor + specialist multi-agent architecture:

        START -> supervisor -> {research_agent, budget_agent, itinerary_agent,
                                 critique_agent, FINISH}

    research_agent / budget_agent can call tools (routed through a shared
    ToolNode and back to themselves) before reporting back to the supervisor.
    itinerary_agent emits a structured Itinerary object which critique_agent
    reviews; on approval, control returns to the supervisor which then FINISHes.
    """

    def __init__(self, model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        # --- tool groups, scoped per specialist agent -------------------
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.knowledge_base_tools = KnowledgeBaseTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()

        self.research_tools = [
            *self.weather_tools.weather_tool_list,
            *self.place_search_tools.place_search_tool_list,
            *self.knowledge_base_tools.knowledge_base_tool_list,
        ]
        self.budget_tools = [
            *self.calculator_tools.calculator_tool_list,
            *self.currency_converter_tools.currency_converter_tool_list,
        ]
        self.all_tools = [*self.research_tools, *self.budget_tools]

        # --- per-agent LLM bindings --------------------------------------
        # with_retry: providers (e.g. Groq) occasionally reject a malformed
        # tool call with a 400 "output_parse_failed"; a resample usually works.
        self.research_llm = self.llm.bind_tools(tools=self.research_tools).with_retry(stop_after_attempt=3)
        self.budget_llm = self.llm.bind_tools(tools=self.budget_tools).with_retry(stop_after_attempt=3)
        # Groq's native JSON-schema mode is far more reliable than forced
        # tool-calling for this large schema (forced tool calls often come
        # back empty with "Tool choice is required, but model did not call a tool").
        if model_provider == "groq":
            structured_base = self.llm
            if "gpt-oss" in getattr(self.llm, "model_name", ""):
                # Default (medium) reasoning with temp 0.7 frequently returns an
                # empty generation in JSON-schema mode (~2/5 success when
                # measured); low effort + low temperature succeeded 5/5.
                structured_base = self.llm.model_copy(update={"reasoning_effort": "low", "temperature": 0.2})
            self.itinerary_llm = structured_base.with_structured_output(Itinerary, method="json_schema")
        else:
            self.itinerary_llm = self.llm.with_structured_output(Itinerary)

        self.graph = None

    # ---------------------------------------------------------------- #
    # Nodes
    # ---------------------------------------------------------------- #
    def supervisor_node(self, state: TripState):
        messages = list(state["messages"])
        counts = dict(state.get("agent_call_counts") or {})

        critique_approved = False
        if messages:
            last_content = getattr(messages[-1], "content", "") or ""
            critique_approved = isinstance(last_content, str) and last_content.strip().upper().startswith("APPROVED")

        # Fast-path: if critique just approved, finish without another LLM call.
        if critique_approved:
            return {"next_agent": "FINISH"}

        if sum(counts.values()) >= MAX_TOTAL_HOPS:
            return {"next_agent": "FINISH"}

        sys = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
        response = self.llm.invoke([sys] + _compact_messages(messages, SUPERVISOR_MSG_CHARS))
        choice = (response.content or "").strip().split()[0] if response.content else "FINISH"
        choice = choice.strip(".:,")

        if choice not in {*AGENT_NAMES, "FINISH"}:
            choice = None  # fall through to the sane-default logic below

        # Hard guard: FINISH is only legitimate once itinerary_agent has produced
        # a draft AND critique_agent has approved it. This runs regardless of what
        # the LLM said, so a model that "feels satisfied" after budget numbers
        # (or gives a malformed response) can never end the run without an
        # actual itinerary being built.
        itinerary_ran = counts.get("itinerary_agent", 0) > 0
        if choice is None or (choice == "FINISH" and not (itinerary_ran and critique_approved)):
            if counts.get("research_agent", 0) == 0:
                choice = "research_agent"
            elif counts.get("budget_agent", 0) == 0:
                choice = "budget_agent"
            elif not itinerary_ran:
                choice = "itinerary_agent"
            else:
                # itinerary already ran but wasn't approved yet - send it back
                # through critique rather than ending prematurely.
                choice = "critique_agent"

        if choice != "FINISH" and counts.get(choice, 0) >= MAX_CALLS_PER_AGENT:
            choice = "itinerary_agent" if choice != "itinerary_agent" else "FINISH"

        if choice != "FINISH":
            counts[choice] = counts.get(choice, 0) + 1

        return {"next_agent": choice, "agent_call_counts": counts}

    def research_agent_node(self, state: TripState):
        messages = _compact_messages(state["messages"], AGENT_MSG_CHARS)
        profile_note = self._profile_note(state)
        input_messages = [RESEARCH_AGENT_PROMPT]
        if profile_note:
            input_messages.append(SystemMessage(content=profile_note))
        response = self.research_llm.invoke(input_messages + messages)
        return {"messages": [response], "next_agent": "research_agent"}

    def budget_agent_node(self, state: TripState):
        messages = _compact_messages(state["messages"], AGENT_MSG_CHARS)
        response = self.budget_llm.invoke([BUDGET_AGENT_PROMPT] + messages)
        return {"messages": [response], "next_agent": "budget_agent"}

    def itinerary_agent_node(self, state: TripState):
        messages = _compact_messages(state["messages"], AGENT_MSG_CHARS)
        sys_content = ITINERARY_AGENT_PROMPT
        feedback = state.get("critique_feedback")
        if feedback and feedback.strip().upper().startswith("REVISE"):
            sys_content += f"\n\nThe previous draft needs revision: {feedback}"
        sys = SystemMessage(content=sys_content)

        # Flatten the history (incl. tool calls / tool results) into one plain
        # text message: sending raw tool-call messages to a structured-output
        # call with no tools bound makes some providers (Groq) return empty output.
        context = HumanMessage(content="Conversation and research so far:\n\n" + self._transcript(messages))
        itinerary_obj = self._invoke_itinerary_llm_with_retry([sys, context])

        # Backfill destination from the conversation if the model dropped it -
        # cheap, deterministic, and avoids a second LLM round-trip for one field.
        if not itinerary_obj.destination:
            itinerary_obj.destination = self._guess_destination(messages)

        # Deterministic cost validation/repair instead of trusting LLM arithmetic.
        itinerary_obj = self._recompute_costs(itinerary_obj)

        markdown = itinerary_to_markdown(itinerary_obj)
        ai_msg = AIMessage(content=markdown)
        return {
            "messages": [ai_msg],
            "itinerary_draft": itinerary_obj.model_dump(),
            "next_agent": "itinerary_agent",
        }

    @staticmethod
    def _transcript(messages) -> str:
        """Render messages as plain text, labelled by role / tool name."""
        lines = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if not content.strip():
                continue  # e.g. an AI message that only contained tool calls
            if m.type == "tool":
                label = f"Tool result ({getattr(m, 'name', None) or 'tool'})"
            elif m.type == "human":
                label = "User"
            else:
                label = "Assistant"
            lines.append(f"{label}: {content}")
        return "\n\n".join(lines)

    def _invoke_itinerary_llm_with_retry(self, input_messages, max_retries: int = 3) -> Itinerary:
        """Structured-output calls occasionally fail schema validation (e.g. the
        model drops a required field on a large JSON payload). Retry once with
        an explicit corrective instruction before giving up, rather than
        failing the whole conversation over one malformed call."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.itinerary_llm.invoke(input_messages)
            except Exception as e:  # noqa: BLE001
                last_error = e
                input_messages = input_messages + [
                    SystemMessage(
                        content=(
                            "Your previous response failed schema validation: "
                            f"{str(e)[:500]}. Re-generate the FULL itinerary again, making sure "
                            "every required field is present and every field name "
                            "matches the schema exactly."
                        )
                    )
                ]
        raise last_error

    @staticmethod
    def _guess_destination(messages) -> str:
        """Best-effort fallback: pull a plausible destination string from the
        first human message if the model omitted the structured field."""
        for m in messages:
            if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                content = m.content if isinstance(m.content, str) else str(m.content)
                return content.strip()[:80]
        return "Unknown destination"

    def critique_agent_node(self, state: TripState):
        draft = state.get("itinerary_draft") or {}
        human = HumanMessage(content=f"Itinerary draft (JSON):\n{json.dumps(draft, indent=2, default=str)}")
        response = self.llm.invoke([CRITIQUE_AGENT_PROMPT, human])
        counts = dict(state.get("agent_call_counts") or {})
        counts["critique_reviews"] = counts.get("critique_reviews", 0) + 1
        return {"messages": [response], "critique_feedback": response.content, "agent_call_counts": counts}

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _profile_note(state: TripState) -> str:
        profile = state.get("user_profile") or {}
        if not profile:
            return ""
        return f"Known preferences for this user (use to tailor results, don't ask again unless unclear): {json.dumps(profile)}"

    @staticmethod
    def _recompute_costs(itinerary: Itinerary) -> Itinerary:
        """Recalculate daily/total budgets from activity-level costs so the
        numbers are internally consistent, rather than trusting whatever the
        LLM wrote for the summary fields."""
        if itinerary.daily_itinerary and itinerary.total_days != len(itinerary.daily_itinerary):
            itinerary.total_days = len(itinerary.daily_itinerary)

        total = 0.0
        any_cost = False
        for day in itinerary.daily_itinerary:
            day_total = sum(a.estimated_cost for a in day.activities if a.estimated_cost)
            if day_total:
                day.daily_budget = round(day_total, 2)
                total += day_total
                any_cost = True
        if any_cost:
            itinerary.total_estimated_cost = round(total, 2)
            if itinerary.total_days:
                itinerary.per_day_budget = round(total / itinerary.total_days, 2)
        return itinerary

    # ---------------------------------------------------------------- #
    # Graph wiring
    # ---------------------------------------------------------------- #
    @staticmethod
    def _route_after_tool_capable_agent(state: TripState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "supervisor"

    @staticmethod
    def _route_after_tools(state: TripState):
        # Tools always report back to whichever agent invoked them.
        return state.get("next_agent", "research_agent")

    @staticmethod
    def _route_after_critique(state: TripState):
        last_content = (state["messages"][-1].content or "").strip().upper()
        if last_content.startswith("APPROVED"):
            return "supervisor"
        # This edge bypasses the supervisor's loop guards, so cap revisions here;
        # otherwise a critic that never approves loops until the recursion limit.
        reviews = (state.get("agent_call_counts") or {}).get("critique_reviews", 0)
        if reviews > MAX_REVISIONS:
            return "supervisor"
        return "itinerary_agent"

    def build_graph(self, checkpointer=None):
        graph_builder = StateGraph(TripState)

        graph_builder.add_node("supervisor", self.supervisor_node)
        graph_builder.add_node("research_agent", self.research_agent_node)
        graph_builder.add_node("budget_agent", self.budget_agent_node)
        graph_builder.add_node("itinerary_agent", self.itinerary_agent_node)
        graph_builder.add_node("critique_agent", self.critique_agent_node)
        graph_builder.add_node("tools", ToolNode(tools=self.all_tools))

        graph_builder.add_edge(START, "supervisor")

        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next_agent"],
            {
                "research_agent": "research_agent",
                "budget_agent": "budget_agent",
                "itinerary_agent": "itinerary_agent",
                "critique_agent": "critique_agent",
                "FINISH": END,
            },
        )

        graph_builder.add_conditional_edges(
            "research_agent",
            self._route_after_tool_capable_agent,
            {"tools": "tools", "supervisor": "supervisor"},
        )
        graph_builder.add_conditional_edges(
            "budget_agent",
            self._route_after_tool_capable_agent,
            {"tools": "tools", "supervisor": "supervisor"},
        )
        graph_builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"research_agent": "research_agent", "budget_agent": "budget_agent"},
        )

        graph_builder.add_edge("itinerary_agent", "critique_agent")
        graph_builder.add_conditional_edges(
            "critique_agent",
            self._route_after_critique,
            {"supervisor": "supervisor", "itinerary_agent": "itinerary_agent"},
        )

        self.graph = graph_builder.compile(checkpointer=checkpointer)
        return self.graph

    def __call__(self, checkpointer=None):
        return self.build_graph(checkpointer=checkpointer)