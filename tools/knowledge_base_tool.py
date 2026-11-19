from typing import List

from langchain.tools import tool
from utils.knowledge_base import TravelKnowledgeBase


class KnowledgeBaseTool:
    def __init__(self):
        self.kb = TravelKnowledgeBase()
        self.kb.load()
        self.knowledge_base_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        @tool
        def search_travel_knowledge(query: str) -> str:
            """Search the curated travel knowledge base for stable facts like visa
            requirements, best time to visit, local customs, and packing tips.
            Use this for slow-changing background knowledge, NOT for live prices
            or current weather (use the dedicated tools for those)."""
            results = self.kb.query(query, k=3)
            if not results:
                return "No relevant information found in the knowledge base for this query."
            formatted = "\n\n".join(
                f"[Source: {r.source}]\n{r.text}" for r in results
            )
            return formatted

        return [search_travel_knowledge]
