import requests
from langchain_tavily import TavilySearch


class GooglePlaceSearchTool:
    """Google Places API (New) text search.

    The legacy Places API (used by langchain's GooglePlacesTool) is not
    enabled for new Google Cloud projects, so this calls the new endpoint
    directly: https://developers.google.com/maps/documentation/places/web-service/text-search
    """

    URL = "https://places.googleapis.com/v1/places:searchText"
    FIELD_MASK = (
        "places.displayName,places.formattedAddress,places.rating,"
        "places.userRatingCount,places.priceLevel,places.primaryTypeDisplayName"
    )

    def __init__(self, api_key: str, max_results: int = 8):
        self.api_key = api_key
        self.max_results = max_results

    def _search(self, query: str) -> str:
        if not self.api_key:
            raise RuntimeError("GPLACES_API_KEY is not set")
        response = requests.post(
            self.URL,
            json={"textQuery": query, "pageSize": self.max_results},
            headers={"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": self.FIELD_MASK},
            timeout=15,
        )
        data = response.json()
        if response.status_code != 200:
            raise RuntimeError(data.get("error", {}).get("message", f"HTTP {response.status_code}"))

        lines = []
        for i, p in enumerate(data.get("places", []), 1):
            name = p.get("displayName", {}).get("text", "Unknown")
            kind = p.get("primaryTypeDisplayName", {}).get("text")
            line = f"{i}. {name}" + (f" ({kind})" if kind else "")
            if p.get("rating"):
                line += f" - rating {p['rating']} ({p.get('userRatingCount', 0)} reviews)"
            if p.get("priceLevel"):
                line += f", {p['priceLevel'].replace('PRICE_LEVEL_', '').lower().replace('_', ' ')}"
            if p.get("formattedAddress"):
                line += f" - {p['formattedAddress']}"
            lines.append(line)
        return "\n".join(lines)

    def google_search_attractions(self, place: str) -> str:
        """Searches for tourist attractions in the specified place."""
        return self._search(f"top tourist attractions in {place}")

    def google_search_restaurants(self, place: str) -> str:
        """Searches for top restaurants in the specified place."""
        return self._search(f"best restaurants in {place}")

    def google_search_activity(self, place: str) -> str:
        """Searches for popular activities in the specified place."""
        return self._search(f"things to do and activities in {place}")

    def google_search_transportation(self, place: str) -> str:
        """Searches for transport hubs (stations, bus stands, rentals) in the specified place."""
        return self._search(f"bus stations, railway stations, airports and vehicle rentals in {place}")

class TavilyPlaceSearchTool:
    def __init__(self):
        pass

    def tavily_search_attractions(self, place: str) -> dict:
        """
        Searches for attractions in the specified place using TavilySearch.
        """
        tavily_tool = TavilySearch(topic="general", include_answer="advanced")
        result = tavily_tool.invoke({"query": f"top attractive places in and around {place}"})
        if isinstance(result, dict) and result.get("answer"):
            return result["answer"]
        return result
    
    def tavily_search_restaurants(self, place: str) -> dict:
        """
        Searches for available restaurants in the specified place using TavilySearch.
        """
        tavily_tool = TavilySearch(topic="general", include_answer="advanced")
        result = tavily_tool.invoke({"query": f"what are the top 10 restaurants and eateries in and around {place}."})
        if isinstance(result, dict) and result.get("answer"):
            return result["answer"]
        return result
    
    def tavily_search_activity(self, place: str) -> dict:
        """
        Searches for popular activities in the specified place using TavilySearch.
        """
        tavily_tool = TavilySearch(topic="general", include_answer="advanced")
        result = tavily_tool.invoke({"query": f"activities in and around {place}"})
        if isinstance(result, dict) and result.get("answer"):
            return result["answer"]
        return result

    def tavily_search_transportation(self, place: str) -> dict:
        """
        Searches for available modes of transportation in the specified place using TavilySearch.
        """
        tavily_tool = TavilySearch(topic="general", include_answer="advanced")
        result = tavily_tool.invoke({"query": f"What are the different modes of transportations available in {place}"})
        if isinstance(result, dict) and result.get("answer"):
            return result["answer"]
        return result
    
