"""
Golden eval set. Each case checks *properties* of the run rather than exact
text, since LLM output varies run to run. Run with: python -m eval.run_eval
"""

TEST_CASES = [
    {
        "id": "goa_3day_budget",
        "query": "Plan a 3-day budget trip to Goa, India.",
        "checks": {
            "min_days_covered": 3,
            "must_call_agents": ["research_agent", "itinerary_agent"],
            "itinerary_has_cost_fields": True,
        },
    },
    {
        "id": "tokyo_5day_food",
        "query": "Plan a 5-day trip to Tokyo focused on food and off-beat neighbourhoods, budget in USD.",
        "checks": {
            "min_days_covered": 5,
            "must_call_agents": ["research_agent", "itinerary_agent"],
            "itinerary_has_cost_fields": True,
        },
    },
    {
        "id": "paris_weekend",
        "query": "Weekend trip to Paris for 2 days, mid-range budget.",
        "checks": {
            "min_days_covered": 2,
            "must_call_agents": ["research_agent", "itinerary_agent"],
            "itinerary_has_cost_fields": False,  # short trips may not always price everything
        },
    },
    {
        "id": "currency_conversion",
        "query": "Plan a 4-day trip to Bangkok and show total cost converted to EUR.",
        "checks": {
            "min_days_covered": 4,
            "must_call_agents": ["research_agent", "budget_agent", "itinerary_agent"],
            "itinerary_has_cost_fields": True,
        },
    },
    {
        "id": "vague_query_still_completes",
        "query": "I want to go somewhere relaxing for a week.",
        "checks": {
            "min_days_covered": 1,  # agent should still produce *something* structured
            "must_call_agents": ["research_agent"],
            "itinerary_has_cost_fields": False,
        },
    },
]
