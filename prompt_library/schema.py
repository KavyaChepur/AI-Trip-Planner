"""
Structured output schemas for the AI Trip Planner.

Forcing the final answer into these Pydantic models (instead of a raw
markdown string) is what unlocks: clean UI rendering, PDF/markdown export,
programmatic cost validation, and downstream automation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Activity(BaseModel):
    time: str = Field(description="Approximate time, e.g. '09:00 AM' or 'Morning'")
    title: str = Field(description="Short name of the activity or place")
    description: str = Field(description="1-3 sentence description")
    location: Optional[str] = Field(default=None, description="Area / neighbourhood / address")
    category: Optional[str] = Field(
        default=None,
        description="One of: attraction, restaurant, activity, transport, hotel, other",
    )
    estimated_cost: Optional[float] = Field(default=None, description="Estimated cost in the trip currency")
    duration_minutes: Optional[int] = Field(default=None, description="Approximate duration in minutes")


class DayPlan(BaseModel):
    day_number: int
    theme: Optional[str] = Field(default=None, description="Short theme for the day, e.g. 'Old Town & Museums'")
    date: Optional[str] = Field(default=None, description="Calendar date if known, else null")
    activities: list[Activity] = Field(default_factory=list)
    daily_budget: Optional[float] = Field(default=None, description="Sum of estimated costs for the day")


class HotelOption(BaseModel):
    name: str
    area: Optional[str] = None
    price_per_night: Optional[float] = None
    notes: Optional[str] = None


class Itinerary(BaseModel):
    destination: Optional[str] = Field(default=None, description="Name of the destination, e.g. 'Goa, India'")
    trip_summary: str = Field(description="2-4 sentence overview of the trip")
    total_days: int = Field(description="Total number of days of the trip, as a plain integer, e.g. 3. This is a COUNT, not a list.")
    currency: str = Field(default="INR", description="ISO currency code used for all cost fields")
    daily_itinerary: list[DayPlan] = Field(
        description="REQUIRED. A JSON ARRAY with one object per day of the trip. "
        "This must be an array of day objects (NOT a number, and NOT named 'days' or 'itinerary')."
    )
    alternate_offbeat_days: Optional[list[DayPlan]] = Field(
        default=None, description="Optional alternate off-beat itinerary (array of day objects), if generated"
    )
    hotel_options: list[HotelOption] = Field(default_factory=list)
    weather_summary: Optional[str] = Field(default=None, description="Brief current/forecast weather summary")
    transportation_notes: Optional[str] = Field(default=None)
    total_estimated_cost: Optional[float] = Field(default=None, description="Sum of all daily budgets")
    per_day_budget: Optional[float] = Field(default=None, description="total_estimated_cost / total_days")
    sources_used: list[str] = Field(
        default_factory=list, description="Which tools/knowledge sources contributed (weather, places, rag, etc.)"
    )


class MemoryFact(BaseModel):
    """A single durable preference extracted from conversation, for long-term memory."""

    key: str = Field(description="short snake_case key, e.g. 'budget_level', 'dietary'")
    value: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)