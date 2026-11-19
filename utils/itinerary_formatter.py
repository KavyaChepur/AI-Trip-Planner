from prompt_library.schema import Itinerary


def itinerary_to_markdown(itinerary: Itinerary) -> str:
    lines = [f"# 🌍 Trip to {itinerary.destination}", "", itinerary.trip_summary, ""]

    if itinerary.weather_summary:
        lines += ["## ⛅ Weather", itinerary.weather_summary, ""]

    lines.append("## 🗓️ Day-by-Day Itinerary")
    for day in itinerary.daily_itinerary:
        theme = f" - {day.theme}" if day.theme else ""
        lines.append(f"### Day {day.day_number}{theme}")
        for act in day.activities:
            cost = f" (~{act.estimated_cost} {itinerary.currency})" if act.estimated_cost else ""
            loc = f" @ {act.location}" if act.location else ""
            lines.append(f"- **{act.time}** — {act.title}{loc}{cost}: {act.description}")
        if day.daily_budget:
            lines.append(f"\n*Estimated day budget: {day.daily_budget} {itinerary.currency}*")
        lines.append("")

    if itinerary.alternate_offbeat_days:
        lines.append("## 🧭 Alternate Off-Beat Itinerary")
        for day in itinerary.alternate_offbeat_days:
            lines.append(f"### Day {day.day_number}")
            for act in day.activities:
                lines.append(f"- **{act.time}** — {act.title}: {act.description}")
            lines.append("")

    if itinerary.hotel_options:
        lines.append("## 🏨 Hotel Options")
        for h in itinerary.hotel_options:
            price = f" — ~{h.price_per_night} {itinerary.currency}/night" if h.price_per_night else ""
            area = f" ({h.area})" if h.area else ""
            lines.append(f"- **{h.name}**{area}{price}")
        lines.append("")

    if itinerary.transportation_notes:
        lines += ["## 🚗 Transportation", itinerary.transportation_notes, ""]

    lines.append("## 💰 Cost Summary")
    if itinerary.total_estimated_cost:
        lines.append(f"- **Total estimated cost:** {itinerary.total_estimated_cost} {itinerary.currency}")
    if itinerary.per_day_budget:
        lines.append(f"- **Per-day budget:** {itinerary.per_day_budget} {itinerary.currency}")

    if itinerary.sources_used:
        lines.append("")
        lines.append(f"_Sources used: {', '.join(itinerary.sources_used)}_")

    return "\n".join(lines)
