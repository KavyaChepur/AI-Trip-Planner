"""
Long-term (cross-session) user memory.

This is deliberately separate from LangGraph's checkpointer (which only holds
*within-conversation* short-term memory). This store holds durable facts about
a user - budget level, dietary needs, preferred pace - that should be reused
across brand new conversations/threads.

Kept intentionally simple (stdlib sqlite3, no extra dependency) so it runs
anywhere without extra setup.
"""

import json
import sqlite3
import datetime
from pathlib import Path
from typing import Optional


class UserProfileStore:
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_facts (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trip_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    destination TEXT,
                    thread_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    # Durable preference facts
    # ------------------------------------------------------------------ #
    def get_profile(self, user_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value, confidence FROM user_facts WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return {key: value for key, value, _confidence in rows}

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 0.8):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_facts (user_id, key, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value=excluded.value,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (user_id, key, value, confidence, datetime.datetime.utcnow().isoformat()),
            )
            conn.commit()

    def upsert_facts(self, user_id: str, facts: list[dict]):
        """facts: list of {"key": ..., "value": ..., "confidence": ...}"""
        for fact in facts:
            self.upsert_fact(
                user_id,
                fact["key"],
                str(fact["value"]),
                float(fact.get("confidence", 0.8)),
            )

    # ------------------------------------------------------------------ #
    # Lightweight trip history (useful context, e.g. "last time you went to Goa...")
    # ------------------------------------------------------------------ #
    def log_trip(self, user_id: str, destination: str, thread_id: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trip_history (user_id, destination, thread_id, created_at) VALUES (?, ?, ?, ?)",
                (user_id, destination, thread_id, datetime.datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_recent_trips(self, user_id: str, limit: int = 5) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT destination, created_at FROM trip_history WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"destination": d, "created_at": c} for d, c in rows]

    def profile_as_prompt_snippet(self, user_id: str) -> str:
        """Render the stored profile as a short text block to inject into a system prompt."""
        profile = self.get_profile(user_id)
        trips = self.get_recent_trips(user_id, limit=3)
        if not profile and not trips:
            return "No known preferences yet for this user."
        lines = []
        if profile:
            lines.append("Known user preferences: " + json.dumps(profile))
        if trips:
            dest_list = ", ".join(t["destination"] for t in trips if t["destination"])
            if dest_list:
                lines.append(f"Recent trips planned: {dest_list}")
        return "\n".join(lines)


def extract_memory_facts(llm, message_text: str) -> list[dict]:
    """Run a cheap LLM pass to pull out durable preferences from a user message.

    Deliberately a separate, small call rather than trusting the main agent to
    manage the profile inline - keeps the profile from getting polluted with
    one-off trip details.
    """
    from prompt_library.prompt import MEMORY_EXTRACTION_PROMPT

    prompt = MEMORY_EXTRACTION_PROMPT.format(message=message_text)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content.split("\n", 1)[-1] if "\n" in content else content
        facts = json.loads(content)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict) and "key" in f and "value" in f]
    except Exception:
        pass
    return []
