"""
Short-term (within-conversation) memory for the LangGraph agent.

Uses a persistent SQLite-backed checkpointer so conversation history survives
process restarts (`checkpoints.db`). Falls back to the in-memory checkpointer
if `langgraph-checkpoint-sqlite` isn't installed, so the app still runs
out of the box - just without cross-restart persistence.
"""

import sqlite3

_connection = None
_checkpointer = None


def get_checkpointer():
    global _connection, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        _connection = sqlite3.connect("checkpoints.db", check_same_thread=False)
        _checkpointer = SqliteSaver(_connection)
        print("Using SqliteSaver checkpointer (checkpoints.db) - conversations persist across restarts.")
    except Exception as e:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
        print(f"SqliteSaver unavailable ({e}); falling back to in-memory checkpointer "
              f"(conversation history is lost on restart).")

    return _checkpointer
