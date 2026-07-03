import threading
from collections import deque

_MAX_TURNS = 4           # remembered Q&A pairs per session
_MAX_ANSWER_CHARS = 400  # truncate stored answers to keep prompts small
_MAX_SESSIONS = 500      # evict the oldest session beyond this

_lock = threading.Lock()
_sessions: dict[str, deque] = {}


def add_turn(session_id: str | None, question: str, answer: str | None) -> None:
    """Record one Q&A exchange for a session."""
    if not session_id:
        return
    with _lock:
        if session_id not in _sessions and len(_sessions) >= _MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        turns = _sessions.setdefault(session_id, deque(maxlen=_MAX_TURNS))
        turns.append((question, (answer or "")[:_MAX_ANSWER_CHARS]))


def get_history(session_id: str | None) -> list[tuple[str, str]]:
    """Return the recorded (question, answer) turns for a session, oldest first."""
    if not session_id:
        return []
    with _lock:
        return list(_sessions.get(session_id, ()))


def format_history(history: list[tuple[str, str]]) -> str:
    """Render turns as a compact transcript for use inside prompts."""
    lines = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)
