import psycopg
from dotenv import load_dotenv

from core.embedder import DB_CONFIG

load_dotenv()

_MAX_TURNS = 4           # remembered Q&A pairs per session
_MAX_ANSWER_CHARS = 400  # truncate stored answers to keep prompts small
_RETENTION_DAYS = 7      # drop sessions idle longer than this

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_memory (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""
_INDEX_SQL = 'CREATE INDEX IF NOT EXISTS chat_memory_session_idx ON chat_memory (session_id, id)'


def _connect():
    conn = psycopg.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(_TABLE_SQL)
        cur.execute(_INDEX_SQL)
    return conn


def add_turn(session_id: str | None, question: str, answer: str | None) -> None:
    """Record one Q&A exchange for a session. Memory is persisted in Postgres so
    conversations survive server restarts; failures must never break the chat."""
    if not session_id:
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_memory (session_id, question, answer) VALUES (%s, %s, %s)",
                    (session_id, question, (answer or "")[:_MAX_ANSWER_CHARS]),
                )
                # Keep only the newest turns per session
                cur.execute(
                    "DELETE FROM chat_memory WHERE session_id = %s AND id NOT IN ("
                    "  SELECT id FROM chat_memory WHERE session_id = %s ORDER BY id DESC LIMIT %s)",
                    (session_id, session_id, _MAX_TURNS),
                )
                # Opportunistic cleanup of long-idle sessions
                cur.execute(
                    "DELETE FROM chat_memory WHERE created_at < now() - make_interval(days => %s)",
                    (_RETENTION_DAYS,),
                )
    except Exception as e:
        print(f"  [Memory] Failed to record turn ({e}).")


def get_history(session_id: str | None) -> list[tuple[str, str]]:
    """Return the recorded (question, answer) turns for a session, oldest first."""
    if not session_id:
        return []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, answer FROM chat_memory "
                    "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                    (session_id, _MAX_TURNS),
                )
                rows = cur.fetchall()
        return [(question, answer) for question, answer in reversed(rows)]
    except Exception as e:
        print(f"  [Memory] Failed to load history ({e}).")
        return []


def clear_session(session_id: str | None) -> None:
    """Forget a session's working memory (used when its conversation is deleted)."""
    if not session_id:
        return
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM chat_memory WHERE session_id = %s", (session_id,))
    except Exception as e:
        print(f"  [Memory] Failed to clear session ({e}).")


def format_history(history: list[tuple[str, str]]) -> str:
    """Render turns as a compact transcript for use inside prompts."""
    lines = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)
