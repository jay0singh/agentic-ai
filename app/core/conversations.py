import json

import psycopg
from dotenv import load_dotenv

from core.embedder import DB_CONFIG

load_dotenv()

_RETENTION_DAYS = 30
_MAX_CONVERSATIONS = 30  # newest conversations listed in the UI

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_transcripts (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""
_INDEX_SQL = 'CREATE INDEX IF NOT EXISTS chat_transcripts_session_idx ON chat_transcripts (session_id, id)'


def _connect():
    conn = psycopg.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(_TABLE_SQL)
        cur.execute(_INDEX_SQL)
    return conn


def record_turn(session_id: str | None, question: str, answer: str, details: dict | None = None) -> None:
    """Persist one full Q&A turn for transcript display. Must never break the chat."""
    if not session_id:
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_transcripts (session_id, question, answer, details) "
                    "VALUES (%s, %s, %s, %s)",
                    (session_id, question, answer, json.dumps(details or {})),
                )
                cur.execute(
                    "DELETE FROM chat_transcripts WHERE created_at < now() - make_interval(days => %s)",
                    (_RETENTION_DAYS,),
                )
    except Exception as e:
        print(f"  [Transcripts] Failed to record turn ({e}).")


def list_conversations() -> list[dict]:
    """Recent conversations, newest activity first: id, title (first question),
    number of turns, last activity."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, (array_agg(question ORDER BY id))[1], COUNT(*), MAX(created_at) "
                    "FROM chat_transcripts GROUP BY session_id "
                    "ORDER BY MAX(created_at) DESC LIMIT %s",
                    (_MAX_CONVERSATIONS,),
                )
                return [
                    {"session_id": sid, "title": title, "turns": turns, "last_at": last_at.isoformat()}
                    for sid, title, turns, last_at in cur.fetchall()
                ]
    except Exception as e:
        print(f"  [Transcripts] Failed to list conversations ({e}).")
        return []


def get_transcript(session_id: str) -> list[dict]:
    """All turns of one conversation, oldest first."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, answer, details FROM chat_transcripts "
                    "WHERE session_id = %s ORDER BY id",
                    (session_id,),
                )
                return [
                    {"question": q, "answer": a, "details": d or {}}
                    for q, a, d in cur.fetchall()
                ]
    except Exception as e:
        print(f"  [Transcripts] Failed to load transcript ({e}).")
        return []


def delete_conversation(session_id: str) -> int:
    """Remove a conversation's transcript. Returns the number of deleted turns."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_transcripts WHERE session_id = %s", (session_id,))
            return cur.rowcount
