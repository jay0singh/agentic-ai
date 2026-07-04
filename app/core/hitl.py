import psycopg
from dotenv import load_dotenv

from core.embedder import DB_CONFIG

load_dotenv()

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hitl_queue (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    attempted_answer TEXT,
    judge_reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    human_answer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
)
"""


def _connect():
    conn = psycopg.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(_TABLE_SQL)
    return conn


def add_to_queue(question: str, attempted_answer: str | None, judge_reason: str | None) -> int:
    """Persist a question the judge could not get answered. Returns the queue id."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO hitl_queue (question, attempted_answer, judge_reason) "
                "VALUES (%s, %s, %s) RETURNING id",
                (question, attempted_answer, judge_reason),
            )
            return cur.fetchone()[0]


def list_pending() -> list[dict]:
    """Pending questions, oldest first."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, question, attempted_answer, judge_reason, created_at "
                "FROM hitl_queue WHERE status = 'pending' ORDER BY created_at"
            )
            return [
                {
                    "id": row[0],
                    "question": row[1],
                    "attempted_answer": row[2],
                    "judge_reason": row[3],
                    "created_at": row[4].isoformat(),
                }
                for row in cur.fetchall()
            ]


def resolve(item_id: int, human_answer: str) -> str | None:
    """Mark a pending item resolved. Returns its question, or None if not found."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE hitl_queue SET status = 'resolved', human_answer = %s, resolved_at = now() "
                "WHERE id = %s AND status = 'pending' RETURNING question",
                (human_answer, item_id),
            )
            row = cur.fetchone()
            return row[0] if row else None


def dismiss(item_id: int) -> bool:
    """Mark a pending item dismissed. Returns False if it wasn't pending."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE hitl_queue SET status = 'dismissed', resolved_at = now() "
                "WHERE id = %s AND status = 'pending' RETURNING id",
                (item_id,),
            )
            return cur.fetchone() is not None
