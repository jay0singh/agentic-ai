import uuid

import pytest

from core import memory


@pytest.fixture(autouse=True)
def require_db():
    """Memory is Postgres-backed; skip these tests when no database is reachable
    (they run against the local pgvector-db container and the CI service)."""
    try:
        memory._connect().close()
    except Exception:
        pytest.skip("Postgres not available")


@pytest.fixture
def session_id():
    sid = f"test-mem-{uuid.uuid4()}"
    yield sid
    with memory._connect() as conn:
        conn.execute("DELETE FROM chat_memory WHERE session_id = %s", (sid,))


@pytest.fixture
def other_session():
    sid = f"test-mem-{uuid.uuid4()}"
    yield sid
    with memory._connect() as conn:
        conn.execute("DELETE FROM chat_memory WHERE session_id = %s", (sid,))


def test_add_and_get_turns(session_id):
    memory.add_turn(session_id, "q1", "a1")
    memory.add_turn(session_id, "q2", "a2")
    assert memory.get_history(session_id) == [("q1", "a1"), ("q2", "a2")]


def test_history_survives_reconnect(session_id):
    """The whole point of Postgres-backed memory: nothing lives in the process."""
    memory.add_turn(session_id, "q1", "a1")
    # A fresh connection (as after a server restart) still sees the turn
    assert memory.get_history(session_id) == [("q1", "a1")]


def test_sessions_are_isolated(session_id, other_session):
    memory.add_turn(session_id, "q1", "a1")
    memory.add_turn(other_session, "other", "answer")
    assert memory.get_history(session_id) == [("q1", "a1")]
    assert memory.get_history(other_session) == [("other", "answer")]


def test_none_session_is_noop():
    memory.add_turn(None, "q", "a")
    assert memory.get_history(None) == []


def test_history_caps_at_max_turns(session_id):
    for i in range(10):
        memory.add_turn(session_id, f"q{i}", f"a{i}")
    history = memory.get_history(session_id)
    assert len(history) == memory._MAX_TURNS
    assert history[-1] == ("q9", "a9")   # newest kept
    assert history[0] == (f"q{10 - memory._MAX_TURNS}", f"a{10 - memory._MAX_TURNS}")


def test_answer_truncated(session_id):
    memory.add_turn(session_id, "q", "x" * 10_000)
    _, answer = memory.get_history(session_id)[0]
    assert len(answer) == memory._MAX_ANSWER_CHARS


def test_none_answer_stored_as_empty(session_id):
    memory.add_turn(session_id, "q", None)
    assert memory.get_history(session_id) == [("q", "")]


def test_db_failure_degrades_gracefully(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memory, "_connect", boom)
    memory.add_turn("test-mem-db-down", "q", "a")          # must not raise
    assert memory.get_history("test-mem-db-down") == []    # must not raise


def test_format_history():
    text = memory.format_history([("q1", "a1"), ("q2", "a2")])
    assert text == "User: q1\nAssistant: a1\nUser: q2\nAssistant: a2"
    assert memory.format_history([]) == ""
