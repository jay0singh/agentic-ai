import pytest

from core import memory


@pytest.fixture(autouse=True)
def clean_store():
    memory._sessions.clear()
    yield
    memory._sessions.clear()


def test_add_and_get_turns():
    memory.add_turn("s1", "q1", "a1")
    memory.add_turn("s1", "q2", "a2")
    assert memory.get_history("s1") == [("q1", "a1"), ("q2", "a2")]


def test_sessions_are_isolated():
    memory.add_turn("s1", "q1", "a1")
    memory.add_turn("s2", "other", "answer")
    assert memory.get_history("s1") == [("q1", "a1")]
    assert memory.get_history("s2") == [("other", "answer")]


def test_none_session_is_noop():
    memory.add_turn(None, "q", "a")
    assert memory.get_history(None) == []


def test_history_caps_at_max_turns():
    for i in range(10):
        memory.add_turn("s1", f"q{i}", f"a{i}")
    history = memory.get_history("s1")
    assert len(history) == memory._MAX_TURNS
    assert history[-1] == ("q9", "a9")  # newest kept, oldest dropped


def test_answer_truncated():
    memory.add_turn("s1", "q", "x" * 10_000)
    _, answer = memory.get_history("s1")[0]
    assert len(answer) == memory._MAX_ANSWER_CHARS


def test_none_answer_stored_as_empty():
    memory.add_turn("s1", "q", None)
    assert memory.get_history("s1") == [("q", "")]


def test_session_eviction():
    for i in range(memory._MAX_SESSIONS + 5):
        memory.add_turn(f"s{i}", "q", "a")
    assert len(memory._sessions) == memory._MAX_SESSIONS
    assert memory.get_history("s0") == []                # oldest evicted
    assert memory.get_history(f"s{memory._MAX_SESSIONS + 4}")  # newest kept


def test_format_history():
    text = memory.format_history([("q1", "a1"), ("q2", "a2")])
    assert text == "User: q1\nAssistant: a1\nUser: q2\nAssistant: a2"
    assert memory.format_history([]) == ""
