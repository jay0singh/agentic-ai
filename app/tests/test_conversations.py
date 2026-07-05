import uuid

import pytest

from core import conversations


@pytest.fixture(autouse=True)
def require_db():
    try:
        conversations._connect().close()
    except Exception:
        pytest.skip("Postgres not available")


@pytest.fixture
def session_id():
    sid = f"test-conv-{uuid.uuid4()}"
    yield sid
    with conversations._connect() as conn:
        conn.execute("DELETE FROM chat_transcripts WHERE session_id = %s", (sid,))


def test_record_and_get_transcript(session_id):
    conversations.record_turn(session_id, "q1", "a1", {"steps_taken": ["generate"], "trace_id": "t1"})
    conversations.record_turn(session_id, "q2", "a2", {"citations": [{"source": "doc", "distance": 0.2}]})

    turns = conversations.get_transcript(session_id)
    assert [t["question"] for t in turns] == ["q1", "q2"]   # oldest first
    assert turns[0]["details"]["trace_id"] == "t1"
    assert turns[1]["details"]["citations"][0]["source"] == "doc"


def test_listed_with_first_question_as_title(session_id):
    conversations.record_turn(session_id, "What is the return policy?", "30 days.")
    conversations.record_turn(session_id, "Does it cost anything?", "Sometimes.")

    convs = {c["session_id"]: c for c in conversations.list_conversations()}
    assert session_id in convs
    assert convs[session_id]["title"] == "What is the return policy?"
    assert convs[session_id]["turns"] == 2


def test_none_session_is_noop():
    conversations.record_turn(None, "q", "a")  # must not raise or store


def test_delete_conversation(session_id):
    conversations.record_turn(session_id, "q", "a")
    assert conversations.delete_conversation(session_id) == 1
    assert conversations.get_transcript(session_id) == []
    assert conversations.delete_conversation(session_id) == 0
