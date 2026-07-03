import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_empty_question_is_400(client):
    r = client.post("/chat", json={"question": "   "})
    assert r.status_code == 400


def test_chat_passes_session_through_and_shapes_response(client, monkeypatch):
    captured = {}

    def fake_orchestrator(question, session_id=None, user_id=None, top_k=3):
        captured.update(question=question, session_id=session_id, user_id=user_id, top_k=top_k)
        return {
            "response": "an answer",
            "steps_taken": ["vector_search", "generate", "judge"],
            "context": ["chunk"],
            "citations": [{"source": "handbook.docx", "distance": 0.2}],
            "judge_log": ["ACCEPT: fine"],
        }

    monkeypatch.setattr(api, "run_orchestrator", fake_orchestrator)

    r = client.post("/chat", json={"question": "What is X?", "session_id": "s-1", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "an answer"
    assert body["steps_taken"] == ["vector_search", "generate", "judge"]
    assert body["citations"] == [{"source": "handbook.docx", "distance": 0.2}]
    assert captured == {"question": "What is X?", "session_id": "s-1", "user_id": None, "top_k": 5}


def test_chat_rejects_out_of_range_top_k(client):
    r = client.post("/chat", json={"question": "What is X?", "top_k": 0})
    assert r.status_code == 422
    r = client.post("/chat", json={"question": "What is X?", "top_k": 50})
    assert r.status_code == 422


def test_chat_internal_error_is_generic(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("postgresql://user:secret@host/db exploded")

    monkeypatch.setattr(api, "run_orchestrator", boom)

    r = client.post("/chat", json={"question": "What is X?"})
    assert r.status_code == 500
    # Internals (like connection strings) must not leak to the client
    assert "secret" not in r.text
    assert "postgresql" not in r.text


def test_ingest_rejects_unsupported_file_type(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
