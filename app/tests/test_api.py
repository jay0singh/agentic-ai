import json

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
    r = client.post("/ingest", files={"file": ("data.csv", b"a,b,c", "text/csv")})
    assert r.status_code == 400


def test_ingest_accepts_txt(client, monkeypatch):
    monkeypatch.setattr(api, "load_document", lambda p: "some extracted text")
    monkeypatch.setattr(api, "setup_table", lambda: None)
    monkeypatch.setattr(api, "embed_and_store", lambda chunks, source: 0)

    r = client.post("/ingest", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_stored"] >= 1


def test_ingest_url_success(client, monkeypatch):
    monkeypatch.setattr(api, "load_url", lambda u: ("Page Title", "page text content"))
    monkeypatch.setattr(api, "setup_table", lambda: None)
    captured = {}

    def fake_embed(chunks, source):
        captured["source"] = source
        return 3  # embed_and_store now reports how many old chunks it replaced

    monkeypatch.setattr(api, "embed_and_store", fake_embed)

    r = client.post("/ingest/url", json={"url": "https://example.com/article"})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Page Title"
    assert body["chunks_replaced"] == 3
    assert captured["source"] == "https://example.com/article"


def test_ingest_url_rejects_bad_scheme(client):
    r = client.post("/ingest/url", json={"url": "ftp://example.com/file"})
    assert r.status_code == 400


def test_ingest_url_fetch_failure_is_400_without_internals(client, monkeypatch):
    def boom(url):
        raise RuntimeError("connection refused to internal-host:5432")
    monkeypatch.setattr(api, "load_url", boom)

    r = client.post("/ingest/url", json={"url": "https://example.com"})
    assert r.status_code == 400
    assert "internal-host" not in r.text


def test_list_documents(client, monkeypatch):
    monkeypatch.setattr(api, "list_documents", lambda: [
        {"source": "handbook.docx", "chunks": 117, "ingested_at": "2026-07-03T10:00:00+00:00"},
    ])
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert docs[0]["source"] == "handbook.docx"
    assert docs[0]["chunks"] == 117


def test_delete_document(client, monkeypatch):
    captured = {}

    def fake_delete(source):
        captured["source"] = source
        return 7

    monkeypatch.setattr(api, "delete_by_source", fake_delete)
    r = client.delete("/documents/handbook.docx")
    assert r.status_code == 200
    assert r.json()["chunks_deleted"] == 7
    assert captured["source"] == "handbook.docx"


def test_delete_missing_document_is_404(client, monkeypatch):
    monkeypatch.setattr(api, "delete_by_source", lambda source: 0)
    r = client.delete("/documents/nope.docx")
    assert r.status_code == 404


def test_delete_url_source_document(client, monkeypatch):
    captured = {}

    def fake_delete(source):
        captured["source"] = source
        return 4

    monkeypatch.setattr(api, "delete_by_source", fake_delete)
    r = client.delete("/documents/https://example.com/article")
    assert r.status_code == 200
    assert captured["source"] == "https://example.com/article"


def test_hitl_list(client, monkeypatch):
    monkeypatch.setattr(api.hitl, "list_pending", lambda: [
        {"id": 1, "question": "Unanswerable?", "attempted_answer": "dunno",
         "judge_reason": "RETRY: not grounded", "created_at": "2026-07-04T10:00:00+00:00"},
    ])
    r = client.get("/hitl")
    assert r.status_code == 200
    assert r.json()["items"][0]["question"] == "Unanswerable?"


def test_hitl_resolve_ingests_answer(client, monkeypatch):
    monkeypatch.setattr(api.hitl, "resolve", lambda item_id, answer: "What is the Q4 policy?")
    monkeypatch.setattr(api, "setup_table", lambda: None)
    captured = {}
    monkeypatch.setattr(api, "embed_and_store",
                        lambda chunks, source: captured.update(chunks=chunks, source=source))

    r = client.post("/hitl/5/resolve", json={"answer": "The Q4 policy is X."})
    assert r.status_code == 200
    assert r.json()["ingested"] is True
    assert captured["source"] == "hitl-5"
    assert "What is the Q4 policy?" in captured["chunks"][0]
    assert "The Q4 policy is X." in captured["chunks"][0]


def test_hitl_resolve_empty_answer_is_400(client):
    r = client.post("/hitl/5/resolve", json={"answer": "   "})
    assert r.status_code == 400


def test_hitl_resolve_missing_item_is_404(client, monkeypatch):
    monkeypatch.setattr(api.hitl, "resolve", lambda item_id, answer: None)
    r = client.post("/hitl/99/resolve", json={"answer": "x"})
    assert r.status_code == 404


def test_hitl_dismiss(client, monkeypatch):
    monkeypatch.setattr(api.hitl, "dismiss", lambda item_id: True)
    assert client.post("/hitl/5/dismiss").status_code == 200
    monkeypatch.setattr(api.hitl, "dismiss", lambda item_id: False)
    assert client.post("/hitl/99/dismiss").status_code == 404


def test_chat_stream_emits_sse_events(client, monkeypatch):
    def fake_stream(question, session_id=None, user_id=None, top_k=3):
        yield {"type": "token", "content": "Hel"}
        yield {"type": "token", "content": "lo"}
        yield {"type": "done", "answer": "Hello", "steps_taken": ["generate", "judge"],
               "context_sources": [], "citations": [], "judge_log": []}

    monkeypatch.setattr(api, "stream_orchestrator", fake_stream)

    with client.stream("POST", "/chat/stream", json={"question": "hi there"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[len("data: "):])
                  for line in r.iter_lines() if line.startswith("data: ")]

    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[-1]["answer"] == "Hello"


def test_chat_stream_empty_question_is_400(client):
    r = client.post("/chat/stream", json={"question": "  "})
    assert r.status_code == 400


def test_chat_stream_internal_error_becomes_error_event(client, monkeypatch):
    def broken_stream(*args, **kwargs):
        raise RuntimeError("postgresql://user:secret@host/db exploded")
        yield  # pragma: no cover — makes this a generator

    monkeypatch.setattr(api, "stream_orchestrator", broken_stream)

    with client.stream("POST", "/chat/stream", json={"question": "hi"}) as r:
        body = "".join(r.iter_text())
    assert '"type": "error"' in body.replace("'", '"') or '"error"' in body
    assert "secret" not in body
