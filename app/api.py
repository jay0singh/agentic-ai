import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.ingestor import load_document, load_url, SUPPORTED_EXTENSIONS
from core.chunker import chunk_text
from core.embedder import setup_table, embed_and_store, delete_by_source, list_documents
from core.retriever import retrieve
from core.generator import run_orchestrator
from core.graph import stream_orchestrator
from core.memory import clear_session
from core import conversations
from core import hitl


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush any queued Langfuse events before the process exits so traces aren't lost.
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client
            get_client().shutdown()
        except Exception as e:
            print(f"[Config] Langfuse shutdown failed: {e}")


app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and query them using free cloud LLMs (Groq + Gemini embeddings).",
    version="1.0.0",
    lifespan=lifespan
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(3, ge=1, le=10, description="How many document chunks to retrieve")
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    steps_taken: list[str]
    context_sources: list[str]
    citations: list[dict]
    judge_log: list[str]
    trace_id: str | None = None

class HealthResponse(BaseModel):
    status: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API is running."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """
    Upload a .pdf, .docx, .txt or .md file.
    Extracts text, chunks it, embeds it, and stores in pgvector.
    Re-ingesting a file replaces its previously stored chunks.

    Sync on purpose: FastAPI runs it in a threadpool, so slow embedding
    (including free-tier rate-limit pauses) doesn't block the event loop.
    """
    # Validate file type
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported."
        )

    # Save uploaded file to a temp location so we can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run the ingestion pipeline
        text = load_document(tmp_path)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        setup_table()
        replaced = embed_and_store(chunks, source=filename)
    except Exception:
        # Log the full error server-side only — raw exception text can leak
        # connection strings or other internals to the client.
        print(f"[Ingest] Failed for '{filename}':\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed. Check the server logs for details."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "filename": filename,
        "characters_extracted": len(text),
        "chunks_stored": len(chunks),
        "chunks_replaced": replaced,
        "message": "Document ingested successfully."
    }


class IngestUrlRequest(BaseModel):
    url: str


@app.post("/ingest/url")
def ingest_url(request: IngestUrlRequest):
    """
    Fetch a web page, extract its text, chunk, embed and store it.
    The URL itself is the document's source; re-ingesting replaces its chunks.
    """
    url = request.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://.")

    try:
        title, text = load_url(url)
    except Exception:
        print(f"[Ingest/url] Failed to fetch '{url}':\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail="Could not fetch or read that URL.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="The page contained no readable text.")

    try:
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        setup_table()
        replaced = embed_and_store(chunks, source=url)
    except Exception:
        print(f"[Ingest/url] Failed for '{url}':\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed. Check the server logs for details."
        )

    return {
        "url": url,
        "title": title,
        "characters_extracted": len(text),
        "chunks_stored": len(chunks),
        "chunks_replaced": replaced,
        "message": "Page ingested successfully."
    }


@app.get("/documents")
def documents():
    """List ingested documents with chunk counts and ingest timestamps."""
    try:
        return {"documents": list_documents()}
    except Exception:
        print(f"[Documents] Failed to list documents:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Could not list documents. Check the server logs for details."
        )


@app.delete("/documents/{filename:path}")
def delete_document(filename: str):
    """Remove all stored chunks for one ingested document.
    The :path converter lets URL sources (which contain slashes) match too."""
    try:
        deleted = delete_by_source(filename)
    except Exception:
        print(f"[Documents] Failed to delete '{filename}':\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Could not delete the document. Check the server logs for details."
        )

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{filename}' found.")

    return {
        "filename": filename,
        "chunks_deleted": deleted,
        "message": "Document deleted."
    }


@app.get("/conversations")
def list_conversations():
    """Recent conversations (id, title, turn count, last activity)."""
    try:
        return {"conversations": conversations.list_conversations()}
    except Exception:
        print(f"[Conversations] Failed to list:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not list conversations.")


@app.get("/conversations/{session_id}")
def get_conversation(session_id: str):
    """Full transcript of one conversation, oldest turn first."""
    try:
        return {"session_id": session_id, "turns": conversations.get_transcript(session_id)}
    except Exception:
        print(f"[Conversations] Failed to load '{session_id}':\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not load the conversation.")


@app.delete("/conversations/{session_id}")
def delete_conversation(session_id: str):
    """Delete a conversation's transcript and its working memory."""
    try:
        deleted = conversations.delete_conversation(session_id)
        clear_session(session_id)
    except Exception:
        print(f"[Conversations] Failed to delete '{session_id}':\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not delete the conversation.")

    if deleted == 0:
        raise HTTPException(status_code=404, detail="No conversation with that id.")
    return {"session_id": session_id, "turns_deleted": deleted, "message": "Conversation deleted."}


class FeedbackRequest(BaseModel):
    trace_id: str
    helpful: bool
    comment: str | None = None


def record_feedback(trace_id: str, helpful: bool, comment: str | None) -> None:
    """Store user feedback as a boolean score on the answer's Langfuse trace."""
    from langfuse import get_client
    client = get_client()
    client.create_score(
        name="user-thumbs",
        value=1 if helpful else 0,
        trace_id=trace_id,
        data_type="BOOLEAN",
        comment=comment or None,
    )
    client.flush()


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Record 👍/👎 feedback for an answer, attached to its Langfuse trace."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        raise HTTPException(status_code=503, detail="Feedback requires Langfuse tracing to be configured.")

    try:
        record_feedback(request.trace_id, request.helpful, request.comment)
    except Exception:
        print(f"[Feedback] Failed to record:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not record feedback.")

    return {"message": "Feedback recorded."}


class HitlResolveRequest(BaseModel):
    answer: str
    ingest: bool = True


@app.get("/hitl")
def hitl_queue():
    """List questions the judge flagged for human review."""
    try:
        return {"items": hitl.list_pending()}
    except Exception:
        print(f"[HITL] Failed to list queue:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not load the review queue.")

@app.post("/hitl/{item_id}/resolve")
def hitl_resolve(item_id: int, request: HitlResolveRequest):
    """Record a human answer for a flagged question. By default the Q&A pair is
    also embedded into the vector store so future similar questions retrieve it."""
    answer = request.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")

    try:
        question = hitl.resolve(item_id, answer)
    except Exception:
        print(f"[HITL] Failed to resolve item {item_id}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not resolve the item.")

    if question is None:
        raise HTTPException(status_code=404, detail="No pending review item with that id.")

    ingested = False
    if request.ingest:
        try:
            setup_table()
            embed_and_store(
                [f"Question: {question}\nAnswer: {answer}"],
                source=f"hitl-{item_id}",
            )
            ingested = True
        except Exception:
            # The resolution is already saved; ingestion is best-effort.
            print(f"[HITL] Failed to ingest resolved answer {item_id}:\n{traceback.format_exc()}")

    return {"id": item_id, "question": question, "ingested": ingested, "message": "Resolved."}


@app.post("/hitl/{item_id}/dismiss")
def hitl_dismiss(item_id: int):
    """Dismiss a flagged question without answering it."""
    try:
        dismissed = hitl.dismiss(item_id)
    except Exception:
        print(f"[HITL] Failed to dismiss item {item_id}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not dismiss the item.")

    if not dismissed:
        raise HTTPException(status_code=404, detail="No pending review item with that id.")
    return {"id": item_id, "message": "Dismissed."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question. Returns the answer, steps taken, and accumulated context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        state = run_orchestrator(
            request.question,
            session_id=request.session_id,
            user_id=request.user_id,
            top_k=request.top_k,
        )
    except Exception:
        print(f"[Chat] Failed to answer question:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check the server logs for details."
        )

    return {
        "question": request.question,
        "answer": state.get("response", "No response generated."),
        "steps_taken": state.get("steps_taken", []),
        "context_sources": state.get("context", []),
        "citations": state.get("citations", []),
        "judge_log": state.get("judge_log", []),
        "trace_id": state.get("trace_id")
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Ask a question and stream the answer as Server-Sent Events.
    Events: {"type": "token"|"retry"|"done"|"error", ...} — "done" carries the
    same fields as the non-streaming /chat response.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_source():
        try:
            for event in stream_orchestrator(
                request.question,
                session_id=request.session_id,
                user_id=request.user_id,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            print(f"[Chat/stream] Failed to answer question:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate an answer. Check the server logs for details.'})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")