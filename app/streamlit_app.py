import json
import os
import uuid
from urllib.parse import quote

import streamlit as st
import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

STEP_ICONS = {
    "vector_search": "🔍",
    "web_search":    "🌐",
    "github_read":   "🐙",
    "generate":      "✨",
    "judge":         "⚖️",
    "rewrite":       "🔄",
    "hitl":          "🚨",
}

st.set_page_config(
    page_title="Agentic AI",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.step-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin: 2px 3px 6px 0;
    background: #f0f2f6;
    color: #4b4b4b;
}
.step-rewrite { background: #fff8e1; color: #e65100; }
.step-hitl    { background: #ffebee; color: #c62828; }
</style>
""", unsafe_allow_html=True)


def render_steps(steps: list):
    badges = []
    for step in steps:
        icon = STEP_ICONS.get(step, "•")
        css = "step-badge"
        if step == "hitl":
            css += " step-hitl"
        elif step == "rewrite":
            css += " step-rewrite"
        badges.append(f'<span class="{css}">{icon} {step}</span>')
    st.markdown("".join(badges), unsafe_allow_html=True)


def render_citations(citations: list):
    """One caption line naming each cited document and how many chunks it contributed."""
    if not citations:
        return
    counts: dict[str, int] = {}
    for c in citations:
        src = c.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    parts = [f"{src} ({n} chunk{'s' if n > 1 else ''})" for src, n in counts.items()]
    st.caption("📄 Cited: " + " · ".join(parts))


def stream_chat_events(payload: dict):
    """Yield parsed SSE events from the streaming chat endpoint."""
    with requests.post(f"{API_BASE}/chat/stream", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: "):])


def api_online() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except Exception:
        return False


def render_extras(sources: list, judge_log: list):
    if judge_log:
        with st.expander("⚖️ Judge reasoning"):
            for i, entry in enumerate(judge_log, 1):
                if entry.startswith("RETRY"):
                    st.markdown(f"**Round {i}:** 🔄 {entry}")
                else:
                    st.markdown(f"**Round {i}:** ✅ {entry}")

    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}**")
                st.text(src)
                if i < len(sources):
                    st.divider()


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # Groups this conversation's traces in Langfuse's Sessions view
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic AI")

    if api_online():
        st.success("API online", icon="✅")
    else:
        st.error("API offline — start the FastAPI server", icon="🔴")

    st.divider()

    st.subheader("📄 Ingest Document")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                r = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=600
                )
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"✅ {d['chunks_stored']} chunks stored from **{d['filename']}**")
                else:
                    st.error(r.json().get("detail", r.text))
            except Exception as e:
                st.error(f"Connection error: {e}")

    url_to_ingest = st.text_input("Or ingest a web page", placeholder="https://example.com/article")
    if url_to_ingest and st.button("Ingest URL", use_container_width=True):
        with st.spinner("Fetching and ingesting…"):
            try:
                r = requests.post(f"{API_BASE}/ingest/url", json={"url": url_to_ingest}, timeout=600)
                if r.status_code == 200:
                    d = r.json()
                    label = d.get("title") or d["url"]
                    st.success(f"✅ {d['chunks_stored']} chunks stored from **{label}**")
                else:
                    st.error(r.json().get("detail", r.text))
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    st.subheader("🗂️ Ingested Documents")
    try:
        doc_list = requests.get(f"{API_BASE}/documents", timeout=5).json().get("documents", [])
    except Exception:
        doc_list = None

    if doc_list is None:
        st.caption("Could not load the document list.")
    elif not doc_list:
        st.caption("No documents ingested yet.")
    else:
        for doc in doc_list:
            name_col, del_col = st.columns([5, 1])
            date = (doc.get("ingested_at") or "")[:10]
            detail = f"{doc['chunks']} chunks" + (f" · {date}" if date else "")
            name_col.markdown(f"**{doc['source']}**  \n<small>{detail}</small>", unsafe_allow_html=True)
            if del_col.button("🗑️", key=f"del-{doc['source']}", help=f"Delete {doc['source']}"):
                try:
                    r = requests.delete(f"{API_BASE}/documents/{quote(doc['source'], safe='')}", timeout=30)
                    if r.status_code != 200:
                        st.error(r.json().get("detail", r.text))
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ── HITL review queue ──────────────────────────────────────────────────────────
try:
    hitl_items = requests.get(f"{API_BASE}/hitl", timeout=5).json().get("items", [])
except Exception:
    hitl_items = []

if hitl_items:
    n = len(hitl_items)
    with st.expander(f"🚨 Review queue — {n} unanswered question{'s' if n > 1 else ''}"):
        for item in hitl_items:
            st.markdown(f"**Q: {item['question']}**")
            if item.get("judge_reason"):
                st.caption(f"⚖️ {item['judge_reason']}")
            if item.get("attempted_answer"):
                st.caption(f"Last attempt: {item['attempted_answer'][:200]}")
            answer = st.text_area(
                "Answer", key=f"hitl-answer-{item['id']}",
                label_visibility="collapsed", placeholder="Write the correct answer…"
            )
            resolve_col, dismiss_col = st.columns(2)
            if resolve_col.button("✅ Resolve & teach", key=f"hitl-resolve-{item['id']}",
                                  help="Saves the answer and adds it to the knowledge base"):
                if not answer.strip():
                    st.warning("Write an answer first.")
                else:
                    try:
                        r = requests.post(f"{API_BASE}/hitl/{item['id']}/resolve",
                                          json={"answer": answer}, timeout=120)
                        if r.status_code == 200:
                            st.rerun()
                        st.error(r.json().get("detail", r.text))
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            if dismiss_col.button("🗑️ Dismiss", key=f"hitl-dismiss-{item['id']}"):
                try:
                    requests.post(f"{API_BASE}/hitl/{item['id']}/dismiss", timeout=30)
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection error: {e}")
            st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps_taken"):
            render_steps(msg["steps_taken"])
            if "hitl" in msg["steps_taken"]:
                st.warning("Could not produce a satisfactory answer after multiple retries.")
        render_citations(msg.get("citations", []))
        render_extras(msg.get("context_sources", []), msg.get("judge_log", []))

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*Routing and retrieving…*")
        streamed_text = ""
        final = None
        error = None

        try:
            for event in stream_chat_events(
                {"question": prompt, "session_id": st.session_state.session_id}
            ):
                if event["type"] == "token":
                    streamed_text += event["content"]
                    placeholder.markdown(streamed_text + "▌")
                elif event["type"] == "retry":
                    streamed_text = ""
                    placeholder.markdown("*⚖️ Judge requested a better answer — retrying…*")
                elif event["type"] == "done":
                    final = event
                elif event["type"] == "error":
                    error = event["message"]
        except requests.exceptions.Timeout:
            error = "Request timed out. The pipeline may still be running — try again."
        except Exception as e:
            error = f"Connection error: {e}"

        if error or final is None:
            message = f"Error: {error or 'The stream ended unexpectedly.'}"
            placeholder.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "steps_taken": []}
            )
        else:
            answer = final.get("answer") or streamed_text
            steps = final.get("steps_taken", [])
            sources = final.get("context_sources", [])
            citations = final.get("citations", [])
            judge_log = final.get("judge_log", [])

            placeholder.markdown(answer)
            render_steps(steps)

            if "hitl" in steps:
                st.warning("Could not produce a satisfactory answer after multiple retries.")

            render_citations(citations)
            render_extras(sources, judge_log)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "steps_taken": steps,
                "context_sources": sources,
                "citations": citations,
                "judge_log": judge_log
            })
