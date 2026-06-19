import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic AI")

    if api_online():
        st.success("API online", icon="✅")
    else:
        st.error("API offline — start the FastAPI server", icon="🔴")

    st.divider()

    st.subheader("📄 Ingest Document")
    uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                r = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60
                )
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"✅ {d['chunks_stored']} chunks stored from **{d['filename']}**")
                else:
                    st.error(r.json().get("detail", r.text))
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps_taken"):
            render_steps(msg["steps_taken"])
            if "hitl" in msg["steps_taken"]:
                st.warning("Could not produce a satisfactory answer after multiple retries.")
        render_extras(msg.get("context_sources", []), msg.get("judge_log", []))

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": prompt},
                    timeout=600
                )
                if r.status_code == 200:
                    data = r.json()
                    answer = data["answer"]
                    steps = data["steps_taken"]
                    sources = data.get("context_sources", [])
                    judge_log = data.get("judge_log", [])

                    st.markdown(answer)
                    render_steps(steps)

                    if "hitl" in steps:
                        st.warning("Could not produce a satisfactory answer after multiple retries.")

                    render_extras(sources, judge_log)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "steps_taken": steps,
                        "context_sources": sources,
                        "judge_log": judge_log
                    })
                else:
                    err = r.json().get("detail", r.text)
                    st.error(f"Error: {err}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {err}",
                        "steps_taken": []
                    })

            except requests.exceptions.Timeout:
                st.error("Request timed out. The pipeline may still be running — try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")
