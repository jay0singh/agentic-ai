---
title: Agentic RAG Demo
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Agentic RAG Demo

A LangGraph-powered RAG system with multi-tool routing, an LLM-as-judge loop,
human-in-the-loop review, streaming answers, and persistent conversations —
running entirely on free-tier cloud APIs (Groq, Google Gemini, Neon Postgres).

Source code, docs and tests: https://github.com/jay0singh/agentic-ai

## Required Space secrets

Set these under **Settings → Variables and secrets**:

`GROQ_API_KEY`, `GOOGLE_API_KEY`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`,
`DB_NAME`, `DB_PORT` (usually 5432), `DB_SSLMODE=require`, `DB_TABLE=documents`.
Optional: `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_BASE_URL`, `CHAT_MODEL`, `JUDGE_MODEL`.
