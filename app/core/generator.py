import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from core.graph import run_orchestrator

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")

chat_model = ChatGroq(model=CHAT_MODEL, temperature=0)


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def try_direct_answer(query: str) -> str:
    prompt = f"""You are a helpful assistant. Respond to the user's input.
If the input is asking about company policies, rules, benefits, guidelines, work procedures, or specific internal document details (even if asked generally, e.g., "What is the remote work policy?"), or any information you do not have definitive factual knowledge of, reply with exactly "NEED_CONTEXT".
Otherwise, if it is a greeting, casual conversation, general query (e.g., "Who wrote Romeo and Juliet?"), or simple help request, answer it directly.

INPUT: {query}
RESPONSE:"""

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()