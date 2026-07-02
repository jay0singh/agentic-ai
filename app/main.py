import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pathlib

from core.ingestor import load_document
from core.chunker import chunk_text
from core.embedder import setup_table, embed_and_store, delete_by_source
from core.retriever import retrieve
from core.generator import run_orchestrator


def ingest(file_path: str):
    print(f"\n--- INGESTING: {file_path} ---")

    print("Step 1: Loading document...")
    text = load_document(file_path)
    print(f"  Extracted {len(text)} characters.")

    print("Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"  Created {len(chunks)} chunks.")

    print("Step 3: Setting up DB table...")
    setup_table()

    print("Step 4: Embedding and storing chunks...")
    source = pathlib.Path(file_path).name
    replaced = delete_by_source(source)
    if replaced:
        print(f"  Replacing {replaced} existing chunks for '{source}'.")
    embed_and_store(chunks, source=source)

    print("\nIngestion complete! Run 'chat' mode to start querying.")


def chat():
    print("\n--- RAG CHATBOT (Powered by Groq + Gemini embeddings) ---")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        print("Orchestrator analyzing and routing query...")
        state = run_orchestrator(query)

        # Display the steps taken by the graph
        steps = " -> ".join(state.get("steps_taken", []))
        print(f"[Graph Path]: {steps}\n")

        answer = state.get("response", "No response generated.")
        print(f"Bot: {answer}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m app.main ingest <file_path>")
        print("  python -m app.main chat\n")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a file path:")
            print("  python -m app.main ingest documents\\file.pdf")
            sys.exit(1)
        ingest(sys.argv[2])

    elif mode == "chat":
        chat()

    else:
        print(f"Unknown mode: '{mode}'. Use 'ingest' or 'chat'.")