"""
chat_cli.py  —  Interactive terminal chatbot

Run with:
  python backend/chat_cli.py

Commands during chat:
  clear  → reset conversation memory
  quit   → exit
  help   → show available commands
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from vectorstore import VectorStoreManager
from rag_chain import RAGChain


def print_banner():
    print("\n" + "=" * 55)
    print("  RAG Chatbot — TechCorp Knowledge Base")
    print("  Powered by Gemini 2.5 Flash + FAISS + BM25")
    print("=" * 55)
    print("  Type your question and press Enter.")
    print("  Commands: 'clear' | 'quit' | 'help'")
    print("=" * 55 + "\n")


def print_answer(result: dict):
    print(f"\n  Bot: {result['answer']}")
    print(f"\n  Sources:")
    for src in result["sources"]:
        print(f"    • {src['file']}  (Page {src['page']})")
    if result["condensed"] != result["question"]:
        print(f"  [Condensed to: {result['condensed']}]")
    print()


def main():
    print_banner()

    # ── Load pipeline ─────────────────────────────────────────
    print("  Loading knowledge base...")
    ingester    = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    chunks      = ingester.load_and_split("data/sample_docs")
    manager     = VectorStoreManager(index_path="faiss_index")
    vectorstore = manager.load()

    chain = RAGChain(
        chunks=chunks,
        vectorstore=vectorstore,
        domain="TechCorp Employee Handbook",
        memory_window=5,
        top_k=3,
    )

    print("  Knowledge base ready. Ask your first question!\n")
    print("  Note: 12 second delay between answers (free API tier)\n")

    # ── Chat loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "quit":
            print("\n  Goodbye!\n")
            break

        if user_input.lower() == "clear":
            chain.clear_memory()
            print("  Memory cleared. Starting fresh conversation.\n")
            continue

        if user_input.lower() == "help":
            print("\n  Commands:")
            print("    clear  → reset conversation memory")
            print("    quit   → exit the chatbot")
            print("    help   → show this message\n")
            continue

        # Ask the question
        print("  Thinking...", end="\r")
        try:
            result = chain.ask(user_input)
            print_answer(result)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("  Rate limit hit. Wait 1 minute and try again.\n")
            else:
                print(f"  Error: {msg[:100]}\n")


if __name__ == "__main__":
    main()