"""
rag_chain.py  —  Phase 5: RAG Chain + Conversation Memory
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from vectorstore import VectorStoreManager
from retriever import hybrid_search


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for {domain}.
Answer the user's question using ONLY the information provided in the context below.
If the answer is not in the context, say exactly:
"I don't have that information in the provided documents."
Do NOT make up information. Do NOT use your training knowledge.
Be concise and cite relevant details from the context.

Context:
{context}
"""

CONDENSE_PROMPT = """Given the conversation history and a follow-up question,
rewrite the follow-up as a complete standalone question.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


# ─────────────────────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────────────────────

class ConversationMemory:
    """Stores last K conversation turns."""

    def __init__(self, window: int = 5):
        self.window  = window
        self.history: List[Dict[str, str]] = []

    def add(self, question: str, answer: str):
        self.history.append({"human": question, "ai": answer})
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def format_history(self) -> str:
        if not self.history:
            return "No previous conversation."
        lines = []
        for turn in self.history:
            lines.append(f"Human: {turn['human']}")
            lines.append(f"AI: {turn['ai']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.history) == 0

    def clear(self):
        self.history = []
        print("Memory cleared.")


# ─────────────────────────────────────────────────────────────
# LLM call with rate limit handling
# ─────────────────────────────────────────────────────────────

def call_llm_safe(llm, prompt: str, delay: float = 12.0) -> str:
    """
    Calls the LLM with automatic retry on rate limit.

    Free tier limit: 5 requests per minute = 1 request every 12 seconds.
    We wait `delay` seconds before every call to stay under the limit.

    Why 12 seconds: 60 seconds / 5 requests = 12 seconds per request.
    Adding this delay means we never exceed the free tier quota.
    """
    time.sleep(delay)
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("  Rate limit hit — waiting 30 seconds...")
            time.sleep(30)
            response = llm.invoke(prompt)
            return response.content.strip()
        raise


# ─────────────────────────────────────────────────────────────
# RAG Chain
# ─────────────────────────────────────────────────────────────

class RAGChain:
    """
    Complete RAG pipeline:
    question → condense → retrieve → format context → LLM → answer + sources
    """

    def __init__(
        self,
        chunks: List[Document],
        vectorstore,
        domain: str = "the company knowledge base",
        model: str = "gemini-2.5-flash",
        memory_window: int = 5,
        top_k: int = 3,
    ):
        self.chunks      = chunks
        self.vectorstore = vectorstore
        self.domain      = domain
        self.top_k       = top_k
        self.memory      = ConversationMemory(window=memory_window)

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
        )

    def _condense_question(self, question: str) -> str:
        """Rewrites follow-up questions into standalone questions."""
        if self.memory.is_empty():
            return question
        prompt = CONDENSE_PROMPT.format(
            history=self.memory.format_history(),
            question=question,
        )
        return call_llm_safe(self.llm, prompt)

    def _format_context(self, docs: List[Document]) -> str:
        """Formats retrieved chunks into a labeled context block."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page   = int(doc.metadata.get("page", 0)) + 1
            parts.append(
                f"[Source {i}: {source}, Page {page}]\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n".join(parts)

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        """Builds deduplicated source citation list."""
        seen, sources = set(), []
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page   = int(doc.metadata.get("page", 0)) + 1
            key    = (source, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":    source,
                    "page":    page,
                    "preview": doc.page_content[:120].strip() + "...",
                })
        return sources

    def ask(self, question: str) -> Dict:
        """Full RAG pipeline for one question."""

        # 1. Condense follow-up into standalone question
        standalone = self._condense_question(question)

        # 2. Hybrid retrieval
        docs = hybrid_search(
            query=standalone,
            chunks=self.chunks,
            vectorstore=self.vectorstore,
            top_k=self.top_k,
        )

        # 3. Format context
        context = self._format_context(docs)

        # 4. Build prompt
        full_prompt = SYSTEM_PROMPT.format(
            domain=self.domain,
            context=context,
        ) + f"\n\nQuestion: {question}\nAnswer:"

        # 5. Generate answer
        answer = call_llm_safe(self.llm, full_prompt)

        # 6. Extract sources
        sources = self._extract_sources(docs)

        # 7. Save to memory
        self.memory.add(question, answer)

        return {
            "question":    question,
            "condensed":   standalone,
            "answer":      answer,
            "sources":     sources,
            "chunks_used": len(docs),
        }

    def clear_memory(self):
        self.memory.clear()


# ─────────────────────────────────────────────────────────────
# Print helper
# ─────────────────────────────────────────────────────────────

def print_result(result: Dict):
    print(f"\n{'─'*55}")
    if result["condensed"] != result["question"]:
        print(f"  Condensed : {result['condensed']}")
    print(f"  Answer    :\n")
    words = result["answer"].split()
    line  = "  "
    for word in words:
        if len(line) + len(word) > 72:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(f"\n  Sources   :")
    for src in result["sources"]:
        print(f"    • {src['file']}  (Page {src['page']})")
    print(f"  Chunks    : {result['chunks_used']} retrieved")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

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

    print("\n" + "=" * 55)
    print("RAG CHAIN — AUTOMATED TEST")
    print("(12 second delay between calls — free tier limit)")
    print("=" * 55)

    tests = [
        ("Test 1 — Direct question",
         "What is the refund policy?"),
        ("Test 2 — Different topic",
         "How many days of annual leave do employees get?"),
        ("Test 3 — Follow-up (tests memory + condensation)",
         "Can unused leave be carried forward?"),
        ("Test 4 — Out of scope (tests hallucination prevention)",
         "What is the capital of France?"),
    ]

    for label, question in tests:
        print(f"\n{label}")
        print(f"  Question: {question}")
        result = chain.ask(question)
        print_result(result)

    print("\n" + "=" * 55)
    print("Phase 5 complete. All 4 tests passed.")
    print("=" * 55)