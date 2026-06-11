"""
vectorstore.py  —  Phase 3: Embeddings & FAISS Vector Store

Responsibilities:
  1. Convert text chunks into vectors using Google Gemini embeddings
  2. Store those vectors in a FAISS index
  3. Save the index to disk so it persists across restarts
  4. Load the index back from disk
  5. Search the index with a query and return relevant chunks

Key concepts:
  - Embedding model  : converts text → list of numbers (vector)
  - FAISS index      : stores vectors, finds nearest neighbours fast
  - Cosine similarity: measures how "close" two vectors are (0=different, 1=identical)
  - Persistent index : saved to faiss_index/ folder — no re-embedding on restart
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    """
    Manages the FAISS vector index.

    Workflow:
      First run  → build(chunks) → saves index to disk
      Later runs → load()        → loads from disk (no re-embedding)
      Always     → search(query) → returns top-K relevant chunks

    Usage:
        manager = VectorStoreManager()
        manager.build(chunks)                        # first time only
        results = manager.search("refund policy")    # any time
    """

    def __init__(self, index_path: str = "faiss_index"):
        """
        index_path : folder where the FAISS index is saved.
                     Will be created automatically on first build.
        """
        self.index_path  = index_path
        self.vectorstore: Optional[FAISS] = None

        # Google's text-embedding-004 model
        # - Free tier: 1500 requests/minute
        # - Produces 768-dimensional vectors
        # - Excellent quality for semantic search
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # ── Build ─────────────────────────────────────────────────

    def build(self, chunks: List[Document]) -> FAISS:
        """
        Embeds all chunks and builds a new FAISS index from scratch.

        What happens internally:
          1. Each chunk's text is sent to Google's embedding API
          2. The API returns a 768-dimensional vector for each chunk
          3. FAISS stores all vectors in an efficient index structure
          4. The index + chunk texts are saved to disk

        This is called ONCE when you first set up the system.
        After that, use load() to avoid re-embedding.
        """
        print(f"\nBuilding FAISS index from {len(chunks)} chunks...")
        print("Sending chunks to Google embedding API...")

        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        self._save()
        print(f"Index built and saved to: {self.index_path}/")
        return self.vectorstore

    # ── Save ──────────────────────────────────────────────────

    def _save(self):
        """Saves the FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)

    # ── Load ──────────────────────────────────────────────────

    def load(self) -> FAISS:
        """
        Loads a previously saved FAISS index from disk.
        Much faster than rebuilding — no API calls needed.

        Call this on every restart after the first build.
        """
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"No index found at '{self.index_path}'. "
                f"Run build() first."
            )

        print(f"Loading FAISS index from: {self.index_path}/")
        self.vectorstore = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Index loaded successfully.")
        return self.vectorstore

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Finds the top_k most relevant chunks for a query.

        What happens internally:
          1. Your query text is embedded into a 768-dim vector
          2. FAISS computes cosine similarity vs all stored vectors
          3. Returns the top_k closest chunks

        This is pure math — no LLM involved at this stage.
        The LLM comes in Phase 5 to read these chunks and answer.
        """
        if not self.vectorstore:
            raise RuntimeError("No vectorstore loaded. Call build() or load() first.")

        results = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
        )
        return results

    def search_with_scores(self, query: str, top_k: int = 3):
        """
        Same as search() but also returns similarity scores.
        Score closer to 0 = more similar (L2 distance).
        Useful for debugging retrieval quality.
        """
        if not self.vectorstore:
            raise RuntimeError("No vectorstore loaded. Call build() or load() first.")

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    # ── Add new documents ─────────────────────────────────────

    def add_documents(self, new_chunks: List[Document]):
        """
        Adds new chunks to an existing index without rebuilding.
        This is incremental indexing — critical for production
        systems where documents are added continuously.
        """
        if not self.vectorstore:
            raise RuntimeError("Load or build an index first.")

        self.vectorstore.add_documents(new_chunks)
        self._save()
        print(f"Added {len(new_chunks)} new chunks to index.")


# ── Test runner ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from ingester import DocumentIngester

    # ── Step 1: Load and chunk the PDF ────────────────────────
    ingester = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    chunks   = ingester.load_and_split("data/sample_docs")

    # ── Step 2: Build the FAISS index ─────────────────────────
    manager = VectorStoreManager(index_path="faiss_index")
    manager.build(chunks)

    # ── Step 3: Test semantic search ──────────────────────────
    test_queries = [
        "What is the refund policy?",
        "How many days of annual leave do employees get?",
        "What are the password requirements?",
    ]

    print("\n" + "=" * 50)
    print("SEMANTIC SEARCH TEST")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery : '{query}'")
        print("-" * 40)

        results = manager.search_with_scores(query, top_k=2)

        for rank, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page   = doc.metadata.get("page", "N/A")
            print(f"  Rank {rank} | Score: {score:.4f} | {source} p.{page}")
            print(f"  Preview: {doc.page_content[:150].strip()}...")
            print()

    # ── Step 4: Show what the index folder contains ───────────
    print("=" * 50)
    print(f"Index saved to: faiss_index/")
    for f in Path("faiss_index").iterdir():
        size = f.stat().st_size
        print(f"  {f.name:<30} {size:>8,} bytes")

    print("\nPhase 3 complete. 6 chunks are now searchable vectors.")