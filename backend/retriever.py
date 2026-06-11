"""
retriever.py  —  Phase 4: Hybrid Retrieval (BM25 + FAISS)

Uses a custom hybrid merger instead of EnsembleRetriever
to avoid LangChain version conflicts. Implements the same
Reciprocal Rank Fusion algorithm internally.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from vectorstore import VectorStoreManager


# ─────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion — the merging algorithm
# ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    weights: List[float],
    k: int = 60,
) -> List[Document]:
    """
    Merges multiple ranked result lists into one using RRF.

    Formula for each document:
      score += weight × (1 / (rank + k))

    k=60 is standard — dampens the impact of rank differences.
    Higher weight = that retriever's rankings matter more.

    Returns deduplicated list sorted by combined score.
    """
    scores: Dict[str, float]    = {}
    doc_map: Dict[str, Document] = {}

    for results, weight in zip(results_list, weights):
        for rank, doc in enumerate(results, start=1):
            # Use page_content as unique key for deduplication
            key = doc.page_content[:100]
            if key not in scores:
                scores[key]  = 0.0
                doc_map[key] = doc
            scores[key] += weight * (1.0 / (rank + k))

    # Sort by score descending — highest score = most relevant
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ─────────────────────────────────────────────────────────────
# Individual retriever builders
# ─────────────────────────────────────────────────────────────

def build_bm25_retriever(chunks: List[Document], top_k: int = 3) -> BM25Retriever:
    """
    Keyword-based retriever. No API calls needed.
    Scores chunks by word frequency and rarity (IDF).
    Best for: exact terms, product codes, names, numbers.
    """
    return BM25Retriever.from_documents(chunks, k=top_k)


def build_dense_retriever(vectorstore, top_k: int = 3):
    """
    Semantic retriever using FAISS + MMR.
    Best for: meaning-based queries, paraphrases, concepts.
    MMR ensures diverse results — no near-duplicate chunks.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": min(10, top_k * 3),
        },
    )


# ─────────────────────────────────────────────────────────────
# Hybrid search function
# ─────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    chunks: List[Document],
    vectorstore,
    top_k: int = 3,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Document]:
    """
    Main hybrid search function.

    Steps:
      1. BM25 retriever finds top_k chunks by keyword match
      2. FAISS retriever finds top_k chunks by semantic similarity
      3. RRF merges both ranked lists with weights
      4. Returns top_k deduplicated results

    dense_weight=0.6, bm25_weight=0.4 — semantic search
    trusted slightly more, but keywords still matter.
    """
    bm25  = build_bm25_retriever(chunks, top_k)
    dense = build_dense_retriever(vectorstore, top_k)

    bm25_results  = bm25.invoke(query)
    dense_results = dense.invoke(query)

    merged = reciprocal_rank_fusion(
        results_list=[bm25_results, dense_results],
        weights=[bm25_weight, dense_weight],
    )
    return merged[:top_k]


# ─────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────

def compare_retrievers(
    query: str,
    chunks: List[Document],
    vectorstore,
    top_k: int = 2,
):
    print(f"\nQuery: '{query}'")
    print("─" * 55)

    bm25  = build_bm25_retriever(chunks, top_k)
    dense = build_dense_retriever(vectorstore, top_k)

    bm25_results  = bm25.invoke(query)
    dense_results = dense.invoke(query)
    hybrid_results = hybrid_search(query, chunks, vectorstore, top_k)

    def show(label, results):
        print(f"\n  {label}:")
        for i, doc in enumerate(results, 1):
            src  = os.path.basename(doc.metadata.get("source", "?"))
            page = doc.metadata.get("page", "?")
            prev = doc.page_content[:100].replace("\n", " ").strip()
            print(f"    {i}. [p.{page}] {prev}...")

    show("BM25  (keyword)", bm25_results)
    show("FAISS (semantic)", dense_results)
    show("Hybrid (RRF merged)", hybrid_results)
    print()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ingester    = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    chunks      = ingester.load_and_split("data/sample_docs")

    manager     = VectorStoreManager(index_path="faiss_index")
    vectorstore = manager.load()

    print("\n" + "=" * 55)
    print("HYBRID RETRIEVAL — COMPARISON TEST")
    print("=" * 55)

    queries = [
        "What are the password requirements?",
        "How do I get a refund?",
        "What is the parental leave policy?",
    ]

    for q in queries:
        compare_retrievers(q, chunks, vectorstore, top_k=2)

    print("=" * 55)
    print("Hybrid retriever working.")
    print("  Algorithm    : Reciprocal Rank Fusion (custom)")
    print("  Dense weight : 0.6  (FAISS + MMR semantic)")
    print("  BM25  weight : 0.4  (keyword matching)")
    print("  Top-K        : 3 results per query")
    print("=" * 55)
    print("\nPhase 4 complete. Ready for Phase 5.")