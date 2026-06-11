"""
RAG (Retrieval-Augmented Generation) Engine
Custom Domain ChatBot - Full Production Implementation

Tech Stack:
- LangChain for orchestration
- FAISS for vector storage
- OpenAI / HuggingFace for embeddings
- BM25 + Embeddings for Hybrid Search
- ConversationBufferWindowMemory for multi-turn reasoning
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Core LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Embeddings & Vector Store
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
)

# Hybrid Search (BM25 + Embeddings)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ─────────────────────────────────────────────
# 1. DOCUMENT INGESTION
# ─────────────────────────────────────────────

class DocumentIngester:
    """Loads PDFs, TXTs, DOCX from a folder and splits them into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def load_pdf(self, path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        return loader.load()

    def load_directory(self, dir_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        docs = []
        path = Path(dir_path)

        for pdf in path.glob("**/*.pdf"):
            docs.extend(PyPDFLoader(str(pdf)).load())
        for txt in path.glob("**/*.txt"):
            docs.extend(TextLoader(str(txt)).load())
        for docx in path.glob("**/*.docx"):
            docs.extend(UnstructuredWordDocumentLoader(str(docx)).load())

        print(f"[Ingester] Loaded {len(docs)} raw documents from {dir_path}")
        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        print(f"[Ingester] Split into {len(chunks)} chunks")
        return chunks


# ─────────────────────────────────────────────
# 2. VECTOR STORE (FAISS)
# ─────────────────────────────────────────────

class VectorStoreManager:
    """Manages FAISS vector index — build, save, load."""

    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore: Optional[FAISS] = None

    def build(self, chunks: List[Document]) -> FAISS:
        print("[VectorStore] Building FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.save()
        return self.vectorstore

    def save(self):
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            print(f"[VectorStore] Saved to {self.index_path}")

    def load(self) -> FAISS:
        print(f"[VectorStore] Loading from {self.index_path}")
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def add_documents(self, chunks: List[Document]):
        """Add new docs to existing index (incremental update)."""
        if self.vectorstore is None:
            raise ValueError("No vectorstore loaded. Call build() or load() first.")
        self.vectorstore.add_documents(chunks)
        self.save()
        print(f"[VectorStore] Added {len(chunks)} new chunks")


# ─────────────────────────────────────────────
# 3. HYBRID RETRIEVER (BM25 + Embeddings)
# ─────────────────────────────────────────────

def build_hybrid_retriever(
    vectorstore: FAISS,
    chunks: List[Document],
    top_k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> EnsembleRetriever:
    """
    Hybrid search combines:
    - Dense retrieval (semantic embeddings via FAISS)
    - Sparse retrieval (keyword matching via BM25)
    
    This handles both conceptual queries AND exact keyword matches.
    """
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance — reduces redundancy
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    sparse_retriever = BM25Retriever.from_documents(chunks, k=top_k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[bm25_weight, vector_weight],
    )
    print(f"[Retriever] Hybrid retriever ready (BM25={bm25_weight}, Dense={vector_weight})")
    return hybrid_retriever


# ─────────────────────────────────────────────
# 4. CUSTOM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for {domain_name}. 
You answer questions ONLY based on the provided context.
If the answer isn't in the context, say "I don't have that information in my knowledge base."

Rules:
- Be concise and precise
- Cite sources when possible (page numbers, document names)
- If a question requires multi-step reasoning, think step by step
- Do not make up facts

Context:
{context}
"""

CONDENSE_QUESTION_PROMPT = """Given a conversation history and a follow-up question, 
rephrase the follow-up into a standalone question that captures all necessary context.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""


# ─────────────────────────────────────────────
# 5. RAG CHAIN (MAIN ENGINE)
# ─────────────────────────────────────────────

class RAGChatbot:
    """
    Full RAG pipeline with:
    - Hybrid retrieval (BM25 + embeddings)
    - Multi-turn conversation memory
    - Custom domain prompting
    """

    def __init__(
        self,
        domain_name: str = "your company",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        memory_window: int = 5,
    ):
        self.domain_name = domain_name
        self.llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.retriever = None

    def setup(self, retriever):
        """Wire the retriever into the conversational chain."""
        self.retriever = retriever

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SYSTEM_PROMPT.replace("{domain_name}", self.domain_name)
            + "\nQuestion: {question}\nAnswer:",
        )

        condense_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=CONDENSE_QUESTION_PROMPT,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            condense_question_prompt=condense_prompt,
            return_source_documents=True,
            verbose=False,
        )
        print(f"[RAGChatbot] Ready for domain: {self.domain_name}")

    def chat(self, question: str) -> Dict:
        """Send a question and get a grounded answer with sources."""
        if not self.chain:
            raise RuntimeError("Call setup() before chat()")

        result = self.chain.invoke({"question": question})

        sources = []
        for doc in result.get("source_documents", []):
            src = {
                "content": doc.page_content[:300] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            }
            if src not in sources:
                sources.append(src)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources,
        }

    def clear_memory(self):
        self.memory.clear()
        print("[RAGChatbot] Memory cleared")


# ─────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def build_rag_pipeline(
    data_dir: str,
    index_path: str = "faiss_index",
    domain_name: str = "Your Company",
    force_rebuild: bool = False,
) -> RAGChatbot:
    """
    Full pipeline: Ingest → Embed → Index → Retrieve → Chat
    
    Args:
        data_dir: Folder with PDFs/TXTs/DOCXs
        index_path: Where to save/load FAISS index
        domain_name: Name shown in the system prompt
        force_rebuild: Re-embed even if index exists
    """
    ingester = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    vs_manager = VectorStoreManager(index_path=index_path)

    # Load or build the vector index
    if Path(index_path).exists() and not force_rebuild:
        print("[Pipeline] Loading existing FAISS index...")
        vectorstore = vs_manager.load()
        # For BM25 we still need the chunks — re-load and split
        raw_docs = ingester.load_directory(data_dir)
        chunks = ingester.split(raw_docs)
    else:
        print("[Pipeline] Building new FAISS index from scratch...")
        raw_docs = ingester.load_directory(data_dir)
        chunks = ingester.split(raw_docs)
        vectorstore = vs_manager.build(chunks)

    # Build hybrid retriever
    retriever = build_hybrid_retriever(vectorstore, chunks, top_k=5)

    # Create chatbot and wire everything up
    bot = RAGChatbot(domain_name=domain_name, model="gpt-4o-mini")
    bot.setup(retriever)

    return bot


# ─────────────────────────────────────────────
# 7. QUICK START (CLI)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  RAG Chatbot — Custom Domain Q&A")
    print("=" * 60)

    # Set your OpenAI key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    # Build pipeline (put your PDFs in ./data/)
    bot = build_rag_pipeline(
        data_dir="./data",
        index_path="./faiss_index",
        domain_name="My Company Knowledge Base",
    )

    print("\nChatbot ready! Type 'exit' to quit, 'clear' to reset memory.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "clear":
            bot.clear_memory()
            print("Memory cleared.\n")
            continue

        result = bot.chat(question)
        print(f"\nBot: {result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            for i, src in enumerate(result["sources"], 1):
                print(f"  {i}. {src['source']} (page {src['page']})")
        print()
