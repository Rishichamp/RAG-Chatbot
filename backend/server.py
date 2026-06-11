"""
server.py  —  Phase 6: FastAPI REST Server

Exposes the RAG chatbot as a web API.

Endpoints:
  GET  /              → health check
  GET  /status        → index and document info
  POST /chat          → ask a question
  POST /upload        → upload a new document
  DELETE /memory      → clear conversation history

Run with:
  uvicorn backend.server:app --reload --port 8000
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from vectorstore import VectorStoreManager
from rag_chain import RAGChain


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="Custom domain chatbot powered by Gemini + FAISS + BM25",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Global state — loaded once on startup
# ─────────────────────────────────────────────────────────────

DATA_DIR   = "data/sample_docs"
INDEX_DIR  = "faiss_index"
chain: Optional[RAGChain] = None


def load_pipeline():
    """Loads the full RAG pipeline into memory."""
    global chain
    ingester    = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    chunks      = ingester.load_and_split(DATA_DIR)
    manager     = VectorStoreManager(index_path=INDEX_DIR)
    vectorstore = manager.load()
    chain = RAGChain(
        chunks=chunks,
        vectorstore=vectorstore,
        domain="TechCorp Employee Handbook",
        memory_window=5,
        top_k=3,
    )
    return len(chunks)


@app.on_event("startup")
async def startup_event():
    """Loads the pipeline when the server starts."""
    try:
        n = load_pipeline()
        print(f"RAG pipeline loaded. {n} chunks ready.")
    except Exception as e:
        print(f"Pipeline load failed: {e}")


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list
    chunks_used: int


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "running",
        "message": "RAG Chatbot API is live.",
        "docs":    "Visit /docs for interactive API documentation.",
    }


@app.get("/status")
async def status():
    """Returns info about the loaded index and documents."""
    data_files = list(Path(DATA_DIR).rglob("*.*"))
    index_ok   = Path(INDEX_DIR).exists()
    return {
        "pipeline_loaded": chain is not None,
        "index_exists":    index_ok,
        "documents":       [f.name for f in data_files],
        "document_count":  len(data_files),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask the chatbot a question.

    The pipeline runs:
      question → condense → hybrid retrieve → LLM → answer + sources

    Rate limit note: free tier allows 5 requests/minute.
    The chain adds a 12-second delay automatically.
    """
    if not chain:
        raise HTTPException(503, "Pipeline not loaded. Check server logs.")

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    result = chain.ask(req.question)

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    The pipeline rebuilds the index automatically after upload.
    """
    allowed = {".pdf", ".txt", ".docx"}
    suffix  = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use PDF, TXT, or DOCX.")

    dest = Path(DATA_DIR) / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild pipeline with new document
    try:
        ingester    = DocumentIngester(chunk_size=1000, chunk_overlap=200)
        chunks      = ingester.load_and_split(DATA_DIR)
        manager     = VectorStoreManager(index_path=INDEX_DIR)
        vectorstore = manager.build(chunks)
        global chain
        chain = RAGChain(
            chunks=chunks,
            vectorstore=vectorstore,
            domain="TechCorp Employee Handbook",
            memory_window=5,
            top_k=3,
        )
        return {
            "message":      f"Uploaded and indexed: {file.filename}",
            "chunks_added": len(chunks),
        }
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {str(e)}")


@app.delete("/memory")
async def clear_memory():
    """Clears conversation history — starts a fresh session."""
    if chain:
        chain.clear_memory()
    return {"message": "Conversation memory cleared."}