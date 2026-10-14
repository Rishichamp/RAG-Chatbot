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

import hmac
import os
import sys
import threading
import uuid
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from vectorstore import VectorStoreManager
from rag_chain import ConversationMemory, RAGChain


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the pipeline when the server starts."""
    try:
        n = load_pipeline()
        print(f"RAG pipeline loaded. {n} chunks ready.")
    except Exception as e:
        print(f"Pipeline load failed: {e}")
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Custom domain chatbot powered by Gemini + FAISS + BM25",
    version="1.0.0",
    lifespan=lifespan,
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
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SESSIONS = 500
# If set, POST /upload requires the header `X-Upload-Token: <value>`.
# Leave unset when running locally; set it on any public deployment, or
# anyone could add documents (and steer the bot's answers).
UPLOAD_TOKEN = os.getenv("UPLOAD_TOKEN", "")
chain: Optional[RAGChain] = None

# One conversation memory per client session (newest last), so visitors
# never see each other's history. Bounded: the oldest sessions are dropped.
_sessions: "OrderedDict[str, ConversationMemory]" = OrderedDict()
_sessions_lock = threading.Lock()
_index_lock = threading.Lock()


def _memory_for(session_id: str) -> ConversationMemory:
    with _sessions_lock:
        mem = _sessions.pop(session_id, None) or ConversationMemory(window=5)
        _sessions[session_id] = mem
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return mem


def _rebuild_pipeline() -> int:
    """Re-index every document and swap in a new chain (one rebuild at a time)."""
    global chain
    with _index_lock:
        ingester    = DocumentIngester(chunk_size=1000, chunk_overlap=200)
        chunks      = ingester.load_and_split(DATA_DIR)
        manager     = VectorStoreManager(index_path=INDEX_DIR)
        vectorstore = manager.build(chunks)
        chain = RAGChain(
            chunks=chunks,
            vectorstore=vectorstore,
            domain="TechCorp Employee Handbook",
            memory_window=5,
            top_k=3,
        )
        return len(chunks)


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


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(max_length=2000)
    # Returned by the first /chat call; send it back to continue that conversation.
    session_id: Optional[str] = Field(default=None, max_length=64)

class ChatResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list
    chunks_used: int
    session_id:  str


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


# Plain `def` (not async): the chain blocks for many seconds (LLM calls and a
# rate-limit sleep), and FastAPI runs sync endpoints in a thread pool. As an
# `async def` it froze the whole server for every other request meanwhile.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
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

    session_id = req.session_id or uuid.uuid4().hex
    result = chain.ask(req.question, memory=_memory_for(session_id))

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        session_id=session_id,
    )


@app.post("/upload")
def upload_document(file: UploadFile = File(...), x_upload_token: str = Header(default="")):
    """
    Upload a PDF or TXT document.
    The pipeline rebuilds the index automatically after upload.
    """
    if UPLOAD_TOKEN and not hmac.compare_digest(x_upload_token, UPLOAD_TOKEN):
        raise HTTPException(401, "Uploads need a valid X-Upload-Token header.")

    allowed = {".pdf", ".txt"}
    # Keep only the base name: a filename like "../../x.pdf" would otherwise
    # be written outside DATA_DIR.
    name    = Path(file.filename or "").name
    suffix  = Path(name).suffix.lower()

    if not name or suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix or '(none)'}. Use PDF or TXT.")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if suffix == ".pdf" and not contents.startswith(b"%PDF-"):
        raise HTTPException(400, "That file is not a valid PDF.")
    if suffix == ".txt":
        try:
            contents.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Text files must be UTF-8.")

    # Never overwrite an existing document (e.g. the bundled handbook).
    dest = Path(DATA_DIR) / name
    if dest.exists():
        dest = Path(DATA_DIR) / f"{Path(name).stem}-{uuid.uuid4().hex[:8]}{suffix}"
    with open(dest, "wb") as f:
        f.write(contents)

    # Rebuild pipeline with new document
    try:
        n_chunks = _rebuild_pipeline()
        return {
            "message":      f"Uploaded and indexed: {dest.name}",
            "chunks_added": n_chunks,
        }
    except Exception as e:
        print(f"Indexing failed: {e}")
        raise HTTPException(500, "Indexing failed. Check the server logs.")


@app.delete("/memory")
def clear_memory(session_id: Optional[str] = None):
    """Clears one conversation's history (pass the session_id from /chat)."""
    if session_id:
        with _sessions_lock:
            _sessions.pop(session_id, None)
    return {"message": "Conversation memory cleared."}