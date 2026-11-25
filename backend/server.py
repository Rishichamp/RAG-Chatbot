"""
server.py  —  FastAPI server: one document per chat

Every visitor can open chats. Each chat holds exactly ONE uploaded
document and its own search index (BM25 + FAISS), and answers questions
only from that document. To ask about another document, start a new chat.

Chats are kept apart structurally: each chat owns its chunks, its FAISS
index and its conversation memory, and a chat is only reachable by its
random, unguessable id. Nothing is shared between chats or visitors.

Endpoints:
  GET    /                         → web chat page
  GET    /health                   → health check
  GET    /status                   → server limits and active chat count
  POST   /chats                    → start a new chat
  GET    /chats/{chat_id}          → chat info and history
  POST   /chats/{chat_id}/document → upload the chat's one document (PDF/TXT)
  POST   /chats/{chat_id}/messages → ask a question about that document
  DELETE /chats/{chat_id}          → delete a chat and its index

Run with:
  uvicorn backend.server:app --reload --port 8000

Chats live in memory: they are lost when the server restarts.
"""

import os
import secrets
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, Path as PathParam, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from ingester import DocumentIngester
from rag_chain import ConversationMemory


# ─────────────────────────────────────────────────────────────
# Settings (all overridable with environment variables)
# ─────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_UPLOAD_BYTES     = _env_int("MAX_UPLOAD_MB", 10) * 1024 * 1024
MAX_CHUNKS           = _env_int("MAX_CHUNKS_PER_DOCUMENT", 200)
MAX_CHATS            = _env_int("MAX_CHATS", 100)            # oldest chats are dropped
CHAT_TTL_SECONDS     = _env_int("CHAT_TTL_MINUTES", 120) * 60  # idle chats expire
UPLOADS_PER_HOUR_IP  = _env_int("UPLOADS_PER_HOUR_PER_IP", 10)
MESSAGES_PER_HOUR_IP = _env_int("MESSAGES_PER_HOUR_PER_IP", 60)
ALLOWED_SUFFIXES     = {".pdf", ".txt"}
STATIC_DIR           = Path(__file__).parent / "static"

CHAT_ID_PATTERN = r"^[A-Za-z0-9_-]{20,64}$"


# ─────────────────────────────────────────────────────────────
# Chat state
# ─────────────────────────────────────────────────────────────

@dataclass
class Chat:
    id: str
    memory: ConversationMemory = field(default_factory=lambda: ConversationMemory(window=5))
    document: Optional[str] = None   # file name, set once
    chunks: List[Any] = field(default_factory=list)
    chain: Any = None                # RAGChain over this chat's document only
    indexing: bool = False           # an upload is being processed
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def info(self) -> dict:
        return {
            "chat_id":  self.id,
            "document": self.document,
            "chunks":   len(self.chunks),
            "history":  list(self.memory.history),
        }


class ChatStore:
    """Bounded, expiring map of chat id → Chat."""

    def __init__(self, max_chats: int, ttl_seconds: int):
        self.max_chats = max_chats
        self.ttl = ttl_seconds
        self._chats: "OrderedDict[str, Chat]" = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float):
        for cid in [c.id for c in self._chats.values() if now - c.last_used > self.ttl]:
            del self._chats[cid]
        while len(self._chats) > self.max_chats:
            self._chats.popitem(last=False)

    def create(self) -> Chat:
        chat = Chat(id=secrets.token_urlsafe(24))
        with self._lock:
            self._chats[chat.id] = chat
            self._expire(time.time())
        return chat

    def get(self, chat_id: str) -> Chat:
        now = time.time()
        with self._lock:
            self._expire(now)
            chat = self._chats.get(chat_id)
            if chat is None:
                raise HTTPException(404, "Chat not found. It may have expired — start a new chat.")
            chat.last_used = now
            self._chats.move_to_end(chat_id)
            return chat

    def delete(self, chat_id: str) -> None:
        with self._lock:
            self._chats.pop(chat_id, None)

    def clear(self) -> None:
        with self._lock:
            self._chats.clear()

    def __len__(self) -> int:
        with self._lock:
            self._expire(time.time())
            return len(self._chats)


class RateLimiter:
    """Soft sliding-window limit per client (protects the shared API quota)."""

    def __init__(self, limit: int, window_seconds: int = 3600):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.time()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                raise HTTPException(429, "Too many requests from your address. Please try again later.")
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


chats           = ChatStore(MAX_CHATS, CHAT_TTL_SECONDS)
upload_limiter  = RateLimiter(UPLOADS_PER_HOUR_IP)
message_limiter = RateLimiter(MESSAGES_PER_HOUR_IP)


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the visitor's address is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────────────────────
# Index + chain builders (replaced by fakes in tests)
# ─────────────────────────────────────────────────────────────

def build_chain(chunks: list, document_name: str):
    """Embeds one document's chunks into a fresh FAISS index and returns a RAGChain over it."""
    from vectorstore import VectorStoreManager
    from rag_chain import RAGChain

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set")
    embeddings = VectorStoreManager(index_path="").embeddings
    store = _faiss_from_documents(chunks, embeddings)
    return RAGChain(
        chunks=chunks,
        vectorstore=store,
        domain=f'the uploaded document "{document_name}"',
        memory_window=5,
        top_k=3,
    )


def _faiss_from_documents(chunks: list, embeddings):
    # Built in memory only: each chat gets its own index, nothing saved to disk.
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="Chat with one document per chat — Gemini + FAISS + BM25 hybrid search",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    question: str = Field(max_length=2000)


class MessageResponse(BaseModel):
    chat_id:     str
    document:    str
    question:    str
    answer:      str
    sources:     list
    chunks_used: int


ChatId = PathParam(..., pattern=CHAT_ID_PATTERN)


@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {
        "active_chats":            len(chats),
        "llm_configured":          bool(os.getenv("GOOGLE_API_KEY")),
        "max_upload_mb":           MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_chunks_per_document": MAX_CHUNKS,
        "chat_ttl_minutes":        CHAT_TTL_SECONDS // 60,
        "allowed_types":           sorted(ALLOWED_SUFFIXES),
    }


@app.post("/chats", status_code=201)
def create_chat():
    """Starts a new, empty chat. Upload its one document next."""
    return chats.create().info()


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str = ChatId):
    return chats.get(chat_id).info()


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str = ChatId):
    chats.delete(chat_id)
    return {"message": "Chat deleted."}


# Plain `def` endpoints: embedding and LLM calls block for seconds, and FastAPI
# runs sync endpoints in a thread pool so other visitors are not frozen.
@app.post("/chats/{chat_id}/document")
def upload_document(request: Request, chat_id: str = ChatId, file: UploadFile = File(...)):
    """
    Uploads the chat's ONE document (PDF or TXT). A chat that already has a
    document refuses another one (409): start a new chat for a new document.
    """
    chat = chats.get(chat_id)

    name   = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if not name or suffix not in ALLOWED_SUFFIXES:
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

    # Claim the chat's single document slot before the slow work, so two
    # parallel uploads to one chat cannot both succeed.
    with chat.lock:
        if chat.document or chat.indexing:
            raise HTTPException(409, "This chat already has a document. Start a new chat to use another one.")
        chat.indexing = True

    try:
        upload_limiter.check(_client_ip(request))
        try:
            ingester = DocumentIngester(chunk_size=1000, chunk_overlap=200)
            chunks   = ingester.split(ingester.load_bytes(name, contents))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Could not read that file.")
        if not chunks:
            raise HTTPException(400, "No text found in that file (scanned PDFs are not supported).")
        if len(chunks) > MAX_CHUNKS:
            raise HTTPException(413, f"Document is too long for this demo (limit {MAX_CHUNKS} chunks, "
                                     f"about {MAX_CHUNKS * 1000 // 3000} pages).")
        try:
            chain = build_chain(chunks, name)
        except Exception as e:
            print(f"Indexing failed: {e}")
            raise HTTPException(503, "Indexing failed. The server may be missing its API key or "
                                     "over its quota — please try again later.")
        with chat.lock:
            chat.document, chat.chunks, chat.chain = name, chunks, chain
    finally:
        chat.indexing = False

    return {"message": f"Indexed {name}. Ask anything about it.", **chat.info()}


@app.post("/chats/{chat_id}/messages", response_model=MessageResponse)
def send_message(req: MessageRequest, request: Request, chat_id: str = ChatId):
    """Asks a question about this chat's document only."""
    chat = chats.get(chat_id)
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if chat.chain is None:
        raise HTTPException(409, "Upload a document to this chat first.")
    message_limiter.check(_client_ip(request))

    # One question at a time per chat, so its memory stays in order.
    with chat.lock:
        try:
            result = chat.chain.ask(req.question, memory=chat.memory)
        except Exception as e:
            print(f"Answering failed: {e}")
            raise HTTPException(503, "The language model is unavailable right now. Please try again.")

    return MessageResponse(
        chat_id=chat.id,
        document=chat.document,
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
    )
