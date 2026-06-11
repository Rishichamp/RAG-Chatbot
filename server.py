"""
FastAPI Server for RAG Chatbot
Exposes REST endpoints + WebSocket streaming
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import our RAG engine
from rag_engine import build_rag_pipeline, RAGChatbot, DocumentIngester, VectorStoreManager, build_hybrid_retriever

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global bot instance (one per server; scale with Redis for multi-user) ──
bot: Optional[RAGChatbot] = None
DATA_DIR = "./data"
INDEX_DIR = "./faiss_index"

os.makedirs(DATA_DIR, exist_ok=True)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str

class StatusResponse(BaseModel):
    status: str
    indexed_files: int


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global bot
    if Path(INDEX_DIR).exists() and os.listdir(DATA_DIR):
        print("[Server] Loading existing RAG pipeline...")
        bot = build_rag_pipeline(
            data_dir=DATA_DIR,
            index_path=INDEX_DIR,
            domain_name=os.getenv("DOMAIN_NAME", "Knowledge Base"),
        )
    else:
        print("[Server] No index found. Upload documents to initialize.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
async def get_status():
    data_files = list(Path(DATA_DIR).rglob("*.*"))
    return {
        "status": "ready" if bot else "no_documents",
        "indexed_files": len(data_files),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or DOCX for indexing."""
    allowed = {".pdf", ".txt", ".docx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    dest = Path(DATA_DIR) / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Rebuild the index with new file
    global bot
    bot = build_rag_pipeline(
        data_dir=DATA_DIR,
        index_path=INDEX_DIR,
        domain_name=os.getenv("DOMAIN_NAME", "Knowledge Base"),
        force_rebuild=True,
    )

    return {"message": f"Indexed {file.filename} successfully", "file": file.filename}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question and get a grounded answer."""
    if not bot:
        raise HTTPException(503, "No documents indexed yet. Upload documents first.")

    session_id = req.session_id or str(uuid.uuid4())
    result = bot.chat(req.question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": session_id,
    }


@app.delete("/memory")
async def clear_memory():
    """Reset conversation memory."""
    if bot:
        bot.clear_memory()
    return {"message": "Memory cleared"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming responses.
    Client sends: {"question": "..."}
    Server streams tokens back in real-time.
    """
    await websocket.accept()
    if not bot:
        await websocket.send_json({"error": "No documents indexed"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            if not question:
                continue

            result = bot.chat(question)
            # Stream word by word (simulated — replace with real LangChain streaming)
            words = result["answer"].split()
            for i, word in enumerate(words):
                await websocket.send_json({
                    "token": word + (" " if i < len(words) - 1 else ""),
                    "done": False,
                })
                await asyncio.sleep(0.02)

            await websocket.send_json({
                "token": "",
                "done": True,
                "sources": result["sources"],
            })
    except Exception:
        await websocket.close()


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
