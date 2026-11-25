# RAG Chatbot — Custom Domain ChatGPT

A production-ready Retrieval-Augmented Generation (RAG) system
that lets you chat with your own PDF documents using AI.

## What It Does

Upload any PDF → Ask questions in natural language → Get grounded
answers with source citations. No hallucination. No guessing.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash (free tier) |
| Embeddings | Gemini Embedding-001 |
| Vector Store | FAISS |
| Keyword Search | BM25 |
| Framework | LangChain 0.3.x |
| API Server | FastAPI |
| Language | Python 3.13 |

## Architecture

PDF Documents
↓
DocumentIngester (PyPDF + chunking)
↓
VectorStoreManager (Gemini embeddings → FAISS index)
↓
HybridRetriever (BM25 40% + FAISS 60% → RRF merge)
↓
RAGChain (memory + condensation + Gemini → grounded answer)
↓
FastAPI Server (REST endpoints)

## Key Features

- **Hybrid Search** — BM25 keyword + FAISS semantic via custom RRF
- **Multi-turn Memory** — remembers last 5 conversation turns
- **Question Condensation** — rewrites follow-ups as standalone questions
- **Hallucination Prevention** — LLM constrained to document context only
- **Source Citations** — every answer cites filename and page number
- **Incremental Indexing** — add documents without full rebuild
- **One document per chat** — every visitor can upload one document per chat; each chat has its own isolated index and memory
- **Web app + REST API** — a built-in chat page served by FastAPI

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/rag_chatbot.git
cd rag_chatbot
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API key

Create a `.env` file in the root folder:
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
Get a free key at: https://aistudio.google.com/api-keys

### 4. Add your documents

Place PDF or TXT files in `data/sample_docs/`

### 5. Build the vector index

```bash
python backend/vectorstore.py
```

### 6. Start chatting

**Terminal chat:**

```bash
python backend/chat_cli.py
```

**Web app + API server:**

```bash
uvicorn backend.server:app --reload --port 8000
```
Then open http://localhost:8000 for the chat page, or http://localhost:8000/docs
for interactive API docs. (The web app does not need step 5: each chat
indexes its own uploaded document.)

## One document per chat (web app)

Open the server's home page (`/`) in a browser:

1. Click **+ New chat**.
2. Upload **one** PDF or TXT file. The chat builds its own BM25 + FAISS index
   for that file only.
3. Ask questions. Answers come only from that document, with page citations.
4. To use a different document, start a **new chat**. A chat that already has
   a document refuses a second upload.

Every visitor can do this. Each chat keeps its own document, index and
conversation memory, separate from all other chats and visitors, and is only
reachable by its random, unguessable chat id. Chats live in server memory:
idle chats expire after `CHAT_TTL_MINUTES` (default 120) and everything resets
when the server restarts.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web chat page |
| GET | `/health` | Health check |
| GET | `/status` | Limits and number of active chats |
| POST | `/chats` | Start a new chat → `{"chat_id": "..."}` |
| GET | `/chats/{chat_id}` | Chat info: document name and history |
| POST | `/chats/{chat_id}/document` | Upload the chat's one PDF/TXT (409 if it already has one) |
| POST | `/chats/{chat_id}/messages` | Ask: `{"question": "..."}` (409 until a document is uploaded) |
| DELETE | `/chats/{chat_id}` | Delete a chat and its index |

### Limits (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAX_UPLOAD_MB` | 10 | Largest file accepted |
| `MAX_CHUNKS_PER_DOCUMENT` | 200 | Longest document accepted (≈ 65 pages) |
| `MAX_CHATS` | 100 | Chats kept at once (oldest dropped first) |
| `CHAT_TTL_MINUTES` | 120 | Idle chats expire after this |
| `UPLOADS_PER_HOUR_PER_IP` | 10 | Uploads per visitor address per hour |
| `MESSAGES_PER_HOUR_PER_IP` | 60 | Questions per visitor address per hour |
| `LLM_CALL_DELAY_SECONDS` | 12 | Pause before each Gemini call (free-tier rate limit) |

## Deploy on Render (free)

`render.yaml` is a Render Blueprint. In the Render dashboard: **New → Blueprint**
→ pick this repo → enter `GOOGLE_API_KEY` when asked → **Apply**. The key lives
only in Render's environment settings; `.env` is git-ignored and must never be
committed.

## Tests

```bash
pip install pytest httpx
python -m pytest -q tests
```

The tests replace the Gemini pipeline with fakes, so no API key is needed.

## Project Structure

rag_chatbot/
├── backend/
│   ├── ingester.py          # PDF loading and chunking
│   ├── vectorstore.py       # FAISS index management
│   ├── retriever.py         # BM25 + FAISS hybrid search
│   ├── rag_chain.py         # RAG pipeline + memory
│   ├── server.py            # FastAPI server: one document per chat
│   ├── static/index.html    # Web chat page
│   ├── chat_cli.py          # Interactive terminal chat
│   └── create_sample_pdf.py # Sample data generator
├── data/sample_docs/        # Your documents go here
├── faiss_index/             # Auto-generated vector index
├── .env                     # API keys (never commit)
├── .gitignore
├── render.yaml            # Render deploy blueprint
├── requirements.txt
└── README.md

## Resume Keywords

RAG, LangChain, FAISS, BM25, Hybrid Search, Reciprocal Rank Fusion,
Vector Embeddings, Semantic Search, FastAPI, Gemini API, NLP,
LLM, Retrieval-Augmented Generation, Python, REST API
