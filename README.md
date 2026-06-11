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
- **REST API** — FastAPI server with upload, chat, and memory endpoints

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
GEMINI_MODEL=gemini-2.5-flash
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

**API server:**
```bash
uvicorn backend.server:app --reload --port 8000
```
Then open http://localhost:8000/docs for interactive API docs.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/status` | Index and document info |
| POST | `/chat` | Ask a question |
| POST | `/upload` | Upload new document |
| DELETE | `/memory` | Clear conversation history |

## Project Structure
rag_chatbot/
├── backend/
│   ├── ingester.py          # PDF loading and chunking
│   ├── vectorstore.py       # FAISS index management
│   ├── retriever.py         # BM25 + FAISS hybrid search
│   ├── rag_chain.py         # RAG pipeline + memory
│   ├── server.py            # FastAPI REST server
│   ├── chat_cli.py          # Interactive terminal chat
│   └── create_sample_pdf.py # Sample data generator
├── data/sample_docs/        # Your documents go here
├── faiss_index/             # Auto-generated vector index
├── .env                     # API keys (never commit)
├── .gitignore
├── requirements.txt
└── README.md

## Resume Keywords

RAG, LangChain, FAISS, BM25, Hybrid Search, Reciprocal Rank Fusion,
Vector Embeddings, Semantic Search, FastAPI, Gemini API, NLP,
LLM, Retrieval-Augmented Generation, Python, REST API