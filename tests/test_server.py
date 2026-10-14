"""API tests that need no Gemini key: the RAG pipeline is replaced by fakes."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import server  # noqa: E402
from rag_chain import ConversationMemory, RAGChain  # noqa: E402


class FakeChain:
    """Answers with the conversation so far, like the real chain's memory use."""
    memory = ConversationMemory()

    def ask(self, question, memory=None):
        memory = memory if memory is not None else self.memory
        history = memory.format_history()
        memory.add(question, f"answer to {question}")
        return {"question": question, "answer": history, "sources": [], "chunks_used": 0}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "load_pipeline", lambda: 0)
    monkeypatch.setattr(server, "_rebuild_pipeline", lambda: 1)
    monkeypatch.setattr(server, "chain", FakeChain())
    monkeypatch.setattr(server, "UPLOAD_TOKEN", "")
    server._sessions.clear()
    with TestClient(server.app) as c:
        yield c


def test_sessions_do_not_share_history(client):
    first = client.post("/chat", json={"question": "my secret is 42"}).json()
    sid = first["session_id"]
    # someone else (no session id) must not see the first visitor's question
    other = client.post("/chat", json={"question": "what was said before?"}).json()
    assert other["session_id"] != sid
    assert "42" not in other["answer"]
    # the same session does continue its own conversation
    again = client.post("/chat", json={"question": "and now?", "session_id": sid}).json()
    assert "my secret is 42" in again["answer"]


def test_clearing_one_session(client):
    sid = client.post("/chat", json={"question": "hello"}).json()["session_id"]
    client.delete("/memory", params={"session_id": sid})
    again = client.post("/chat", json={"question": "anything?", "session_id": sid}).json()
    assert again["answer"] == "No previous conversation."


def test_question_limits(client):
    assert client.post("/chat", json={"question": "   "}).status_code == 400
    assert client.post("/chat", json={"question": "x" * 2001}).status_code == 422


def test_upload_validates_content_and_never_overwrites(client, tmp_path):
    (tmp_path / "handbook.pdf").write_bytes(b"%PDF-original")
    r = client.post("/upload", files={"file": ("handbook.pdf", b"%PDF-1.4 new", "application/pdf")})
    assert r.status_code == 200, r.text
    assert (tmp_path / "handbook.pdf").read_bytes() == b"%PDF-original"
    assert len(list(tmp_path.glob("handbook-*.pdf"))) == 1

    assert client.post("/upload", files={"file": ("x.pdf", b"<html>", "application/pdf")}).status_code == 400
    assert client.post("/upload", files={"file": ("x.txt", "hi".encode("utf-16"), "text/plain")}).status_code == 400
    assert client.post("/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    r = client.post("/upload", files={"file": ("../../evil.txt", b"hello", "text/plain")})
    assert r.status_code == 200 and (tmp_path / "evil.txt").exists()


def test_upload_token(client, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_TOKEN", "s3cret")
    files = {"file": ("a.txt", b"hello", "text/plain")}
    assert client.post("/upload", files=files).status_code == 401
    assert client.post("/upload", files=files, headers={"X-Upload-Token": "nope"}).status_code == 401
    assert client.post("/upload", files=files, headers={"X-Upload-Token": "s3cret"}).status_code == 200


def test_chain_uses_given_memory(monkeypatch):
    """RAGChain.ask must read and write the memory it is given, not its own."""
    import rag_chain

    calls = []
    monkeypatch.setattr(rag_chain, "call_llm_safe", lambda llm, prompt, delay=0: calls.append(prompt) or "ans")
    monkeypatch.setattr(rag_chain, "hybrid_search", lambda **kw: [])
    chain = RAGChain.__new__(RAGChain)
    chain.memory, chain.domain, chain.top_k, chain.chunks, chain.vectorstore, chain.llm = (
        ConversationMemory(), "x", 3, [], None, None)
    mine = ConversationMemory()
    mine.add("earlier question", "earlier answer")
    chain.ask("follow up", memory=mine)
    assert "earlier question" in calls[0]          # condensed with this session's history
    assert chain.memory.is_empty()                  # the shared memory was not touched
    assert mine.history[-1]["human"] == "follow up"
