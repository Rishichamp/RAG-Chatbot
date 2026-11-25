"""API tests that need no Gemini key: embedding + LLM are replaced by fakes,
but document parsing, chunking and the chat rules run for real."""
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import server  # noqa: E402
from rag_chain import ConversationMemory, RAGChain  # noqa: E402


class FakeChain:
    """Answers with the text of the chunks it was built from, plus the
    history it was given — so a test can see exactly which document and
    which conversation a chat is using."""

    def __init__(self, chunks, document_name):
        self.chunks = chunks
        self.document_name = document_name

    def ask(self, question, memory):
        text = " ".join(c.page_content for c in self.chunks)
        answer = f"[{self.document_name}] {text} | history: {memory.format_history()}"
        memory.add(question, answer)
        return {"question": question, "answer": answer,
                "sources": [{"file": self.document_name, "page": 1, "preview": ""}],
                "chunks_used": len(self.chunks)}


@pytest.fixture
def built(monkeypatch):
    """Records every index build (one per uploaded document)."""
    calls = []

    def fake_build(chunks, name):
        calls.append(name)
        return FakeChain(chunks, name)

    monkeypatch.setattr(server, "build_chain", fake_build)
    return calls


@pytest.fixture
def client(built):
    server.chats.clear()
    server.upload_limiter.reset()
    server.message_limiter.reset()
    with TestClient(server.app) as c:
        yield c


def make_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def new_chat(client) -> str:
    r = client.post("/chats")
    assert r.status_code == 201
    return r.json()["chat_id"]


def upload(client, chat_id, name, data, mime="text/plain"):
    return client.post(f"/chats/{chat_id}/document", files={"file": (name, data, mime)})


def ask(client, chat_id, question):
    return client.post(f"/chats/{chat_id}/messages", json={"question": question})


# ── One document per chat ─────────────────────────────────────

def test_chat_answers_only_from_its_own_document(client):
    a = new_chat(client)
    r = upload(client, a, "cats.pdf", make_pdf("Cats sleep sixteen hours a day."), "application/pdf")
    assert r.status_code == 200, r.text
    assert r.json()["document"] == "cats.pdf" and r.json()["chunks"] >= 1

    reply = ask(client, a, "How long do cats sleep?").json()
    assert reply["document"] == "cats.pdf"
    assert "sixteen hours" in reply["answer"]
    assert reply["sources"][0]["file"] == "cats.pdf"


def test_second_document_in_same_chat_is_refused(client, built):
    a = new_chat(client)
    assert upload(client, a, "first.txt", b"alpha facts").status_code == 200
    r = upload(client, a, "second.txt", b"beta facts")
    assert r.status_code == 409
    assert "new chat" in r.json()["detail"].lower()
    # the chat still uses only its first document
    assert built == ["first.txt"]
    answer = ask(client, a, "what?").json()["answer"]
    assert "alpha" in answer and "beta" not in answer


def test_new_chat_can_use_another_document(client):
    a, b = new_chat(client), new_chat(client)
    assert a != b
    assert upload(client, a, "a.txt", b"apples are red").status_code == 200
    assert upload(client, b, "b.txt", b"bananas are yellow").status_code == 200
    ans_a = ask(client, a, "colour?").json()["answer"]
    ans_b = ask(client, b, "colour?").json()["answer"]
    assert "apples" in ans_a and "bananas" not in ans_a
    assert "bananas" in ans_b and "apples" not in ans_b


def test_chats_keep_separate_history(client):
    a, b = new_chat(client), new_chat(client)
    upload(client, a, "a.txt", b"doc a")
    upload(client, b, "b.txt", b"doc b")
    ask(client, a, "my secret is 42")
    other = ask(client, b, "what was said before?").json()["answer"]
    assert "42" not in other
    again = ask(client, a, "and now?").json()["answer"]
    assert "my secret is 42" in again
    assert client.get(f"/chats/{a}").json()["history"][0]["human"] == "my secret is 42"


def test_question_before_upload_is_refused(client):
    a = new_chat(client)
    r = ask(client, a, "anything?")
    assert r.status_code == 409
    assert "upload" in r.json()["detail"].lower()


def test_parallel_uploads_to_one_chat_index_only_one(client, monkeypatch):
    started, release = threading.Event(), threading.Event()
    names = []

    def slow_build(chunks, name):
        names.append(name)
        started.set()
        release.wait(5)
        return FakeChain(chunks, name)

    monkeypatch.setattr(server, "build_chain", slow_build)
    a = new_chat(client)
    results = {}
    t = threading.Thread(target=lambda: results.setdefault("first", upload(client, a, "one.txt", b"one")))
    t.start()
    assert started.wait(5)
    second = upload(client, a, "two.txt", b"two")   # arrives while the first is indexing
    release.set()
    t.join(5)
    assert results["first"].status_code == 200
    assert second.status_code == 409
    assert names == ["one.txt"]


def test_failed_upload_frees_the_slot(client, monkeypatch):
    a = new_chat(client)
    assert upload(client, a, "empty.txt", b"   ").status_code == 400      # no text
    assert upload(client, a, "ok.txt", b"real text").status_code == 200    # slot still free

    b = new_chat(client)
    monkeypatch.setattr(server, "build_chain", lambda c, n: (_ for _ in ()).throw(RuntimeError("quota")))
    assert upload(client, b, "x.txt", b"text").status_code == 503
    assert client.get(f"/chats/{b}").json()["document"] is None


# ── Chat ids and lifetime ─────────────────────────────────────

def test_unknown_or_malformed_chat_id(client):
    assert client.get("/chats/" + "x" * 32).status_code == 404
    assert ask(client, "y" * 32, "hi").status_code == 404
    assert client.get("/chats/short").status_code == 422
    assert client.get("/chats/..%2F..%2Fetc").status_code in (404, 422)


def test_delete_chat(client):
    a = new_chat(client)
    upload(client, a, "a.txt", b"text")
    assert client.delete(f"/chats/{a}").status_code == 200
    assert client.get(f"/chats/{a}").status_code == 404


def test_store_expires_idle_and_oldest_chats(monkeypatch):
    store = server.ChatStore(max_chats=2, ttl_seconds=60)
    clock = [1000.0]
    monkeypatch.setattr(server.time, "time", lambda: clock[0])
    first = store.create()
    second = store.create()
    third = store.create()                      # over the cap: the oldest goes
    with pytest.raises(server.HTTPException):
        store.get(first.id)
    assert store.get(second.id) and store.get(third.id)
    clock[0] += 61                              # idle past the TTL
    with pytest.raises(server.HTTPException):
        store.get(third.id)


# ── Upload validation and limits ──────────────────────────────

def test_upload_validation(client):
    a = new_chat(client)
    assert upload(client, a, "x.pdf", b"<html>", "application/pdf").status_code == 400
    assert upload(client, a, "x.txt", "hi".encode("utf-16")).status_code == 400
    assert upload(client, a, "x.exe", b"MZ", "application/octet-stream").status_code == 400
    assert upload(client, a, "noext", b"hi").status_code == 400
    # path parts in the name are dropped, never used as a location
    r = upload(client, a, "../../evil.txt", b"hello")
    assert r.status_code == 200 and r.json()["document"] == "evil.txt"


def test_upload_size_and_length_limits(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    a = new_chat(client)
    assert upload(client, a, "big.txt", b"x" * 11).status_code == 413
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10_000_000)
    monkeypatch.setattr(server, "MAX_CHUNKS", 1)
    long_text = ("word " * 400).encode()                    # ~2000 chars → several chunks
    assert upload(client, a, "long.txt", long_text).status_code == 413
    assert upload(client, a, "short.txt", b"short").status_code == 200


def test_rate_limits(client, monkeypatch):
    monkeypatch.setattr(server, "upload_limiter", server.RateLimiter(1))
    monkeypatch.setattr(server, "message_limiter", server.RateLimiter(1))
    a, b = new_chat(client), new_chat(client)
    assert upload(client, a, "a.txt", b"text").status_code == 200
    assert upload(client, b, "b.txt", b"text").status_code == 429
    assert ask(client, a, "one").status_code == 200
    assert ask(client, a, "two").status_code == 429


def test_question_limits(client):
    a = new_chat(client)
    upload(client, a, "a.txt", b"text")
    assert ask(client, a, "   ").status_code == 400
    assert ask(client, a, "x" * 2001).status_code == 422


def test_page_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "New chat" in page.text
    assert client.get("/status").json()["allowed_types"] == [".pdf", ".txt"]


def test_pdf_pages_keep_page_numbers():
    from ingester import DocumentIngester

    pdf = FPDF()
    for text in ("first page text", "second page text"):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text)
    docs = DocumentIngester().load_bytes("two.pdf", bytes(pdf.output()))
    assert [d.metadata["page"] for d in docs] == [0, 1]
    assert all(d.metadata["source"] == "two.pdf" for d in docs)


# ── RAG chain ─────────────────────────────────────────────────

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


def test_chain_searches_only_its_own_chunks(monkeypatch):
    """The retrieval step gets this chat's chunks and index, nothing global."""
    import rag_chain

    seen = {}
    monkeypatch.setattr(rag_chain, "call_llm_safe", lambda llm, prompt, delay=0: "ans")
    monkeypatch.setattr(rag_chain, "hybrid_search", lambda **kw: seen.update(kw) or [])
    chain = RAGChain.__new__(RAGChain)
    mine_chunks, mine_index = ["chunk-of-this-chat"], object()
    chain.memory, chain.domain, chain.top_k, chain.chunks, chain.vectorstore, chain.llm = (
        ConversationMemory(), "x", 3, mine_chunks, mine_index, None)
    chain.ask("q", memory=ConversationMemory())
    assert seen["chunks"] is mine_chunks and seen["vectorstore"] is mine_index
