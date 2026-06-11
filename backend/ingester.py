"""
ingester.py  —  Phase 2: Document Ingestion

Responsibilities:
  1. Load PDF / TXT files from a folder
  2. Split them into overlapping chunks
  3. Attach metadata (source filename, page number) to every chunk

Key concepts:
  - PyPDFLoader     : reads a PDF page by page
  - RecursiveCharacterTextSplitter : splits text intelligently
  - Document        : LangChain object = text content + metadata dict
  - chunk_size      : max characters per chunk (1000)
  - chunk_overlap   : characters shared between adjacent chunks (200)
"""
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIngester:
    """
    Loads documents from disk and splits them into chunks
    ready for embedding and vector storage.
    Usage:
        ingester = DocumentIngester()
        chunks   = ingester.load_and_split("data/sample_docs")
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        # RecursiveCharacterTextSplitter tries separators IN ORDER:
        #  "\n\n" → paragraph break  (best, most context preserved)
        #  "\n"   → line break
        #  ". "   → sentence end
        #  " "    → word boundary
        #  ""     → character split (last resort)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Loaders ───────────────────────────────────────────────
    def load_pdf(self, path: str) -> List[Document]:
        """
        Loads a PDF file page by page.
        Returns one Document per page, each with metadata:
          {"source": "path/to/file.pdf", "page": 0}
        """
        loader = PyPDFLoader(path)
        return loader.load()

    def load_txt(self, path: str) -> List[Document]:
        """Loads a plain text file as a single Document."""
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        Scans a folder recursively.
        Loads all .pdf and .txt files found.
        Returns a flat list of all pages/documents.
        """
        docs = []
        path = Path(dir_path)

        for pdf in path.rglob("*.pdf"):
            pages = self.load_pdf(str(pdf))
            docs.extend(pages)
            print(f"  Loaded : {pdf.name}  ({len(pages)} page(s))")

        for txt in path.rglob("*.txt"):
            loaded = self.load_txt(str(txt))
            docs.extend(loaded)
            print(f"  Loaded : {txt.name}  ({len(loaded)} doc(s))")

        print(f"  Total raw documents : {len(docs)}")
        return docs

    # ── Splitter ──────────────────────────────────────────────
    def split(self, docs: List[Document]) -> List[Document]:
        """
        Splits each Document into overlapping chunks.
        Adds chunk_index to metadata for traceability.
        """
        chunks = self.splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(f"  Chunks created : {len(chunks)}")
        print(f"  chunk_size     : {self.chunk_size} chars")
        print(f"  chunk_overlap  : {self.chunk_overlap} chars")
        return chunks

    # ── Main entry point ──────────────────────────────────────
    def load_and_split(self, dir_path: str) -> List[Document]:
        """Loads all docs from dir_path and splits into chunks."""
        print(f"\nIngesting: {dir_path}")
        print("-" * 40)
        docs   = self.load_directory(dir_path)
        chunks = self.split(docs)
        print("-" * 40)
        return chunks


# ── Inspection helper ─────────────────────────────────────────
def inspect_chunks(chunks: List[Document], show: int = 3) -> None:
    """
    Prints the first `show` chunks in detail.
    Used during development to verify chunking output.
    """
    print(f"\nShowing first {show} of {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:show]):
        print(f"\n{'─' * 48}")
        print(f"  Chunk #       : {i}")
        print(f"  Source        : {chunk.metadata.get('source', 'unknown')}")
        print(f"  Page          : {chunk.metadata.get('page', 'N/A')}")
        print(f"  Chunk index   : {chunk.metadata.get('chunk_index', 'N/A')}")
        print(f"  Length        : {len(chunk.page_content)} characters")
        print(f"  Preview       :")
        print(f"  {chunk.page_content[:300].strip()}")
        print(f"  ...")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    ingester = DocumentIngester(chunk_size=1000, chunk_overlap=200)
    chunks   = ingester.load_and_split("data/sample_docs")

    inspect_chunks(chunks, show=3)

    print(f"\n{'─' * 48}")
    print("Full text of chunk #0:\n")
    print(chunks[0].page_content)
    print(f"\nMetadata of chunk #0:")
    print(chunks[0].metadata)
    print(f"\n  Total chunks ready for embedding: {len(chunks)}")