"""
Ingestion pipeline: PDF → Markdown → Parent-Child chunks → ChromaDB + JSON parent store

Interview concept: WHY two levels of chunks?
  - Child chunks (500 chars) → used for vector search (precise, focused match)
  - Parent chunks (2000-4000 chars) → passed to LLM (more context around the match)
  This is called the "Parent-Document Retriever" pattern.

Flow per PDF:
  PDF
   └─► pymupdf4llm  →  Markdown text
         └─► MarkdownHeaderTextSplitter  →  parent sections (by H1/H2/H3)
               ├─► merge tiny sections, split huge sections  →  cleaned parents
               ├─► save each parent as JSON  (parent_store/)
               └─► RecursiveCharacterTextSplitter  →  child chunks
                     └─► HuggingFace embeddings  →  ChromaDB
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from .config import (
    DOCS_DIR, CHROMA_DB_PATH, PARENT_STORE_PATH, COLLECTION_NAME,
    EMBED_MODEL,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    MIN_PARENT_SIZE, MAX_PARENT_SIZE,
    HEADERS_TO_SPLIT_ON,
)


# ── Parent Store ───────────────────────────────────────────────────────────────

class _ParentStore:
    """
    Stores full parent chunks as JSON files on disk.
    When the agent retrieves a child chunk, it fetches the full parent by ID.
    """

    def __init__(self, store_path: str = PARENT_STORE_PATH):
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, parent_id: str, content: str, metadata: dict) -> None:
        file_path = self._path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, parent_id: str) -> dict:
        """Returns {} if parent_id not found."""
        file_path = self._path / f"{parent_id}.json"
        if not file_path.exists():
            return {}
        return json.loads(file_path.read_text(encoding="utf-8"))

    def all_sources(self) -> List[str]:
        """Return unique source PDF filenames already present in the store."""
        sources = set()
        for f in self._path.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                src = data.get("metadata", {}).get("source", "")
                if src:
                    sources.add(src)
            except Exception:
                pass
        return list(sources)

    def clear(self) -> None:
        for f in self._path.glob("*.json"):
            f.unlink()


# ── Document Chunker ───────────────────────────────────────────────────────────

class _DocumentChunker:
    """
    Splits a Markdown document into parent-child chunk pairs.
    Adapted from: agentic-rag-for-dummies/project/document_chunker.py
    """

    def __init__(self):
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )

    def chunk(
        self, markdown_text: str, doc_stem: str
    ) -> Tuple[List[Tuple[str, Document]], List[Document]]:
        """
        Args:
            markdown_text: Full Markdown string of one PDF.
            doc_stem:      PDF filename without extension (used for IDs).

        Returns:
            parent_pairs:  [(parent_id, Document), ...]
            child_chunks:  [Document, ...]  each has metadata.parent_id set
        """
        # 1. Split by markdown headers into raw section chunks
        raw_sections = self._header_splitter.split_text(markdown_text)

        # 2. Clean up: merge tiny, split huge
        parents = self._merge_small_parents(raw_sections)
        parents = self._split_large_parents(parents)
        parents = self._clean_small_chunks(parents)

        # 3. Assign IDs → create child chunks
        parent_pairs: List[Tuple[str, Document]] = []
        child_chunks: List[Document] = []

        for i, parent_doc in enumerate(parents):
            parent_id = f"{doc_stem}_parent_{i}"
            parent_doc.metadata.update({
                "source": f"{doc_stem}.pdf",
                "parent_id": parent_id,
            })
            parent_pairs.append((parent_id, parent_doc))

            children = self._child_splitter.split_documents([parent_doc])
            child_chunks.extend(children)

        return parent_pairs, child_chunks

    # ── Private helpers ────────────────────────────────────────────────────

    def _merge_small_parents(self, chunks: List[Document]) -> List[Document]:
        """Concatenate consecutive sections until each parent >= MIN_PARENT_SIZE."""
        if not chunks:
            return []
        merged, current = [], None
        for chunk in chunks:
            current = chunk if current is None else self._concat(current, chunk)
            if len(current.page_content) >= MIN_PARENT_SIZE:
                merged.append(current)
                current = None
        if current:
            if merged:
                merged[-1] = self._concat(merged[-1], current)
            else:
                merged.append(current)
        return merged

    def _split_large_parents(self, chunks: List[Document]) -> List[Document]:
        """Break any chunk exceeding MAX_PARENT_SIZE."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_PARENT_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )
        result = []
        for chunk in chunks:
            if len(chunk.page_content) <= MAX_PARENT_SIZE:
                result.append(chunk)
            else:
                result.extend(splitter.split_documents([chunk]))
        return result

    def _clean_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """Absorb any remaining small chunks into neighbours."""
        cleaned = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < MIN_PARENT_SIZE:
                if cleaned:
                    cleaned[-1] = self._concat(cleaned[-1], chunk)
                elif i < len(chunks) - 1:
                    chunks[i + 1] = self._concat(chunk, chunks[i + 1])
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        return cleaned

    @staticmethod
    def _concat(a: Document, b: Document) -> Document:
        """Merge two Documents by concatenating content and unioning metadata."""
        merged_meta = dict(a.metadata)
        for k, v in b.metadata.items():
            if k not in merged_meta:
                merged_meta[k] = v
            else:
                merged_meta[k] = f"{merged_meta[k]} -> {v}"
        return Document(
            page_content=a.page_content + "\n\n" + b.page_content,
            metadata=merged_meta,
        )


# ── Main Ingestion Class ───────────────────────────────────────────────────────

class Ingestion:
    """
    Orchestrates the full ingestion pipeline.

    Usage:
        ingestor = Ingestion()

        # Ingest all PDFs in docs/ (skips already ingested ones)
        stats = ingestor.ingest_all()

        # Or ingest a single file
        stats = ingestor.ingest_single("docs/rbi_report.pdf")

        # Get vectorstore for query-time search
        vs = ingestor.get_vectorstore()
    """

    def __init__(self):
        print("⏳ Loading embedding model (first run downloads ~90MB)...")
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self._vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        self._parent_store = _ParentStore()
        self._chunker = _DocumentChunker()
        print("✓ Ingestion ready.")

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest_all(self, docs_dir: str = DOCS_DIR) -> Dict:
        """
        Scan docs_dir for PDFs and ingest each one that isn't already stored.
        Returns a summary dict with 'ingested' and 'skipped' lists.
        """
        pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
        if not pdf_files:
            print(f"⚠  No PDFs found in {docs_dir}")
            return {"ingested": [], "skipped": []}

        already_done = set(self._parent_store.all_sources())
        ingested, skipped = [], []

        for pdf_path in sorted(pdf_files):
            filename = os.path.basename(pdf_path)
            if filename in already_done:
                print(f"⏭  Skipping (already ingested): {filename}")
                skipped.append(filename)
                continue
            stats = self.ingest_single(pdf_path)
            ingested.append({"file": filename, **stats})

        print(f"\n✅ Done — {len(ingested)} ingested, {len(skipped)} skipped.")
        return {"ingested": ingested, "skipped": skipped}

    def ingest_single(self, pdf_path: str) -> Dict:
        """
        Full pipeline for one PDF file.
        Returns chunk counts for UI display.
        """
        pdf_path = Path(pdf_path)
        print(f"\n📄 {pdf_path.name}")

        try:
            # Step 1 — PDF → Markdown
            print("  ├─ PDF → Markdown...")
            markdown = pymupdf4llm.to_markdown(str(pdf_path))

            # Step 2 — Markdown → parent + child chunks
            print("  ├─ Chunking (parent-child)...")
            parent_pairs, child_chunks = self._chunker.chunk(markdown, pdf_path.stem)
            print(f"  │   {len(parent_pairs)} parents  |  {len(child_chunks)} children")

            # Step 3 — Save parents to disk
            for parent_id, parent_doc in parent_pairs:
                self._parent_store.save(
                    parent_id,
                    parent_doc.page_content,
                    parent_doc.metadata,
                )

            # Step 4 — Embed children → ChromaDB
            print("  ├─ Embedding → ChromaDB...")
            self._vectorstore.add_documents(child_chunks)
            print("  └─ ✓")

            return {"parent_chunks": len(parent_pairs), "child_chunks": len(child_chunks)}

        except Exception as e:
            print(f"  └─ ✗ Error: {e}")
            return {"error": str(e)}

    def get_vectorstore(self) -> Chroma:
        """Return the live ChromaDB vectorstore (used by search tool for dense retrieval)."""
        return self._vectorstore

    def get_all_texts_and_meta(self) -> List[Tuple[str, dict]]:
        """
        Fetch every stored child chunk as (text, metadata) pairs.
        Called by tools.py at startup to build the BM25 index.
        """
        result = self._vectorstore.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(docs)
        return list(zip(docs, metas))

    def load_parent(self, parent_id: str) -> dict:
        """Load a full parent chunk by ID. Returns {} if not found."""
        return self._parent_store.load(parent_id)

    def list_ingested_files(self) -> List[str]:
        """Sorted list of all ingested PDF filenames (for UI sidebar display)."""
        return sorted(self._parent_store.all_sources())

    def clear_all(self) -> None:
        """Wipe all stored data — ChromaDB collection + parent JSON files."""
        print("🗑  Clearing ChromaDB collection...")
        self._vectorstore.delete_collection()
        print("🗑  Clearing parent store...")
        self._parent_store.clear()
        print("✓ All cleared.")
