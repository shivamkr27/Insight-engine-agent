"""
Ingestion pipeline: PDF → Markdown → Parent-Child chunks → ChromaDB + SQLite parent store

WHY two levels of chunks?
  - Child chunks (500 chars) → used for vector search (precise, focused match)
  - Parent chunks (2000-4000 chars) → passed to LLM (more context around the match)
  This is the "Parent-Document Retriever" pattern.

Flow per PDF:
  PDF
   └► pymupdf4llm → Markdown text
         └► MarkdownHeaderTextSplitter → parent sections (by H1/H2/H3)
               ├► merge tiny / split huge → cleaned parents
               ├► save each parent row to SQLite (parent_store.db)
               └► RecursiveCharacterTextSplitter → child chunks
                     └► HuggingFace embeddings → ChromaDB
"""

import os
import json
import glob
import sqlite3
import threading
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
    DOCS_DIR, CHROMA_DB_PATH, PARENT_STORE_DB_PATH, COLLECTION_NAME,
    EMBED_MODEL, CHROMA_HOST, CHROMA_PORT,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    MIN_PARENT_SIZE, MAX_PARENT_SIZE,
    HEADERS_TO_SPLIT_ON, USER_ISOLATION,
)
from .logging_config import get_logger

logger = get_logger(__name__)


# ── Parent Store (SQLite) ──────────────────────────────────────────────────────

class _ParentStore:
    """
    Stores full parent chunks in a SQLite database.
    Replaces the old per-file JSON approach — faster listing, multi-replica safe.
    """

    def __init__(self, db_path: str = PARENT_STORE_DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS parents (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                source      TEXT,
                metadata    TEXT,
                user_id     TEXT NOT NULL DEFAULT 'default',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migrate older tables that predate the user_id column
        try:
            self._conn.execute("SELECT user_id FROM parents LIMIT 1")
        except Exception:
            self._conn.execute("ALTER TABLE parents ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
        self._conn.commit()

    def save(self, parent_id: str, content: str, metadata: dict, user_id: str = "default") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO parents (id, content, source, metadata, user_id) VALUES (?, ?, ?, ?, ?)",
                (parent_id, content, metadata.get("source", ""), json.dumps(metadata), user_id),
            )
            self._conn.commit()

    def load(self, parent_id: str) -> dict:
        row = self._conn.execute(
            "SELECT content, metadata FROM parents WHERE id = ?", (parent_id,)
        ).fetchone()
        if not row:
            return {}
        return {"content": row[0], "metadata": json.loads(row[1])}

    def all_sources(self, user_id: str = None) -> List[str]:
        if user_id and user_id != "default" and USER_ISOLATION:
            rows = self._conn.execute(
                "SELECT DISTINCT source FROM parents WHERE source != '' AND user_id = ?",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT source FROM parents WHERE source != ''"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self, user_id: str = None) -> None:
        with self._lock:
            if user_id and user_id != "default" and USER_ISOLATION:
                self._conn.execute("DELETE FROM parents WHERE user_id = ?", (user_id,))
            else:
                self._conn.execute("DELETE FROM parents")
            self._conn.commit()


# ── Document Chunker ───────────────────────────────────────────────────────────

class _DocumentChunker:
    """Splits a Markdown document into parent-child chunk pairs."""

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
        raw_sections = self._header_splitter.split_text(markdown_text)

        parents = self._merge_small_parents(raw_sections)
        parents = self._split_large_parents(parents)
        parents = self._clean_small_chunks(parents)

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

    def _merge_small_parents(self, chunks: List[Document]) -> List[Document]:
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
        stats = ingestor.ingest_all()
        stats = ingestor.ingest_single("docs/rbi_report.pdf")
        vs    = ingestor.get_vectorstore()
    """

    def __init__(self):
        logger.info("Loading embedding model (first run downloads ~90MB)...")
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self._vectorstore = self._init_vectorstore()
        self._parent_store = _ParentStore()
        self._chunker = _DocumentChunker()
        logger.info("Ingestion ready.")

    def _init_vectorstore(self) -> Chroma:
        if CHROMA_HOST:
            import chromadb
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            logger.info(f"ChromaDB client-server mode: {CHROMA_HOST}:{CHROMA_PORT}")
            return Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
            )
        logger.info(f"ChromaDB local mode: {CHROMA_DB_PATH}")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_DB_PATH,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest_all(self, docs_dir: str = DOCS_DIR, user_id: str = "default") -> Dict:
        pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {docs_dir}")
            return {"ingested": [], "skipped": []}

        already_done = set(self._parent_store.all_sources(user_id=user_id))
        ingested, skipped = [], []

        for pdf_path in sorted(pdf_files):
            filename = os.path.basename(pdf_path)
            if filename in already_done:
                logger.info(f"Skipping (already ingested): {filename}")
                skipped.append(filename)
                continue
            stats = self.ingest_single(pdf_path, user_id=user_id)
            ingested.append({"file": filename, **stats})

        logger.info(f"Ingest complete — {len(ingested)} ingested, {len(skipped)} skipped.")
        return {"ingested": ingested, "skipped": skipped}

    def ingest_single(self, pdf_path: str, user_id: str = "default") -> Dict:
        pdf_path = Path(pdf_path)
        logger.info(f"Ingesting: {pdf_path.name} (user={user_id})")

        try:
            markdown = pymupdf4llm.to_markdown(str(pdf_path))

            parent_pairs, child_chunks = self._chunker.chunk(markdown, pdf_path.stem)
            logger.info(f"{pdf_path.name}: {len(parent_pairs)} parents | {len(child_chunks)} children")

            for parent_id, parent_doc in parent_pairs:
                self._parent_store.save(
                    parent_id,
                    parent_doc.page_content,
                    parent_doc.metadata,
                    user_id=user_id,
                )

            # Tag every child chunk with user_id so ChromaDB can filter per-user
            for chunk in child_chunks:
                chunk.metadata["user_id"] = user_id

            self._vectorstore.add_documents(child_chunks)
            logger.info(f"{pdf_path.name}: embedding done")

            return {"parent_chunks": len(parent_pairs), "child_chunks": len(child_chunks)}

        except Exception as e:
            logger.error(f"Ingestion failed for {pdf_path.name}: {e}", exc_info=True)
            return {"error": str(e)}

    def get_vectorstore(self) -> Chroma:
        return self._vectorstore

    def get_all_texts_and_meta(self) -> List[Tuple[str, dict]]:
        result = self._vectorstore.get(include=["documents", "metadatas"])
        docs  = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(docs)
        return list(zip(docs, metas))

    def load_parent(self, parent_id: str) -> dict:
        return self._parent_store.load(parent_id)

    def list_ingested_files(self, user_id: str = None) -> List[str]:
        return sorted(self._parent_store.all_sources(user_id=user_id))

    def clear_all(self, user_id: str = None) -> None:
        logger.info(f"Clearing data for user={user_id or 'all'}...")
        if user_id and user_id != "default" and USER_ISOLATION:
            self._parent_store.clear(user_id=user_id)
        else:
            self._vectorstore.delete_collection()
            self._parent_store.clear()
        logger.info("Clear done.")
