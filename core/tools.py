"""
RAG search tool used by the agent subgraph.

Hybrid Search pipeline:
  Query
   ├► Dense search (ChromaDB cosine similarity)
   ├► Sparse search (BM25 keyword-frequency)
   ├► Score fusion  (weighted sum, both normalised to [0,1])
   └► Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

Adaptive Retrieval: search_chunks accepts a retrieval_mode parameter
("factual" | "conceptual" | "comparative" | "auto") that dynamically
adjusts k, dense/BM25 weights, and top-k after reranking.

BM25 index is persisted to disk as a pickle so cold-start rebuilds are
skipped when the corpus hasn't changed. A threading.RLock prevents race
conditions when documents are ingested while searches are in flight.
"""

import pickle
import threading
import numpy as np
from contextvars import ContextVar
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import (
    DEFAULT_K, DENSE_WEIGHT, BM25_WEIGHT, TOP_K_AFTER_RERANK,
    RERANKER_MODEL, BM25_CACHE_PATH, USER_ISOLATION, RETRIEVAL_PROFILES,
)

# Carries the active user_id into tool calls without modifying tool signatures.
# Set this before calling graph.astream() so it is inherited by all async tasks.
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="default")
from .ingestion import Ingestion
from .logging_config import get_logger

logger = get_logger(__name__)


# ── Adaptive retrieval helpers ──────────────────────────────────────────────────

def _classify_query(query: str) -> str:
    """Keyword-based query type classifier — no LLM call needed."""
    q     = query.lower()
    words = set(q.split())

    # Single-word signals — checked at word boundary to avoid substring false-positives
    compare_words  = {"vs", "compare", "difference", "versus", "contrast", "badla"}
    factual_words  = {"amount", "rate", "year", "date", "when", "kitna", "kab",
                      "total", "number", "count", "percentage", "target", "value", "kitne"}

    # Multi-word phrases — substring match is acceptable here
    compare_phrases = ["compared to", "pehle baad", "changes between"]
    factual_phrases = ["kya tha", "kya hai", "how much"]

    if words & compare_words or any(p in q for p in compare_phrases):
        return "comparative"
    if words & factual_words or any(p in q for p in factual_phrases):
        return "factual"
    return "conceptual"


def _get_retrieval_params(mode: str, query: str) -> dict:
    """Return search parameters for the given retrieval mode."""
    if mode == "auto":
        mode = _classify_query(query)
    return RETRIEVAL_PROFILES.get(mode, RETRIEVAL_PROFILES["conceptual"])


class ToolFactory:
    """
    Builds LangChain tools for the RAG agent subgraph.
    Created once at startup; shared across all agent invocations.
    """

    def __init__(self, ingestion: Ingestion):
        self._vectorstore: Chroma = ingestion.get_vectorstore()
        self._ingestion = ingestion
        self._bm25_lock = threading.RLock()

        logger.info("Loading CrossEncoder reranker...")
        self._reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker ready.")

        self._bm25: BM25Okapi | None = None
        self._corpus_texts: List[str] = []
        self._corpus_meta: List[dict] = []

        if not self._load_bm25_cache():
            self._build_bm25_index()

    # ── BM25 index management ───────────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        with self._bm25_lock:
            pairs = self._ingestion.get_all_texts_and_meta()
            if not pairs:
                logger.warning("No documents in ChromaDB yet — BM25 index is empty.")
                self._bm25 = None
                return

            self._corpus_texts = [text for text, _ in pairs]
            self._corpus_meta  = [meta for _, meta in pairs]

            tokenized  = [t.lower().split() for t in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 index built: {len(self._corpus_texts)} chunks.")
            self._save_bm25_cache()

    def rebuild_bm25(self) -> None:
        """Call after ingesting new documents to keep BM25 in sync."""
        self._build_bm25_index()

    def _save_bm25_cache(self) -> None:
        try:
            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "bm25":         self._bm25,
                    "corpus_texts": self._corpus_texts,
                    "corpus_meta":  self._corpus_meta,
                }, f)
            logger.info(f"BM25 cache saved: {BM25_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not save BM25 cache: {e}")

    def _load_bm25_cache(self) -> bool:
        if not Path(BM25_CACHE_PATH).exists():
            return False
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25         = data["bm25"]
            self._corpus_texts = data["corpus_texts"]
            self._corpus_meta  = data["corpus_meta"]
            logger.info(f"BM25 cache loaded: {len(self._corpus_texts)} chunks.")
            return True
        except Exception as e:
            logger.warning(f"BM25 cache load failed (will rebuild): {e}")
            return False

    # ── Hybrid search core ──────────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        source_filter: str = "",
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        top_k_after_rerank: int = TOP_K_AFTER_RERANK,
    ) -> List[Tuple[Document, float]]:
        user_id       = user_id_ctx.get()
        apply_user    = USER_ISOLATION and user_id not in ("default", "all")
        apply_source  = bool(source_filter)

        with self._bm25_lock:
            corpus_texts = list(self._corpus_texts)
            corpus_meta  = list(self._corpus_meta)
            bm25         = self._bm25

        # 1. Dense retrieval — build combined filter when needed
        if apply_user and apply_source:
            chroma_filter = {"$and": [{"user_id": user_id}, {"source": source_filter}]}
        elif apply_user:
            chroma_filter = {"user_id": user_id}
        elif apply_source:
            chroma_filter = {"source": source_filter}
        else:
            chroma_filter = None

        try:
            dense_raw: List[Tuple[Document, float]] = (
                self._vectorstore.similarity_search_with_relevance_scores(
                    query, k=k * 2, filter=chroma_filter
                )
            )
        except Exception as exc:
            logger.warning(f"ChromaDB filtered query failed ({exc}), retrying unfiltered")
            dense_raw = self._vectorstore.similarity_search_with_relevance_scores(query, k=k * 2)
            # Post-filter in Python as fallback
            if apply_user or apply_source:
                dense_raw = [
                    (doc, score) for doc, score in dense_raw
                    if (
                        (not apply_user   or doc.metadata.get("user_id", "default") == user_id) and
                        (not apply_source or doc.metadata.get("source", "") == source_filter)
                    )
                ]

        # 2. BM25 retrieval — mask out documents not matching the active filters
        bm25_scores = np.zeros(len(corpus_texts))
        if bm25 is not None:
            raw_scores = np.array(bm25.get_scores(query.lower().split()), dtype=float)
            if (apply_user or apply_source) and corpus_meta:
                mask = np.array([
                    1.0 if (
                        (not apply_user   or m.get("user_id", "default") == user_id) and
                        (not apply_source or m.get("source", "") == source_filter)
                    ) else 0.0
                    for m in corpus_meta
                ])
                bm25_scores = raw_scores * mask
            else:
                bm25_scores = raw_scores

        # 3. Build candidate pool
        candidates: dict = {}

        for doc, score in dense_raw:
            text = doc.page_content
            idx  = _corpus_idx(text, corpus_texts)
            candidates[text] = {
                "doc":   doc,
                "dense": float(score),
                "bm25":  float(bm25_scores[idx]) if idx >= 0 else 0.0,
            }

        if bm25 is not None:
            top_bm25_idxs = np.argsort(bm25_scores)[::-1][: k * 2]
            for idx in top_bm25_idxs:
                text = corpus_texts[idx]
                if text not in candidates:
                    candidates[text] = {
                        "doc":   Document(page_content=text, metadata=corpus_meta[idx]),
                        "dense": 0.0,
                        "bm25":  float(bm25_scores[idx]),
                    }

        if not candidates:
            return []

        # 4. Normalise and fuse with adaptive weights
        dense_vals = np.array([v["dense"] for v in candidates.values()])
        bm25_vals  = np.array([v["bm25"]  for v in candidates.values()])
        dense_norm = _minmax(dense_vals)
        bm25_norm  = _minmax(bm25_vals)

        for i, data in enumerate(candidates.values()):
            data["hybrid"] = dense_weight * dense_norm[i] + bm25_weight * bm25_norm[i]

        top_candidates = sorted(
            candidates.values(), key=lambda x: x["hybrid"], reverse=True
        )[:k]

        # 5. Cross-encoder reranking
        pairs = [(query, cand["doc"].page_content) for cand in top_candidates]
        rerank_scores: np.ndarray = self._reranker.predict(pairs)

        reranked = sorted(
            zip(top_candidates, rerank_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(item["doc"], score) for item, score in reranked[:top_k_after_rerank]]

    # ── Tool creation ───────────────────────────────────────────────────────────

    def create_rag_tools(self, web_search_enabled: bool = False) -> list:
        factory = self

        @tool
        def search_chunks(query: str, source_filter: str = "", retrieval_mode: str = "auto") -> str:
            """
            Search uploaded documents using hybrid retrieval (dense + BM25 + reranking).
            Use this to find information in any PDF that has been uploaded and ingested —
            policy documents, reports, budget speeches, RBI circulars, or any other document.
            Always call this before answering any factual question.

            Args:
                query: The search query in English. Be specific — include key terms, names,
                       and numbers from the user's question.
                source_filter: Optional PDF filename to restrict search to a single document.
                               Example: "budget_2024.pdf". Leave empty to search all documents.
                retrieval_mode: Search profile — "factual" for specific facts/numbers (BM25-heavy),
                                "conceptual" for explanations/frameworks (dense-heavy),
                                "comparative" for multi-doc comparisons (wide net),
                                "auto" (default) detects mode from the query automatically.

            Returns:
                Relevant document excerpts with source filenames.
            """
            if not factory._corpus_texts:
                return "NO_DOCUMENTS_INGESTED: Please upload and ingest PDF documents first."

            params = _get_retrieval_params(retrieval_mode, query)
            results = factory._hybrid_search(query, source_filter=source_filter, **params)

            if not results:
                return "NO_RELEVANT_CHUNKS: No relevant content found. Try rephrasing the query."

            return _format_search_results(results, factory._ingestion)

        tools = [search_chunks]
        if web_search_enabled:
            from .web_search import web_search
            tools.append(web_search)
        return tools


# ── Helpers ────────────────────────────────────────────────────────────────────

def _corpus_idx(text: str, corpus: List[str]) -> int:
    try:
        return corpus.index(text)
    except ValueError:
        return -1


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _format_search_results(results: List[Tuple[Document, float]], ingestion=None) -> str:
    seen_parents: set = set()
    parts = []
    chunk_num = 1

    for doc, score in results:
        source    = doc.metadata.get("source", "unknown")
        parent_id = doc.metadata.get("parent_id", "")

        if parent_id and ingestion:
            if parent_id in seen_parents:
                continue
            parent_data = ingestion.load_parent(parent_id)
            content = parent_data.get("content") or doc.page_content
            seen_parents.add(parent_id)
        else:
            content = doc.page_content

        parts.append(
            f"--- CHUNK {chunk_num} ---\n"
            f"Source: {source}\n"
            f"Relevance: {score:.3f}\n\n"
            f"{content.strip()}"
        )
        chunk_num += 1

    return "\n\n".join(parts) if parts else "NO_RELEVANT_CHUNKS: No content found."
