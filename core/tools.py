"""
RAG search tool used by the agent subgraph.

Interview concept — Hybrid Search pipeline:

  Query
   ├─► Dense search (ChromaDB)
   │     ChromaDB stores child chunks as float vectors (all-MiniLM-L6-v2).
   │     Cosine similarity finds semantically close chunks.
   │     Good at: meaning-based matches ("government farm subsidy" ≈ "PM-KISAN")
   │
   ├─► Sparse search (BM25)
   │     Classic keyword-frequency algorithm (Best Match 25).
   │     Good at: exact term matches ("RBI repo rate", scheme names, years).
   │
   ├─► Score fusion (0.6 dense + 0.4 BM25, both normalised to [0,1])
   │     Neither alone is perfect — hybrid beats both individually.
   │
   └─► Reranking (CrossEncoder: ms-marco-MiniLM-L-6-v2)
         Bi-encoder (ChromaDB) is fast but shallow.
         Cross-encoder reads query+document together → much better relevance score.
         We rerank the fused top-K and keep only TOP_K_AFTER_RERANK results.
"""

import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import (
    DEFAULT_K, DENSE_WEIGHT, BM25_WEIGHT, TOP_K_AFTER_RERANK, RERANKER_MODEL,
)
from .ingestion import Ingestion


class ToolFactory:
    """
    Builds LangChain tools that the RAG agent can call.

    Created once at application startup; shared across all agent invocations.

    Usage:
        factory = ToolFactory(ingestion)
        tools   = factory.create_rag_tools()   # pass to agent graph
    """

    def __init__(self, ingestion: Ingestion):
        self._vectorstore: Chroma = ingestion.get_vectorstore()
        self._ingestion = ingestion

        print("⏳ Loading CrossEncoder reranker...")
        self._reranker = CrossEncoder(RERANKER_MODEL)
        print("✓ Reranker ready.")

        # BM25 index is built lazily — rebuilt whenever docs are ingested
        self._bm25: BM25Okapi | None = None
        self._corpus_texts: List[str] = []
        self._corpus_meta: List[dict] = []
        self._build_bm25_index()

    # ── BM25 index management ───────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """
        Pull all child-chunk texts from ChromaDB and build a BM25 index.
        Called at startup and after each new ingestion.
        """
        pairs = self._ingestion.get_all_texts_and_meta()
        if not pairs:
            print("⚠  No documents in ChromaDB yet — BM25 index is empty.")
            self._bm25 = None
            return

        self._corpus_texts = [text for text, _ in pairs]
        self._corpus_meta  = [meta for _, meta in pairs]

        tokenized = [t.lower().split() for t in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        print(f"✓ BM25 index built: {len(self._corpus_texts)} chunks.")

    def rebuild_bm25(self) -> None:
        """Call this after ingesting new documents to keep BM25 in sync."""
        self._build_bm25_index()

    # ── Hybrid search core ──────────────────────────────────────────────────

    def _hybrid_search(self, query: str, k: int = DEFAULT_K) -> List[Tuple[Document, float]]:
        """
        Fuse dense + sparse results then rerank.

        Returns:
            List of (Document, rerank_score) sorted by rerank score desc,
            capped at TOP_K_AFTER_RERANK.
        """
        # ── 1. Dense retrieval ────────────────────────────────────────────
        # Returns (doc, relevance_score) in [0,1]; higher = better
        dense_raw: List[Tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_relevance_scores(query, k=k * 2)
        )

        # ── 2. BM25 retrieval ─────────────────────────────────────────────
        bm25_scores = np.zeros(len(self._corpus_texts))
        if self._bm25 is not None:
            bm25_scores = np.array(
                self._bm25.get_scores(query.lower().split()), dtype=float
            )

        # ── 3. Build candidate pool ───────────────────────────────────────
        # Key: page_content  →  {doc, dense_score, bm25_score}
        candidates: dict = {}

        for doc, score in dense_raw:
            text = doc.page_content
            idx = self._text_to_corpus_idx(text)
            candidates[text] = {
                "doc":   doc,
                "dense": float(score),
                "bm25":  float(bm25_scores[idx]) if idx >= 0 else 0.0,
            }

        # Add BM25-top-K docs that dense missed
        if self._bm25 is not None:
            top_bm25_idxs = np.argsort(bm25_scores)[::-1][: k * 2]
            for idx in top_bm25_idxs:
                text = self._corpus_texts[idx]
                if text not in candidates:
                    candidates[text] = {
                        "doc":   Document(
                            page_content=text,
                            metadata=self._corpus_meta[idx],
                        ),
                        "dense": 0.0,
                        "bm25":  float(bm25_scores[idx]),
                    }

        if not candidates:
            return []

        # ── 4. Normalise both scores to [0, 1] and fuse ───────────────────
        dense_vals = np.array([v["dense"] for v in candidates.values()])
        bm25_vals  = np.array([v["bm25"]  for v in candidates.values()])

        dense_norm = _minmax(dense_vals)
        bm25_norm  = _minmax(bm25_vals)

        for i, (text, data) in enumerate(candidates.items()):
            data["hybrid"] = DENSE_WEIGHT * dense_norm[i] + BM25_WEIGHT * bm25_norm[i]

        # Sort by hybrid score, keep top K for reranking
        top_candidates = sorted(
            candidates.values(), key=lambda x: x["hybrid"], reverse=True
        )[:k]

        # ── 5. Cross-encoder reranking ────────────────────────────────────
        pairs = [(query, cand["doc"].page_content) for cand in top_candidates]
        rerank_scores: np.ndarray = self._reranker.predict(pairs)

        reranked = sorted(
            zip(top_candidates, rerank_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(item["doc"], score) for item, score in reranked[:TOP_K_AFTER_RERANK]]

    def _text_to_corpus_idx(self, text: str) -> int:
        """Return position of text in BM25 corpus, or -1 if not found."""
        try:
            return self._corpus_texts.index(text)
        except ValueError:
            return -1

    # ── Tool creation ───────────────────────────────────────────────────────

    def create_rag_tools(self) -> list:
        """
        Return the list of LangChain tools for the RAG agent subgraph.
        Currently: [search_chunks]
        """
        factory = self  # captured in closure

        @tool
        def search_chunks(query: str) -> str:
            """
            Search uploaded documents using hybrid retrieval (dense + BM25 + reranking).
            Use this to find information in any PDF that has been uploaded and ingested —
            policy documents, reports, budget speeches, RBI circulars, or any other document.
            Always call this before answering any factual question.

            Args:
                query: The search query in English. Be specific — include key terms, names,
                       and numbers from the user's question.

            Returns:
                Relevant document excerpts with source filenames.
            """
            if not factory._corpus_texts:
                return "NO_DOCUMENTS_INGESTED: Please upload and ingest PDF documents first."

            results = factory._hybrid_search(query)

            if not results:
                return "NO_RELEVANT_CHUNKS: No relevant content found. Try rephrasing the query."

            return _format_search_results(results, factory._ingestion)

        return [search_chunks]


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. Handles all-zero or all-same arrays."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _format_search_results(results: List[Tuple[Document, float]], ingestion=None) -> str:
    """
    Format retrieved (doc, score) pairs for the LLM.
    Loads the full parent chunk when available — much richer context than the 500-char child.
    Deduplicates by parent_id so the same section isn't repeated.
    """
    seen_parents: set = set()
    parts = []
    chunk_num = 1

    for doc, score in results:
        source    = doc.metadata.get("source", "unknown")
        parent_id = doc.metadata.get("parent_id", "")

        if parent_id and ingestion:
            if parent_id in seen_parents:
                continue  # already included this parent section
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
