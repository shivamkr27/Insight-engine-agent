"""
Persistent semantic memory for the India Policy Intelligence Agent.

Stores memory-worthy facts extracted from conversations in a dedicated ChromaDB
collection (user_memories). On each new conversation, the most topically relevant
memories are fetched and injected into the orchestrator's system prompt so future
responses are personalised without re-asking the user.

Memory types:
  topic_interest — subjects the user frequently asks about
  preference     — how they like answers (concise/detailed/Hindi)
  knowledge_gap  — topics they struggled with or asked about multiple times
  doc_affinity   — which documents they rely on most
"""

import hashlib
from datetime import datetime, timezone
from typing import List, Literal

from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

from .config import CHROMA_DB_PATH, CHROMA_HOST, CHROMA_PORT, MEMORY_COLLECTION
from .logging_config import get_logger

logger = get_logger(__name__)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    memory_type: Literal["topic_interest", "preference", "knowledge_gap", "doc_affinity"]
    content: str
    importance: int  # 1-3


class MemoryList(BaseModel):
    memories: List[MemoryItem]


_EXTRACT_PROMPT = """You extract memory-worthy facts from a conversation for a government policy assistant.

Extract 0-3 short facts worth remembering about this user:
  topic_interest — subjects they frequently ask about (e.g. "Frequently asks about RBI monetary policy")
  preference     — answer style preferences (e.g. "Prefers concise answers with bullet points")
  knowledge_gap  — topics they struggled with or had to ask about multiple times
  doc_affinity   — specific documents they relied on most (e.g. "Frequently references budget_2024.pdf")

Rules:
- Only extract facts that would actually help personalise future responses
- Do NOT extract what was answered — only the user's behaviour and preferences
- If nothing is memory-worthy (very short or generic conversation), return an empty memories list
- Keep each content field under 15 words — concise facts only"""


class UserMemoryStore:
    """Stores and retrieves per-user semantic memories using a dedicated ChromaDB collection."""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._vs = self._init_collection()

    def _init_collection(self) -> Chroma:
        if CHROMA_HOST:
            import chromadb
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            return Chroma(
                client=client,
                collection_name=MEMORY_COLLECTION,
                embedding_function=self._embeddings,
            )
        return Chroma(
            collection_name=MEMORY_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_DB_PATH,
        )

    def extract_and_save(self, user_id: str, conversation_messages: list, llm) -> None:
        """Extract memories from conversation messages and persist to ChromaDB. Fire-and-forget safe."""
        if len(conversation_messages) < 2:
            return

        conv_text = ""
        for m in conversation_messages[-12:]:  # last ~6 exchanges
            if isinstance(m, HumanMessage):
                conv_text += f"User: {m.content}\n"
            elif isinstance(m, AIMessage) and m.content:
                conv_text += f"Assistant: {m.content[:300]}\n"

        if not conv_text.strip():
            return

        try:
            memory_llm = llm.with_structured_output(MemoryList)
            result: MemoryList = memory_llm.invoke([
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user",   "content": f"Conversation:\n{conv_text}"},
            ])

            if not result.memories:
                return

            docs, metas, ids = [], [], []
            now = datetime.now(timezone.utc).isoformat()
            for item in result.memories:
                doc_id = (
                    f"{user_id}_{item.memory_type}_"
                    + hashlib.md5(item.content.encode()).hexdigest()[:8]
                )
                docs.append(item.content)
                metas.append({
                    "user_id":     user_id,
                    "memory_type": item.memory_type,
                    "importance":  item.importance,
                    "timestamp":   now,
                })
                ids.append(doc_id)

            self._vs.add_texts(texts=docs, metadatas=metas, ids=ids)
            logger.info(f"Saved {len(docs)} memories for user={user_id}")

        except Exception as exc:
            logger.warning(f"extract_and_save failed for user={user_id}: {exc}")

    def fetch_relevant(self, user_id: str, query: str, k: int = 3) -> List[str]:
        """Semantic search over this user's stored memories."""
        try:
            results = self._vs.similarity_search(
                query, k=k,
                filter={"user_id": user_id},
            )
            return [doc.page_content for doc in results]
        except Exception as exc:
            logger.warning(f"fetch_relevant failed for user={user_id}: {exc}")
            return []

    def get_all(self, user_id: str) -> List[dict]:
        """Return all stored memories for a user (for the 'what do you know about me?' command)."""
        try:
            result = self._vs._collection.get(where={"user_id": user_id})
            memories = []
            for i, doc in enumerate(result.get("documents") or []):
                meta = (result.get("metadatas") or [{}])[i]
                memories.append({
                    "content":     doc,
                    "memory_type": meta.get("memory_type", "unknown"),
                    "importance":  meta.get("importance", 1),
                    "timestamp":   meta.get("timestamp", ""),
                })
            return sorted(memories, key=lambda x: x.get("importance", 1), reverse=True)
        except Exception as exc:
            logger.warning(f"get_all failed for user={user_id}: {exc}")
            return []
