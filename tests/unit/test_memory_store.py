"""Unit tests for core/memory_store.py — UserMemoryStore."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from core.memory_store import UserMemoryStore, MemoryItem, MemoryList


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def store_and_vs(tmp_path):
    """UserMemoryStore backed by a mocked ChromaDB collection."""
    with patch("core.memory_store.Chroma") as MockChroma, \
         patch("core.memory_store.CHROMA_HOST", ""), \
         patch("core.memory_store.CHROMA_DB_PATH", str(tmp_path)):
        mock_vs = MagicMock()
        MockChroma.return_value = mock_vs
        embeddings = MagicMock()
        store = UserMemoryStore(embeddings)
        yield store, mock_vs


# ── MemoryItem / MemoryList models ─────────────────────────────────────────────

class TestMemoryModels:
    def test_memory_item_valid(self):
        item = MemoryItem(memory_type="topic_interest", content="Likes RBI policy", importance=2)
        assert item.memory_type == "topic_interest"
        assert item.importance == 2

    def test_memory_item_all_types(self):
        for t in ("topic_interest", "preference", "knowledge_gap", "doc_affinity"):
            item = MemoryItem(memory_type=t, content="test", importance=1)
            assert item.memory_type == t

    def test_memory_item_invalid_type_raises(self):
        with pytest.raises(Exception):
            MemoryItem(memory_type="unknown_type", content="x", importance=1)

    def test_memory_list_empty(self):
        ml = MemoryList(memories=[])
        assert ml.memories == []

    def test_memory_list_with_items(self):
        items = [MemoryItem(memory_type="preference", content="Concise answers", importance=1)]
        ml = MemoryList(memories=items)
        assert len(ml.memories) == 1


# ── fetch_relevant ─────────────────────────────────────────────────────────────

class TestFetchRelevant:
    def test_returns_content_list(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs.similarity_search.return_value = [
            MagicMock(page_content="User interested in RBI"),
            MagicMock(page_content="Prefers detailed answers"),
        ]
        results = store.fetch_relevant("alice", "RBI policy")
        assert results == ["User interested in RBI", "Prefers detailed answers"]

    def test_returns_empty_on_exception(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs.similarity_search.side_effect = Exception("ChromaDB error")
        results = store.fetch_relevant("alice", "some query")
        assert results == []

    def test_passes_user_filter(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs.similarity_search.return_value = []
        store.fetch_relevant("bob", "query", k=2)
        call_kwargs = mock_vs.similarity_search.call_args
        assert call_kwargs[1]["filter"] == {"user_id": "bob"}
        assert call_kwargs[1]["k"] == 2

    def test_returns_empty_list_when_no_memories(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs.similarity_search.return_value = []
        assert store.fetch_relevant("alice", "query") == []


# ── extract_and_save ───────────────────────────────────────────────────────────

class TestExtractAndSave:
    def _make_llm(self, memories):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(
            invoke=MagicMock(return_value=MemoryList(memories=memories))
        )
        return mock_llm

    def test_saves_extracted_memories(self, store_and_vs):
        store, mock_vs = store_and_vs
        llm = self._make_llm([
            MemoryItem(memory_type="topic_interest", content="Interested in RBI", importance=2),
        ])
        msgs = [HumanMessage(content="Tell me about RBI"), AIMessage(content="RBI is...")]
        store.extract_and_save("alice", msgs, llm)
        mock_vs.add_texts.assert_called_once()
        _, kwargs = mock_vs.add_texts.call_args
        assert len(kwargs["texts"]) == 1
        assert kwargs["texts"][0] == "Interested in RBI"

    def test_noop_when_too_few_messages(self, store_and_vs):
        store, mock_vs = store_and_vs
        store.extract_and_save("alice", [], MagicMock())
        mock_vs.add_texts.assert_not_called()

    def test_noop_when_only_one_message(self, store_and_vs):
        store, mock_vs = store_and_vs
        store.extract_and_save("alice", [HumanMessage(content="hi")], MagicMock())
        mock_vs.add_texts.assert_not_called()

    def test_noop_when_empty_memory_list(self, store_and_vs):
        store, mock_vs = store_and_vs
        llm = self._make_llm([])
        msgs = [HumanMessage(content="q"), AIMessage(content="a")]
        store.extract_and_save("alice", msgs, llm)
        mock_vs.add_texts.assert_not_called()

    def test_survives_llm_failure(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(
            invoke=MagicMock(side_effect=Exception("LLM error"))
        )
        msgs = [HumanMessage(content="q"), AIMessage(content="a")]
        store.extract_and_save("alice", msgs, mock_llm)  # must not raise
        mock_vs.add_texts.assert_not_called()

    def test_id_is_deterministic(self, store_and_vs):
        """Same content should produce the same ChromaDB ID (deduplication)."""
        store, mock_vs = store_and_vs
        llm = self._make_llm([
            MemoryItem(memory_type="preference", content="Concise answers", importance=1),
        ])
        msgs = [HumanMessage(content="q"), AIMessage(content="a")]
        store.extract_and_save("alice", msgs, llm)
        store.extract_and_save("alice", msgs, llm)

        all_ids = [call[1]["ids"] for call in mock_vs.add_texts.call_args_list]
        assert all_ids[0] == all_ids[1]  # same content → same ID

    def test_metadata_contains_user_id(self, store_and_vs):
        store, mock_vs = store_and_vs
        llm = self._make_llm([
            MemoryItem(memory_type="doc_affinity", content="Uses budget_2024.pdf", importance=3),
        ])
        msgs = [HumanMessage(content="q"), AIMessage(content="a")]
        store.extract_and_save("charlie", msgs, llm)
        _, kwargs = mock_vs.add_texts.call_args
        assert kwargs["metadatas"][0]["user_id"] == "charlie"


# ── get_all ────────────────────────────────────────────────────────────────────

class TestGetAll:
    def test_returns_memories_sorted_by_importance(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs._collection = MagicMock()
        mock_vs._collection.get.return_value = {
            "documents": ["Low priority", "High priority"],
            "metadatas": [
                {"user_id": "alice", "memory_type": "preference", "importance": 1},
                {"user_id": "alice", "memory_type": "topic_interest", "importance": 3},
            ],
        }
        results = store.get_all("alice")
        assert len(results) == 2
        assert results[0]["importance"] == 3  # sorted descending
        assert results[1]["importance"] == 1

    def test_returns_empty_on_exception(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs._collection = MagicMock()
        mock_vs._collection.get.side_effect = Exception("error")
        results = store.get_all("alice")
        assert results == []

    def test_returns_empty_when_no_documents(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs._collection = MagicMock()
        mock_vs._collection.get.return_value = {"documents": [], "metadatas": []}
        assert store.get_all("alice") == []

    def test_content_field_present(self, store_and_vs):
        store, mock_vs = store_and_vs
        mock_vs._collection = MagicMock()
        mock_vs._collection.get.return_value = {
            "documents": ["Interested in GST"],
            "metadatas": [{"user_id": "alice", "memory_type": "topic_interest", "importance": 2}],
        }
        results = store.get_all("alice")
        assert results[0]["content"] == "Interested in GST"
        assert results[0]["memory_type"] == "topic_interest"
