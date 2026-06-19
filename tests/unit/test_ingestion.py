"""Unit tests for core/ingestion.py — _ParentStore SQLite backend."""

import json
import pytest
from core.ingestion import _ParentStore


class TestParentStoreSQLite:
    @pytest.fixture
    def store(self, tmp_path):
        return _ParentStore(db_path=str(tmp_path / "parents.db"))

    def test_save_and_load_roundtrip(self, store):
        store.save("doc_parent_0", "Full parent text here.", {"source": "doc.pdf"})
        result = store.load("doc_parent_0")
        assert result["content"] == "Full parent text here."
        assert result["metadata"]["source"] == "doc.pdf"

    def test_load_missing_returns_empty(self, store):
        assert store.load("nonexistent_id") == {}

    def test_all_sources_returns_unique_filenames(self, store):
        store.save("doc1_parent_0", "text", {"source": "rbi_report.pdf"})
        store.save("doc1_parent_1", "text", {"source": "rbi_report.pdf"})
        store.save("doc2_parent_0", "text", {"source": "budget.pdf"})

        sources = store.all_sources()
        assert set(sources) == {"rbi_report.pdf", "budget.pdf"}

    def test_save_overwrites_existing(self, store):
        store.save("doc_parent_0", "original", {"source": "a.pdf"})
        store.save("doc_parent_0", "updated", {"source": "a.pdf"})
        result = store.load("doc_parent_0")
        assert result["content"] == "updated"

    def test_clear_removes_all_rows(self, store):
        store.save("p1", "text", {"source": "a.pdf"})
        store.save("p2", "text", {"source": "b.pdf"})
        store.clear()
        assert store.all_sources() == []
        assert store.load("p1") == {}

    def test_metadata_is_preserved_as_dict(self, store):
        meta = {"source": "doc.pdf", "page": 3, "heading": "Section 2"}
        store.save("doc_parent_0", "content", meta)
        result = store.load("doc_parent_0")
        assert result["metadata"] == meta

    def test_all_sources_empty_on_fresh_store(self, store):
        assert store.all_sources() == []

    def test_user_id_isolation_all_sources(self, store):
        store.save("a_parent_0", "text", {"source": "alice.pdf"}, user_id="alice")
        store.save("b_parent_0", "text", {"source": "bob.pdf"},   user_id="bob")
        assert store.all_sources(user_id="alice") == ["alice.pdf"]
        assert store.all_sources(user_id="bob")   == ["bob.pdf"]

    def test_default_user_sees_all_sources(self, store):
        store.save("a_parent_0", "text", {"source": "doc1.pdf"}, user_id="default")
        store.save("b_parent_0", "text", {"source": "doc2.pdf"}, user_id="default")
        # "default" user_id → no filter applied
        assert set(store.all_sources(user_id="default")) == {"doc1.pdf", "doc2.pdf"}

    def test_save_with_user_id_roundtrip(self, store):
        store.save("p1", "content", {"source": "doc.pdf"}, user_id="carol")
        result = store.load("p1")
        assert result["content"] == "content"

    def test_clear_per_user(self, store):
        store.save("a_p0", "text", {"source": "a.pdf"}, user_id="alice")
        store.save("b_p0", "text", {"source": "b.pdf"}, user_id="bob")
        store.clear(user_id="alice")
        assert store.all_sources(user_id="alice") == []
        assert store.all_sources(user_id="bob")   == ["b.pdf"]
