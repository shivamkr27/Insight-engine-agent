"""Unit tests for core/tools.py — minmax normalisation and result formatting."""

import numpy as np
import pytest
from langchain_core.documents import Document

from core.tools import _minmax, _format_search_results, _corpus_idx


class TestMinmax:
    def test_normalises_to_zero_one(self):
        arr = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        result = _minmax(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_all_zeros_returns_zeros(self):
        arr = np.array([0.0, 0.0, 0.0])
        result = _minmax(arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_all_same_value_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = _minmax(arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_preserves_relative_order(self):
        arr = np.array([1.0, 3.0, 5.0])
        result = _minmax(arr)
        assert result[0] < result[1] < result[2]


class TestCorpusIdx:
    def test_found(self):
        corpus = ["alpha", "beta", "gamma"]
        assert _corpus_idx("beta", corpus) == 1

    def test_not_found_returns_minus_one(self):
        corpus = ["alpha", "beta"]
        assert _corpus_idx("delta", corpus) == -1


class TestFormatSearchResults:
    def _make_doc(self, content, source="test.pdf", parent_id="p1"):
        return Document(
            page_content=content,
            metadata={"source": source, "parent_id": parent_id},
        )

    def test_includes_source_filename(self):
        results = [(self._make_doc("some text", source="rbi.pdf"), 0.9)]
        output = _format_search_results(results)
        assert "rbi.pdf" in output

    def test_includes_relevance_score(self):
        results = [(self._make_doc("some text"), 0.854)]
        output = _format_search_results(results)
        assert "0.854" in output

    def test_deduplicates_same_parent(self):
        from unittest.mock import MagicMock
        # Deduplication requires ingestion so the parent_id path is taken
        ingestion = MagicMock()
        ingestion.load_parent.return_value = {"content": "Full parent content"}

        doc1 = self._make_doc("child 1", parent_id="shared_parent")
        doc2 = self._make_doc("child 2", parent_id="shared_parent")
        results = [(doc1, 0.9), (doc2, 0.8)]
        output = _format_search_results(results, ingestion=ingestion)
        # Only CHUNK 1 should appear; CHUNK 2 shares the same parent_id
        assert "CHUNK 1" in output
        assert "CHUNK 2" not in output

    def test_empty_results_returns_no_chunks_message(self):
        output = _format_search_results([])
        assert "NO_RELEVANT_CHUNKS" in output

    def test_loads_parent_content_when_ingestion_provided(self):
        from unittest.mock import MagicMock
        ingestion = MagicMock()
        ingestion.load_parent.return_value = {"content": "Full parent text from DB"}

        doc = self._make_doc("child content", parent_id="p99")
        output = _format_search_results([(doc, 0.9)], ingestion=ingestion)
        assert "Full parent text from DB" in output
        ingestion.load_parent.assert_called_once_with("p99")
