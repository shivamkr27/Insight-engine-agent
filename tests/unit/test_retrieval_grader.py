"""Unit tests for core/retrieval_grader.py — RetrievalGrader."""

import pytest
from unittest.mock import MagicMock
from core.retrieval_grader import RetrievalGrader, GradeResult


def _make_grader(return_value=None, side_effect=None):
    """Build a RetrievalGrader with a mocked LLM."""
    mock_llm = MagicMock()
    invoke_mock = MagicMock()
    if side_effect:
        invoke_mock.side_effect = side_effect
    else:
        invoke_mock.return_value = return_value
    mock_llm.with_structured_output.return_value = MagicMock(invoke=invoke_mock)
    return RetrievalGrader(mock_llm), invoke_mock


class TestGradeResult:
    def test_relevant_grade_accepted(self):
        g = GradeResult(grade="relevant", reason="directly answers")
        assert g.grade == "relevant"

    def test_partial_grade_accepted(self):
        g = GradeResult(grade="partial", reason="tangential")
        assert g.grade == "partial"

    def test_irrelevant_grade_accepted(self):
        g = GradeResult(grade="irrelevant", reason="off-topic")
        assert g.grade == "irrelevant"

    def test_invalid_grade_raises(self):
        with pytest.raises(Exception):
            GradeResult(grade="unknown", reason="bad")

    def test_reason_is_string(self):
        g = GradeResult(grade="relevant", reason="ok")
        assert isinstance(g.reason, str)


class TestRetrievalGrader:
    def test_grade_returns_relevant(self):
        grader, _ = _make_grader(GradeResult(grade="relevant", reason="yes"))
        result = grader.grade("what is repo rate", "The repo rate is 6.5%.")
        assert result.grade == "relevant"

    def test_grade_returns_irrelevant(self):
        grader, _ = _make_grader(GradeResult(grade="irrelevant", reason="off-topic"))
        result = grader.grade("PM-KISAN benefits", "This document covers fiscal policy.")
        assert result.grade == "irrelevant"

    def test_grade_returns_partial_on_llm_failure(self):
        grader, _ = _make_grader(side_effect=Exception("API timeout"))
        result = grader.grade("any question", "any content")
        assert result.grade == "partial"
        assert isinstance(result.reason, str)

    def test_grade_truncates_long_content(self):
        """Grader should not pass >2000 chars to the LLM."""
        received_content = {}

        def capture_invoke(msgs):
            received_content["text"] = msgs[1].content
            return GradeResult(grade="relevant", reason="ok")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(invoke=capture_invoke)
        grader = RetrievalGrader(mock_llm)
        grader.grade("question", "x" * 5000)

        assert len(received_content["text"]) < 4000  # truncation happened

    def test_grade_passes_question_to_llm(self):
        received = {}

        def capture(msgs):
            received["human"] = msgs[1].content
            return GradeResult(grade="partial", reason="ok")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(invoke=capture)
        grader = RetrievalGrader(mock_llm)
        grader.grade("what is FRBM", "some content")

        assert "what is FRBM" in received["human"]

    def test_grade_empty_content_returns_result(self):
        grader, _ = _make_grader(GradeResult(grade="irrelevant", reason="empty"))
        result = grader.grade("question", "")
        assert result.grade in {"relevant", "partial", "irrelevant"}
