"""Unit tests for core/judge.py — badge logic and result caching."""

import pytest
from core.judge import HallucinationJudge


class TestBadge:
    def test_score_1_is_verified(self):
        assert HallucinationJudge._badge(1) == "🟢 Verified"

    def test_score_2_is_verified(self):
        assert HallucinationJudge._badge(2) == "🟢 Verified"

    def test_score_3_is_review(self):
        assert HallucinationJudge._badge(3) == "🟡 Review"

    def test_score_4_is_warning(self):
        assert HallucinationJudge._badge(4) == "🔴 Warning"

    def test_score_5_is_warning(self):
        assert HallucinationJudge._badge(5) == "🔴 Warning"


class TestCaching:
    def test_cache_hit_skips_llm(self, mock_llm):
        judge = HallucinationJudge()

        # First call populates cache via structured output mock
        from pydantic import BaseModel, Field
        from core.judge import JudgeResult

        mock_result = JudgeResult(score=1, reason="Grounded.", is_safe=True)
        mock_llm.invoke.return_value = mock_result

        first = judge.score("Q", "context", "answer", mock_llm)
        second = judge.score("Q", "context", "answer", mock_llm)

        # LLM invoked only once; second call should be cached
        assert mock_llm.invoke.call_count == 1
        assert first == second

    def test_different_inputs_not_cached_together(self, mock_llm):
        judge = HallucinationJudge()
        from core.judge import JudgeResult

        mock_result = JudgeResult(score=2, reason="Mostly grounded.", is_safe=True)
        mock_llm.invoke.return_value = mock_result

        judge.score("Q1", "context", "answer", mock_llm)
        judge.score("Q2", "context", "answer", mock_llm)

        assert mock_llm.invoke.call_count == 2

    def test_judge_failure_returns_safe_default(self, mock_llm):
        judge = HallucinationJudge()
        mock_llm.invoke.side_effect = RuntimeError("API down")

        result = judge.score("Q", "ctx", "answer", mock_llm)
        assert result["is_safe"] is True
        assert result["badge"] == "🟢 Verified"


class TestCacheKey:
    def test_same_inputs_same_key(self):
        judge = HallucinationJudge()
        k1 = judge._cache_key("Q", "ctx", "ans")
        k2 = judge._cache_key("Q", "ctx", "ans")
        assert k1 == k2

    def test_different_question_different_key(self):
        judge = HallucinationJudge()
        k1 = judge._cache_key("Q1", "ctx", "ans")
        k2 = judge._cache_key("Q2", "ctx", "ans")
        assert k1 != k2
