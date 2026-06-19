"""Unit tests for core/graph.py — routing logic, token estimation, node error paths."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.graph import (
    _estimate_tokens,
    route_after_rewrite,
    route_after_orchestrator,
    collect_answer,
    AgentState,
    State,
)


class TestEstimateTokens:
    def test_empty_list_returns_zero(self):
        assert _estimate_tokens([]) == 0

    def test_single_message_returns_nonzero(self):
        msgs = [HumanMessage(content="What is India's GDP?")]
        assert _estimate_tokens(msgs) > 0

    def test_longer_message_has_more_tokens(self):
        short = [HumanMessage(content="Hi")]
        long  = [HumanMessage(content="What is the full text of the RBI monetary policy circular? " * 20)]
        assert _estimate_tokens(long) > _estimate_tokens(short)


class TestRouteAfterRewrite:
    def test_routes_to_clarification_when_unclear(self):
        state = {"question_is_clear": False, "messages": []}
        assert route_after_rewrite(state) == "request_clarification"

    def test_routes_to_route_query_when_clear(self):
        state = {"question_is_clear": True, "messages": []}
        assert route_after_rewrite(state) == "route_query"

    def test_missing_key_defaults_to_clarification(self):
        state = {"messages": []}
        assert route_after_rewrite(state) == "request_clarification"


class TestRouteAfterOrchestrator:
    def _state(self, messages, iterations=0, tool_calls_count=0):
        last_msg = messages[-1] if messages else AIMessage(content="done")
        return {
            "messages": messages,
            "iteration_count": iterations,
            "tool_call_count": tool_calls_count,
        }

    def test_routes_to_tools_when_tool_calls_present(self):
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "search_chunks", "args": {"query": "RBI"}, "id": "1"}]
        state = self._state([msg])
        assert route_after_orchestrator(state) == "tools"

    def test_routes_to_collect_answer_when_no_tool_calls(self):
        msg = AIMessage(content="Here is my answer.")
        msg.tool_calls = []
        state = self._state([msg])
        assert route_after_orchestrator(state) == "collect_answer"

    def test_routes_to_fallback_when_max_iterations_reached(self):
        from core.config import MAX_ITERATIONS
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "search_chunks", "args": {}, "id": "1"}]
        state = {
            "messages": [msg],
            "iteration_count": MAX_ITERATIONS,
            "tool_call_count": 0,
        }
        assert route_after_orchestrator(state) == "fallback_response"


class TestCollectAnswer:
    def test_extracts_last_ai_message(self):
        msg = AIMessage(content="Final answer text.")
        msg.tool_calls = []
        state = {"messages": [HumanMessage(content="Q"), msg], "question": "Q", "question_index": 0}
        result = collect_answer(state)
        assert result["final_answer"] == "Final answer text."
        assert result["agent_answers"][0]["answer"] == "Final answer text."

    def test_fallback_when_last_message_has_tool_calls(self):
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "search", "args": {}, "id": "1"}]
        state = {"messages": [msg], "question": "Q", "question_index": 0}
        result = collect_answer(state)
        assert "Unable to generate" in result["final_answer"]


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("user1") is True

    def test_blocks_requests_over_limit(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False

    def test_different_users_have_independent_windows(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False
        assert limiter.is_allowed("user2") is True
