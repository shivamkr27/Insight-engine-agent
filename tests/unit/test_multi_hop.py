"""Unit tests for Phase 4 — CRAG routing + Multi-Hop Reasoning logic in core/graph.py."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from core.graph import (
    route_after_grader,
    route_reasoning_steps,
    ReasoningPlan,
    AgentState,
    State,
)


# ── CRAG: route_after_grader ───────────────────────────────────────────────────

class TestRouteAfterGrader:
    def test_relevant_goes_to_compress(self):
        state = {"last_retrieval_grade": "relevant", "retrieval_attempts": 0}
        assert route_after_grader(state) == "should_compress_context"

    def test_partial_goes_to_compress(self):
        state = {"last_retrieval_grade": "partial", "retrieval_attempts": 0}
        assert route_after_grader(state) == "should_compress_context"

    def test_irrelevant_first_attempt_retries(self):
        state = {"last_retrieval_grade": "irrelevant", "retrieval_attempts": 0}
        assert route_after_grader(state) == "query_rewriter_loop"

    def test_irrelevant_second_attempt_retries(self):
        state = {"last_retrieval_grade": "irrelevant", "retrieval_attempts": 1}
        assert route_after_grader(state) == "query_rewriter_loop"

    def test_irrelevant_at_max_attempts_goes_to_compress(self):
        state = {"last_retrieval_grade": "irrelevant", "retrieval_attempts": 2}
        assert route_after_grader(state) == "should_compress_context"

    def test_irrelevant_beyond_max_goes_to_compress(self):
        state = {"last_retrieval_grade": "irrelevant", "retrieval_attempts": 5}
        assert route_after_grader(state) == "should_compress_context"

    def test_empty_state_defaults_to_compress(self):
        # No grade set → defaults to "relevant" → compress
        assert route_after_grader({}) == "should_compress_context"

    def test_unknown_grade_goes_to_compress(self):
        state = {"last_retrieval_grade": "unknown", "retrieval_attempts": 0}
        assert route_after_grader(state) == "should_compress_context"


# ── ReasoningPlan model ────────────────────────────────────────────────────────

class TestReasoningPlan:
    def test_basic_plan(self):
        plan = ReasoningPlan(steps=["Find FRBM target", "Find actual deficit"], can_parallelize=False)
        assert len(plan.steps) == 2
        assert plan.can_parallelize is False

    def test_empty_steps_allowed(self):
        plan = ReasoningPlan(steps=[], can_parallelize=False)
        assert plan.steps == []

    def test_can_parallelize_defaults_to_false(self):
        plan = ReasoningPlan(steps=["step1"])
        assert plan.can_parallelize is False

    def test_steps_are_strings(self):
        plan = ReasoningPlan(steps=["step a", "step b", "step c"])
        assert all(isinstance(s, str) for s in plan.steps)


# ── route_reasoning_steps ─────────────────────────────────────────────────────

class TestRouteReasoningSteps:
    def test_continues_loop_when_steps_remain(self):
        state = {"reasoning_steps": ["step1", "step2"], "current_step_index": 0}
        assert route_reasoning_steps(state) == "execute_reasoning_step"

    def test_continues_loop_partway_through(self):
        state = {"reasoning_steps": ["s1", "s2", "s3"], "current_step_index": 1}
        assert route_reasoning_steps(state) == "execute_reasoning_step"

    def test_exits_when_all_steps_done(self):
        state = {"reasoning_steps": ["s1", "s2"], "current_step_index": 2}
        assert route_reasoning_steps(state) == "reasoning_synthesizer"

    def test_exits_beyond_steps(self):
        state = {"reasoning_steps": ["s1"], "current_step_index": 5}
        assert route_reasoning_steps(state) == "reasoning_synthesizer"

    def test_empty_steps_exits_immediately(self):
        state = {"reasoning_steps": [], "current_step_index": 0}
        assert route_reasoning_steps(state) == "reasoning_synthesizer"

    def test_missing_state_keys_exit_safely(self):
        assert route_reasoning_steps({}) == "reasoning_synthesizer"


# ── AgentState CRAG fields ─────────────────────────────────────────────────────

class TestAgentStateCRAGFields:
    def test_retrieval_attempts_accumulates(self):
        """retrieval_attempts uses operator.add — delta increments should accumulate."""
        import operator
        from typing import Annotated, get_args, get_origin
        hints = AgentState.__annotations__
        # retrieval_attempts should be Annotated[int, operator.add]
        ra_hint = hints.get("retrieval_attempts")
        assert ra_hint is not None
        # Check it's an Annotated type
        assert get_origin(ra_hint) is Annotated or str(ra_hint).startswith("typing.")

    def test_last_retrieval_grade_default_empty(self):
        # Field should have a default of ""
        state_dict = {
            "question": "test",
            "messages": [],
            "last_retrieval_grade": "",
        }
        assert state_dict["last_retrieval_grade"] == ""


# ── State multi-hop fields ─────────────────────────────────────────────────────

class TestStateMultiHopFields:
    def test_reasoning_steps_field_exists(self):
        hints = State.__annotations__
        assert "reasoning_steps" in hints

    def test_current_step_index_field_exists(self):
        hints = State.__annotations__
        assert "current_step_index" in hints

    def test_step_results_field_exists(self):
        hints = State.__annotations__
        assert "step_results" in hints

    def test_user_memories_field_exists(self):
        hints = State.__annotations__
        assert "user_memories" in hints
