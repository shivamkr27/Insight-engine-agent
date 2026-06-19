"""Unit tests for adaptive retrieval — core/tools.py _classify_query and _get_retrieval_params."""

import pytest
from core.tools import _classify_query, _get_retrieval_params
from core.config import RETRIEVAL_PROFILES


class TestClassifyQuery:
    # ── Factual signals ────────────────────────────────────────────────────
    def test_factual_kya_hai(self):
        assert _classify_query("repo rate kya hai") == "factual"

    def test_factual_kya_tha(self):
        assert _classify_query("2024 mein repo rate kya tha") == "factual"

    def test_factual_amount(self):
        assert _classify_query("PM-KISAN amount kya hai") == "factual"

    def test_factual_how_much(self):
        assert _classify_query("how much is the fiscal deficit target") == "factual"

    def test_factual_kitna(self):
        assert _classify_query("kitna allocation mila hai") == "factual"

    def test_factual_date(self):
        assert _classify_query("RBI MPC meeting date") == "factual"

    def test_factual_percentage(self):
        assert _classify_query("GDP growth percentage") == "factual"

    # ── Comparative signals ────────────────────────────────────────────────
    def test_comparative_vs(self):
        assert _classify_query("budget 2023 vs 2024") == "comparative"

    def test_comparative_compare(self):
        assert _classify_query("compare RBI policies between 2022 and 2024") == "comparative"

    def test_comparative_difference(self):
        assert _classify_query("what is the difference between FRBM 2018 and 2023") == "comparative"

    def test_comparative_versus(self):
        assert _classify_query("PM-KISAN benefits versus Kisan Credit Card") == "comparative"

    def test_comparative_badla(self):
        assert _classify_query("2023 mein kya badla") == "comparative"

    # ── Conceptual (default) ───────────────────────────────────────────────
    def test_conceptual_explain(self):
        assert _classify_query("explain RBI inflation targeting framework") == "conceptual"

    def test_conceptual_what_is(self):
        assert _classify_query("what is the monetary policy committee") == "conceptual"

    def test_conceptual_hindi_explain(self):
        assert _classify_query("RBI ka kaam kya hota hai") == "conceptual"

    def test_conceptual_empty_defaults(self):
        assert _classify_query("") == "conceptual"

    def test_comparative_wins_over_factual(self):
        # "vs" + "amount" — compare should win because compare signals are checked first
        assert _classify_query("compare PM-KISAN amount vs MGNREGS amount") == "comparative"


class TestGetRetrievalParams:
    def test_factual_profile(self):
        params = _get_retrieval_params("factual", "irrelevant")
        assert params == RETRIEVAL_PROFILES["factual"]
        assert params["k"] == 3
        assert params["bm25_weight"] > params["dense_weight"]

    def test_conceptual_profile(self):
        params = _get_retrieval_params("conceptual", "irrelevant")
        assert params == RETRIEVAL_PROFILES["conceptual"]
        assert params["k"] == 8
        assert params["dense_weight"] > params["bm25_weight"]

    def test_comparative_profile(self):
        params = _get_retrieval_params("comparative", "irrelevant")
        assert params == RETRIEVAL_PROFILES["comparative"]
        assert params["k"] == 12
        assert params["dense_weight"] == params["bm25_weight"]

    def test_auto_resolves_to_factual(self):
        params = _get_retrieval_params("auto", "repo rate kya tha")
        assert params["k"] == 3  # factual profile

    def test_auto_resolves_to_comparative(self):
        params = _get_retrieval_params("auto", "budget 2023 vs 2024")
        assert params["k"] == 12  # comparative profile

    def test_auto_resolves_to_conceptual(self):
        params = _get_retrieval_params("auto", "explain RBI's role in economy")
        assert params["k"] == 8  # conceptual profile

    def test_unknown_mode_defaults_to_conceptual(self):
        params = _get_retrieval_params("unknown_mode", "some query")
        assert params == RETRIEVAL_PROFILES["conceptual"]

    def test_all_profiles_have_required_keys(self):
        required = {"k", "dense_weight", "bm25_weight", "top_k_after_rerank"}
        for mode, profile in RETRIEVAL_PROFILES.items():
            assert required.issubset(profile.keys()), f"Profile '{mode}' missing keys"

    def test_top_k_always_leq_k(self):
        for profile in RETRIEVAL_PROFILES.values():
            assert profile["top_k_after_rerank"] <= profile["k"]

    def test_weights_sum_to_one(self):
        for mode, profile in RETRIEVAL_PROFILES.items():
            total = profile["dense_weight"] + profile["bm25_weight"]
            assert abs(total - 1.0) < 1e-9, f"Profile '{mode}' weights don't sum to 1.0"
