"""
Integration tests for core/ modules working together.
These tests use real SQLite but mock the LLM to stay fast and offline.
"""

import os
import sqlite3
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from core.ingestion import _ParentStore
from core.text2sql import Text2SQLEngine


class TestParentStoreWithTextSQL:
    """Verify that the two SQLite-backed stores coexist in the same data dir."""

    def test_separate_databases_no_conflict(self, tmp_path):
        store = _ParentStore(db_path=str(tmp_path / "parents.db"))

        csv = tmp_path / "budget.csv"
        pd.DataFrame({
            "ministry": ["Education"],
            "year": [2024],
            "allocated_crore": [100000],
            "spent_crore": [90000],
        }).to_csv(csv, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "budget.db")):
            engine = Text2SQLEngine(db_path=str(tmp_path / "budget.db"))

        store.save("p1", "Some policy text.", {"source": "edu_policy.pdf"})
        assert store.load("p1")["content"] == "Some policy text."

        cur = engine._conn.execute("SELECT COUNT(*) FROM budget_allocations")
        assert cur.fetchone()[0] == 1


class TestText2SQLQueryWithMockLLM:
    def test_valid_query_returns_formatted_table(self, tmp_path):
        csv = tmp_path / "budget.csv"
        pd.DataFrame({
            "ministry": ["Agriculture", "Defence"],
            "year": [2024, 2024],
            "allocated_crore": [60000, 600000],
            "spent_crore": [55000, 0],
        }).to_csv(csv, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "budget.db")):
            engine = Text2SQLEngine(db_path=str(tmp_path / "budget.db"))

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="SELECT ministry, allocated_crore FROM budget_allocations ORDER BY allocated_crore DESC"
        )

        result = engine.query("Which ministry has the highest allocation?", mock_llm)
        assert "Defence" in result
        assert "600000" in result

    def test_sql_error_returns_error_string(self, tmp_path):
        csv = tmp_path / "budget.csv"
        pd.DataFrame({"ministry": ["A"], "year": [2024], "allocated_crore": [1], "spent_crore": [0]}).to_csv(csv, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "budget.db")):
            engine = Text2SQLEngine(db_path=str(tmp_path / "budget.db"))

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="SELECT * FROM nonexistent_table")

        result = engine.query("Bad query", mock_llm)
        assert "SQL execution error" in result

    def test_refresh_from_csv_updates_row_count(self, tmp_path):
        csv = tmp_path / "budget.csv"
        pd.DataFrame({"ministry": ["A"], "year": [2024], "allocated_crore": [1], "spent_crore": [0]}).to_csv(csv, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "budget.db")):
            engine = Text2SQLEngine(db_path=str(tmp_path / "budget.db"))

        pd.DataFrame({
            "ministry": ["A", "B", "C"],
            "year": [2024, 2024, 2024],
            "allocated_crore": [1, 2, 3],
            "spent_crore": [0, 0, 0],
        }).to_csv(csv, index=False)

        count = engine.refresh_from_csv(str(csv))
        assert count == 3

        cur = engine._conn.execute("SELECT COUNT(*) FROM budget_allocations")
        assert cur.fetchone()[0] == 3
