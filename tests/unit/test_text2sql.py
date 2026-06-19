"""Unit tests for core/text2sql.py — no LLM calls, no CSV required."""

import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage


class TestExtractSQL:
    def _engine(self):
        from core.text2sql import Text2SQLEngine
        return Text2SQLEngine.__new__(Text2SQLEngine)

    def test_strips_sql_code_fence(self):
        engine = self._engine()
        result = engine._extract_sql("```sql\nSELECT * FROM budget_allocations\n```")
        assert result == "SELECT * FROM budget_allocations"

    def test_strips_generic_code_fence(self):
        engine = self._engine()
        result = engine._extract_sql("```\nSELECT 1\n```")
        assert result == "SELECT 1"

    def test_passthrough_plain_sql(self):
        engine = self._engine()
        sql = "SELECT ministry, allocated_crore FROM budget_allocations LIMIT 5"
        assert engine._extract_sql(sql) == sql

    def test_case_insensitive_fence(self):
        engine = self._engine()
        result = engine._extract_sql("```SQL\nSELECT 1\n```")
        assert result == "SELECT 1"


class TestFormatResult:
    def _engine(self):
        from core.text2sql import Text2SQLEngine
        return Text2SQLEngine.__new__(Text2SQLEngine)

    def test_contains_column_headers(self):
        engine = self._engine()
        cols = ["ministry", "allocated_crore"]
        rows = [("Agriculture", 60000)]
        output = engine._format_result("SELECT ...", cols, rows)
        assert "ministry" in output
        assert "allocated_crore" in output

    def test_contains_row_values(self):
        engine = self._engine()
        cols = ["ministry", "allocated_crore"]
        rows = [("Agriculture", 60000)]
        output = engine._format_result("SELECT ...", cols, rows)
        assert "Agriculture" in output
        assert "60000" in output

    def test_row_count_in_output(self):
        engine = self._engine()
        cols = ["year", "amount"]
        rows = [("2023", 1000), ("2024", 2000)]
        output = engine._format_result("SELECT ...", cols, rows)
        assert "2 row(s)" in output


class TestText2SQLEngineInit:
    def test_creates_table_from_csv(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "budget_data.csv"
        pd.DataFrame({
            "ministry": ["Agriculture", "Defence"],
            "year": [2024, 2024],
            "allocated_crore": [60000, 600000],
            "spent_crore": [55000, 580000],
        }).to_csv(csv_path, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv_path)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "budget.db")):
            from core.text2sql import Text2SQLEngine
            engine = Text2SQLEngine(db_path=str(tmp_path / "budget.db"))

        cur = engine._conn.execute("SELECT COUNT(*) FROM budget_allocations")
        assert cur.fetchone()[0] == 2

    def test_table_exists_returns_false_on_fresh_db(self, tmp_path):
        from core.text2sql import Text2SQLEngine
        engine = Text2SQLEngine.__new__(Text2SQLEngine)
        engine._conn = sqlite3.connect(":memory:")
        assert engine._table_exists() is False

    def test_table_exists_returns_true_after_create(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "budget_data.csv"
        pd.DataFrame({"ministry": ["A"], "year": [2024], "allocated_crore": [1], "spent_crore": [0]}).to_csv(csv_path, index=False)

        with patch("core.text2sql.BUDGET_CSV_PATH", str(csv_path)), \
             patch("core.text2sql.SQLITE_DB_PATH", str(tmp_path / "b.db")):
            from core.text2sql import Text2SQLEngine
            engine = Text2SQLEngine(db_path=str(tmp_path / "b.db"))

        assert engine._table_exists() is True
