"""
Text2SQL engine: natural language question → SQLite SQL → formatted result

Flow:
  budget_data.csv
     └► pandas → file-based SQLite table "budget_allocations"
           └► LLM (schema + question) → SQL string
                 └► sqlite3.execute → rows → formatted string

Using a file-based SQLite (SQLITE_DB_PATH) instead of :memory: so data
survives restarts and concurrent reads are safe via a threading lock.
"""

import re
import sqlite3
import threading
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from .config import BUDGET_CSV_PATH, SQLITE_DB_PATH
from .logging_config import get_logger

logger = get_logger(__name__)

_TABLE = "budget_allocations"


class Text2SQLEngine:
    """
    Manages a file-based SQLite database loaded from budget_data.csv.

    Usage:
        engine = Text2SQLEngine()
        result = engine.query("Which ministry got the most funds in 2024?", llm)
        engine.refresh_from_csv()  # reload without restart
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        if not self._table_exists():
            self._load_csv()
        self._schema_prompt = self._build_schema_prompt()
        logger.info(f"Text2SQLEngine ready — db: {db_path}")

    # ── Setup ──────────────────────────────────────────────────────────────

    def _table_exists(self) -> bool:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
        )
        return cur.fetchone() is not None

    def _load_csv(self) -> None:
        df = pd.read_csv(BUDGET_CSV_PATH)
        df.to_sql(_TABLE, self._conn, if_exists="replace", index=False)
        self._conn.commit()
        logger.info(f"SQLite loaded: {len(df)} rows → table '{_TABLE}'")

    def _build_schema_prompt(self) -> str:
        cursor = self._conn.execute(f"SELECT * FROM {_TABLE} LIMIT 3")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        lines = [
            f"Table: {_TABLE}",
            f"Columns: {', '.join(columns)}",
            "",
            "Sample rows:",
        ]
        for row in rows:
            lines.append("  " + str(dict(zip(columns, row))))

        lines += [
            "",
            "Notes:",
            "- allocated_crore and spent_crore are in Indian Crore (₹)",
            "- spent_crore = 0 means the year is ongoing (budget allocated, not yet spent)",
            "- year values available: 2023, 2024, 2025",
        ]
        return "\n".join(lines)

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh_from_csv(self, csv_path: str = BUDGET_CSV_PATH) -> int:
        """Reload budget data from CSV — callable without restart."""
        with self._lock:
            df = pd.read_csv(csv_path)
            df.to_sql(_TABLE, self._conn, if_exists="replace", index=False)
            self._conn.commit()
            self._schema_prompt = self._build_schema_prompt()
            logger.info(f"Budget data refreshed: {len(df)} rows")
            return len(df)

    def query(self, question: str, llm) -> str:
        """
        Convert a natural language question to SQL, run it, and return
        the result as a formatted string.
        """
        system_prompt = f"""You are a SQLite expert. Your only task is to write a SQL query.

{self._schema_prompt}

Rules:
- Output ONLY the raw SQL query — no explanation, no markdown, no code fences
- Use only the table and columns listed above
- For text matching use LIKE '%value%' (case-insensitive with LOWER())
- Default: ORDER BY allocated_crore DESC LIMIT 20 unless the question specifies otherwise
- spent_crore = 0 means data unavailable — exclude from percentage calculations
"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])

        sql = self._extract_sql(response.content.strip())
        logger.info(f"Generated SQL: {sql[:120]}")

        with self._lock:
            try:
                cursor = self._conn.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                if not rows:
                    return "No records found for this query."

                return self._format_result(sql, columns, rows)

            except sqlite3.Error as e:
                logger.warning(f"SQL execution error: {e} | SQL: {sql}")
                return f"SQL execution error: {e}\n\nGenerated SQL:\n{sql}"

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sql(text: str) -> str:
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text

    @staticmethod
    def _format_result(sql: str, columns: list, rows: list) -> str:
        col_widths = [
            max(len(str(col)), max(len(str(r[i])) for r in rows))
            for i, col in enumerate(columns)
        ]
        sep    = "-+-".join("-" * w for w in col_widths)
        header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))

        lines = [
            f"Query: {sql}",
            f"Results: {len(rows)} row(s)\n",
            header,
            sep,
        ]
        for row in rows:
            lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

        return "\n".join(lines)
