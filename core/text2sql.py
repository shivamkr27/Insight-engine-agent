"""
Text2SQL engine: natural language question → SQLite SQL → formatted result

Interview concept:
  Text2SQL lets non-technical users query structured data in plain English.
  We give the LLM the table schema + sample rows, it generates SQL, we execute
  it with sqlite3, and return the result as a readable string.

Flow:
  budget_data.csv
     └─► pandas.read_sql  →  in-memory SQLite table "budget_allocations"
           └─► LLM (schema + question)  →  SQL string
                 └─► sqlite3.execute  →  rows  →  formatted string
"""

import re
import sqlite3
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from .config import BUDGET_CSV_PATH

_TABLE = "budget_allocations"


class Text2SQLEngine:
    """
    Manages an in-memory SQLite database loaded from budget_data.csv.

    Usage (called from graph nodes, not from agent tools):
        engine = Text2SQLEngine()
        result = engine.query("Which ministry got the most funds in 2024?", llm)
    """

    def __init__(self):
        # In-memory SQLite — fresh each run, loaded from CSV
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._load_csv()
        self._schema_prompt = self._build_schema_prompt()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _load_csv(self) -> None:
        df = pd.read_csv(BUDGET_CSV_PATH)
        df.to_sql(_TABLE, self._conn, if_exists="replace", index=False)
        print(f"✓ SQLite loaded: {len(df)} rows → table '{_TABLE}'")

    def _build_schema_prompt(self) -> str:
        """
        Create a schema description that will be injected into the LLM prompt.
        Showing sample rows helps the LLM understand value formats.
        """
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

    def query(self, question: str, llm) -> str:
        """
        Convert a natural language question to SQL, run it, and return
        the result as a formatted string.

        Args:
            question: Plain English question about budget data.
            llm:      LangChain LLM instance (Groq).

        Returns:
            Formatted string with results, or an error description.
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

        try:
            cursor = self._conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                return "No records found for this query."

            return self._format_result(sql, columns, rows)

        except sqlite3.Error as e:
            # Return the error so the agent can relay it gracefully
            return f"SQL execution error: {e}\n\nGenerated SQL:\n{sql}"

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Strip markdown code fences if the LLM accidentally wraps the SQL."""
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text

    @staticmethod
    def _format_result(sql: str, columns: list, rows: list) -> str:
        """Format query results as a readable plain-text table."""
        col_widths = [
            max(len(str(col)), max(len(str(r[i])) for r in rows))
            for i, col in enumerate(columns)
        ]
        sep = "-+-".join("-" * w for w in col_widths)
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
