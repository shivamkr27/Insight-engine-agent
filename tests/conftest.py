"""Shared pytest fixtures for the India Policy Agent test suite."""

import os
import sys
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Make core/ importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a fixed AIMessage-like response."""
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mock response from LLM.")
    llm.with_structured_output.return_value = llm
    return llm


@pytest.fixture
def in_memory_db():
    """Provide a temporary in-memory SQLite connection for store tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid PDF content (magic bytes only — enough for validation tests)."""
    return b"%PDF-1.4 minimal test pdf"


@pytest.fixture(autouse=False)
def tmp_data_dir(tmp_path):
    """Redirect DATA_DIR config to a temp directory for isolation."""
    import core.config as cfg
    original = cfg.DATA_DIR
    cfg.DATA_DIR = str(tmp_path)
    cfg.SQLITE_DB_PATH = str(tmp_path / "budget.db")
    cfg.SQLITE_CHECKPOINT_PATH = str(tmp_path / "checkpoints.db")
    cfg.PARENT_STORE_DB_PATH = str(tmp_path / "parent_store.db")
    cfg.BM25_CACHE_PATH = str(tmp_path / "bm25_cache.pkl")
    yield tmp_path
    cfg.DATA_DIR = original
