import os

# Root of the project (one level above core/)
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ── Directories ────────────────────────────────────────────────────────────────
DOCS_DIR         = os.path.join(_BASE_DIR, "docs")
DATA_DIR         = os.path.join(_BASE_DIR, "data")
CHROMA_DB_PATH   = os.path.join(_BASE_DIR, "chroma_db")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "india_policy_docs"

# ── Models ─────────────────────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TEMPERATURE = 0

# ── Retrieval ──────────────────────────────────────────────────────────────────
DEFAULT_K             = 5     # candidates from dense + BM25 each
DENSE_WEIGHT          = 0.6
BM25_WEIGHT           = 0.4
TOP_K_AFTER_RERANK    = 3
SCORE_THRESHOLD       = 0.0   # min score to keep a dense result

# ── Agent limits ───────────────────────────────────────────────────────────────
MAX_TOOL_CALLS        = 8
MAX_ITERATIONS        = 10
GRAPH_RECURSION_LIMIT = 50

# ── Context compression ────────────────────────────────────────────────────────
# Parent chunks are 2000-4000 chars each; 3 results = up to 12000 chars = ~3000 tokens.
# Set threshold high enough that one search result does NOT immediately trigger compression.
BASE_TOKEN_THRESHOLD  = 8000
TOKEN_GROWTH_FACTOR   = 0.5

# ── Parent-child chunking ──────────────────────────────────────────────────────
CHILD_CHUNK_SIZE      = 500
CHILD_CHUNK_OVERLAP   = 100
MIN_PARENT_SIZE       = 2000
MAX_PARENT_SIZE       = 4000
HEADERS_TO_SPLIT_ON   = [
    ("#",   "H1"),
    ("##",  "H2"),
    ("###", "H3"),
]

# ── Hallucination judge ────────────────────────────────────────────────────────
# Score 1 = fully grounded, 5 = completely hallucinated
HALLUCINATION_SAFE_THRESHOLD = 2   # score <= 2 → 🟢 Verified
HALLUCINATION_WARN_THRESHOLD = 4   # score >= 4 → 🔴 Warning
                                   # 3           → 🟡 Review

# ── SQLite (Text2SQL) ──────────────────────────────────────────────────────────
SQLITE_DB_PATH  = os.path.join(_BASE_DIR, "data", "budget.db")
BUDGET_CSV_PATH = os.path.join(_BASE_DIR, "data", "budget_data.csv")
