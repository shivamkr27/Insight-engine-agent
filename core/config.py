import os

# Root of the project (one level above core/)
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Ensure data directory exists before any SQLite databases are created
_DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

# ── Directories ────────────────────────────────────────────────────────────────
DOCS_DIR          = os.path.join(_BASE_DIR, "docs")
DATA_DIR          = _DATA_DIR
CHROMA_DB_PATH    = os.path.join(_BASE_DIR, "chroma_db")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")   # kept for volume-mount compat

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "india_policy_docs"
# Set CHROMA_HOST to switch from local disk to client-server mode
CHROMA_HOST     = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT     = int(os.environ.get("CHROMA_PORT", "8001"))

# ── Models ─────────────────────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TEMPERATURE = 0

# ── LLM provider ───────────────────────────────────────────────────────────────
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
LLM_TIMEOUT  = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
DEFAULT_K             = 5
DENSE_WEIGHT          = 0.6
BM25_WEIGHT           = 0.4
TOP_K_AFTER_RERANK    = 3
SCORE_THRESHOLD       = 0.0

# ── Agent limits ───────────────────────────────────────────────────────────────
MAX_TOOL_CALLS        = 8
MAX_ITERATIONS        = 10
GRAPH_RECURSION_LIMIT = 50

# ── Context compression ────────────────────────────────────────────────────────
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
HALLUCINATION_SAFE_THRESHOLD = 2
HALLUCINATION_WARN_THRESHOLD = 4

# ── SQLite paths ───────────────────────────────────────────────────────────────
SQLITE_DB_PATH          = os.path.join(_DATA_DIR, "budget.db")
BUDGET_CSV_PATH         = os.path.join(_DATA_DIR, "budget_data.csv")
SQLITE_CHECKPOINT_PATH  = os.path.join(_DATA_DIR, "checkpoints.db")
PARENT_STORE_DB_PATH    = os.path.join(_DATA_DIR, "parent_store.db")
BM25_CACHE_PATH         = os.path.join(_DATA_DIR, "bm25_cache.pkl")

# ── Rate limiting ──────────────────────────────────────────────────────────────
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW   = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# ── File upload ────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "50"))

# ── Conversation history ────────────────────────────────────────────────────────
HISTORY_DB_PATH = os.path.join(_DATA_DIR, "history.db")

# ── Per-user document isolation ─────────────────────────────────────────────────
# Set USER_ISOLATION=false to disable per-user filtering (shared corpus mode).
# Automatically disabled when user_id is "default" (no-auth / local dev).
USER_ISOLATION = os.environ.get("USER_ISOLATION", "true").lower() != "false"

# ── Study Mode ───────────────────────────────────────────────────────────────────
STUDY_DB_PATH = os.path.join(_DATA_DIR, "study.db")

# ── CRAG (Corrective RAG) ─────────────────────────────────────────────────────────
GRADER_MODEL = "llama-3.1-8b-instant"

# ── Adaptive Retrieval ────────────────────────────────────────────────────────────
RETRIEVAL_PROFILES = {
    "factual":     {"k": 3,  "dense_weight": 0.3, "bm25_weight": 0.7, "top_k_after_rerank": 2},
    "conceptual":  {"k": 8,  "dense_weight": 0.7, "bm25_weight": 0.3, "top_k_after_rerank": 4},
    "comparative": {"k": 12, "dense_weight": 0.5, "bm25_weight": 0.5, "top_k_after_rerank": 5},
}

# ── Semantic Memory ───────────────────────────────────────────────────────────────
MEMORY_COLLECTION = "user_memories"
