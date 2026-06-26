# InsightEngine AI

A production-grade multi-agent RAG (Retrieval-Augmented Generation) system that answers questions from uploaded documents using hybrid search, parallel agent orchestration, and real-time streaming.

**Live:** http://80.225.212.121:8000

--

## Features

- **Document Q&A** — Upload PDFs, Word docs, or text files and ask anything
- **Hybrid Search** — Dense (ChromaDB) + Sparse (BM25) + Cross-encoder reranking pipeline
- **Multi-Agent Orchestration** — Parallel LangGraph agents handle multiple sub-questions simultaneously
- **CRAG** — Corrective RAG loop with automatic query rewriting on irrelevant retrievals
- **Multi-Hop Reasoning** — Breaks complex questions into ordered search steps and chains findings
- **Compare Mode** — Side-by-side comparison of two documents on any topic
- **Study Mode** — Document-based quiz generation with automated answer evaluation
- **Web Search Toggle** — DuckDuckGo fallback when documents don't have the answer
- **Text2SQL** — Natural language queries against structured budget data (SQLite)
- **Hallucination Judge** — LLM-as-Judge scores every answer 1–5 for factual grounding
- **Hindi Mode** — Full Devanagari output with technical terms preserved
- **User Memory** — Semantic memory extracted from conversations, personalizes future responses
- **Streaming** — Word-by-word token streaming via `astream_events`
- **Rate Limiting** — 10 requests/60s per session

---

## Architecture

```
User Message
    │
    ├─ summarize_history        Compact prior context; inject user memories
    ├─ rewrite_query            Clarify + split into sub-questions (structured output)
    │   └─[unclear]──► request_clarification   (HITL interrupt — waits for user)
    ├─ route_query              Classify: rag | sql | multi_hop | compare
    │
    ├─[RAG]──► Send("agent") × N   Parallel agents, one per sub-question
    │           └─ orchestrator → search_chunks → retrieval_grader
    │               ├─[irrelevant, attempts<2]──► query_rewriter_loop → orchestrator
    │               ├─[token limit]──► compress_context → orchestrator
    │               └─[done]──► collect_answer
    │           └─► after_agents ──► aggregate_answers
    │                           └──► diff_synthesizer   (compare mode)
    │
    ├─[SQL]──► text2sql_node        NL → SQL → SQLite → formatted result
    │
    ├─[multi_hop]──► reasoning_planner
    │                └─► execute_reasoning_step (self-loop)
    │                └─► reasoning_synthesizer
    │
    └─ hallucination_judge      Score answer 1–5; badge stored in state
```

**Agent subgraph** (runs N times in parallel for RAG):
```
START → orchestrator → tools (search_chunks / web_search)
             │              └─► retrieval_grader
             │                   ├─[irrelevant]──► query_rewriter_loop → orchestrator
             │                   └─[relevant]──► should_compress_context
             │                                        ├─► compress_context → orchestrator
             │                                        └─► orchestrator (continue)
             ├─[no tool call]──► collect_answer → END
             └─[max iterations]──► fallback_response → collect_answer → END
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Orchestration | LangGraph 0.3+ |
| Vector DB | ChromaDB (in-process) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Sparse Search | BM25Okapi (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Structured Data | SQLite + SQLAlchemy |
| Web Search | DuckDuckGo (`duckduckgo-search`) |
| UI | Chainlit 2.0+ |
| PDF Parsing | PyMuPDF + `pymupdf4llm` |
| Token Counting | `tiktoken` (cl100k_base) |
| LLM Cache | SQLite (`langchain_community.cache.SQLiteCache`) |
| Deployment | Docker on Oracle Cloud (OCI `e2.1.micro`) |
| Reverse Proxy | Caddy (HTTPS via DuckDNS) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
india-policy-agent/
├── core/
│   ├── graph.py          # LangGraph pipeline — State, nodes, edges, build_graph()
│   ├── tools.py          # ToolFactory — hybrid search, BM25, cross-encoder
│   ├── ingestion.py      # Document ingestion — parent-child chunking, ChromaDB
│   ├── prompts.py        # All LLM system prompts (one file for easy tuning)
│   ├── judge.py          # Hallucination judge — LLM-as-Judge scorer
│   ├── retrieval_grader.py  # CRAG relevance grader
│   ├── text2sql.py       # NL → SQL engine for budget data
│   ├── study_graph.py    # Study Mode LangGraph — quiz generation + evaluation
│   ├── memory_store.py   # Semantic user memory (ChromaDB collection)
│   ├── web_search.py     # DuckDuckGo search tool
│   ├── history.py        # Conversation history (SQLite)
│   ├── llm.py            # LLM + grader LLM factory
│   ├── rate_limiter.py   # Sliding-window rate limiter
│   └── config.py         # All config constants from env
├── ui/
│   └── app.py            # Chainlit UI — callbacks, streaming, session state
├── tests/
│   ├── unit/             # 163 unit tests
│   └── integration/      # Integration tests
├── public/
│   ├── custom.css        # Dark navy theme (Design 2)
│   └── budget.html       # Interactive budget dashboard
├── .chainlit/
│   └── config.toml       # Chainlit theme config
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD: test → scan → deploy
├── Dockerfile
├── docker-compose.yml
├── Caddyfile
└── requirements.txt
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► Dense Search      ChromaDB cosine similarity (k×2 candidates)
  ├─► Sparse Search     BM25Okapi with user/source filter masking
  │
  ├─► Score Fusion      hybrid = 0.6×dense_norm + 0.4×bm25_norm
  │                     (both normalised with min-max scaling)
  │
  └─► Cross-Encoder     ms-marco-MiniLM-L-6-v2 reranks top-k candidates
        └─► top results returned with parent chunk expansion
```

**Adaptive Retrieval** — 4 profiles auto-selected by query type:

| Profile | k | Dense Weight | BM25 Weight | Top-k after rerank |
|---|---|---|---|---|
| `factual` | 8 | 0.4 | 0.6 | 3 |
| `conceptual` | 10 | 0.7 | 0.3 | 4 |
| `comparative` | 12 | 0.5 | 0.5 | 5 |
| `auto` | — | — | — | classifier picks profile |

---

## Setup

### Local (Docker)

```bash
git clone https://github.com/shivamkr27/Indian-policy-intelligent-agent.git
cd Indian-policy-intelligent-agent

# Create .env
cp .env.example .env
# Add your GROQ_API_KEY

docker compose up --build
# App available at http://localhost:8000
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
CHAINLIT_AUTH_SECRET=generate_with_secrets.token_hex(32)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_password
USER_ISOLATION=true              # Isolate documents per user
```

---

## CI/CD Pipeline

GitHub Actions runs on every push to `master`:

```
test  →  pytest tests/unit/ (CPU-only PyTorch)
scan  →  Trivy CVE scan (CRITICAL + HIGH)
deploy → SSH to OCI VM → docker compose up -d --build
```

**Deploy details:**
- `appleboy/ssh-action` with `command_timeout: 30m`
- First build ~15 min (model downloads cached after first run)
- Subsequent builds ~2–3 min (Docker layer cache)

---

## Deployment

Runs on Oracle Cloud Infrastructure `e2.1.micro` (1GB RAM, 1 OCPU):

- **2GB swap** configured for ML model loading
- **ChromaDB in-process** — no separate container (saves ~200MB RAM)
- **Memory limit:** `800m` container, `2500m` with swap
- **Thread control:** `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`
- **Idle memory:** ~354MB / 800MB (44%)
- Deployed on OCI e2.1.micro, accessible at `http://80.225.212.121:8000`

---

## Tests

```bash
pytest tests/unit/ -v
# 163 tests across: ingestion, graph nodes, tools, retrieval grader,
# memory store, history, study store, text2sql, multi-hop, adaptive retrieval
```
