"""
InsightEngine AI — Chainlit UI

Startup sequence (once per container):
  1. Load .env
  2. Init Ingestion   → loads ChromaDB + SQLite parent store
  3. Init Text2SQL    → loads budget_data.csv → file-based SQLite
  4. Init ToolFactory → loads CrossEncoder + builds/loads BM25 index
  5. Build LangGraph  → compile the full agent graph with SqliteSaver
  6. Build StudyGraph → reuses the same checkpointer
  7. Init MemoryStore → separate ChromaDB collection for user memories

Per-session (each browser tab / user):
  - Unique thread_id → separate conversation memory
  - HITL state flag  → tracks if graph is waiting for clarification
  - Rate limiting    → 10 requests / 60 seconds per session
  - answer_language  → "english" (default) or "hindi"
  - study_active     → True while a study session is running
  - compare_mode     → True while user is about to send a comparison query
  - web_search_enabled → True when user has toggled web search on
  - user_memories    → fetched from UserMemoryStore on each message (topical)

Auth (optional, enabled by setting CHAINLIT_AUTH_SECRET in .env):
  - Password auth via ADMIN_USERNAME + ADMIN_PASSWORD env vars
  - Without CHAINLIT_AUTH_SECRET the app runs open (local dev mode)
"""

import asyncio
import os
import re
import sqlite3 as _sqlite3
import sys
import io
import shutil
from pathlib import Path
from typing import Optional

# Make 'core' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Enable LangSmith tracing if configured
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "insightengine-ai")

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

from core.config import (
    DOCS_DIR, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW, MAX_UPLOAD_SIZE_MB,
    SQLITE_DB_PATH, HISTORY_DB_PATH, STUDY_DB_PATH,
)
from core.llm import get_llm
from core.ingestion import Ingestion
from core.tools import ToolFactory, user_id_ctx
from core.text2sql import Text2SQLEngine
from core.judge import HallucinationJudge
from core.graph import build_graph, create_checkpointer
from core.study_graph import build_study_graph
from core.memory_store import UserMemoryStore
from core.rate_limiter import RateLimiter
from core.history import ConversationStore
from core.study_store import StudyStore
from core.logging_config import get_logger

logger = get_logger(__name__)

# ── Budget data API (served at /api/budget-data for the dashboard) ─────────────
try:
    from chainlit.server import app as _chainlit_app
    from fastapi.responses import JSONResponse as _JSONResponse

    @_chainlit_app.get("/api/budget-data")
    async def _budget_data_api():
        try:
            conn = _sqlite3.connect(SQLITE_DB_PATH)
            rows = conn.execute(
                "SELECT ministry, scheme, allocated_crore, spent_crore, year "
                "FROM budget_allocations ORDER BY year, ministry"
            ).fetchall()
            conn.close()
            data = [
                {"ministry": r[0], "scheme": r[1], "allocated": r[2], "spent": r[3], "year": r[4]}
                for r in rows
            ]
            return _JSONResponse({"data": data})
        except Exception as exc:
            return _JSONResponse({"error": str(exc)}, status_code=500)

except Exception as _e:
    logger.warning(f"Could not register budget API route: {_e}")

# ── Regex: extract source filenames from formatted search results ──────────────
_SOURCE_PATTERN = re.compile(r'^Source:\s*(.+?)$', re.MULTILINE)

# ── Supported upload extensions ────────────────────────────────────────────────
_SUPPORTED_EXT = (".pdf", ".docx", ".txt")

# ── Global singletons ──────────────────────────────────────────────────────────
logger.info("Starting InsightEngine AI...")

_llm          = get_llm()
_ingestion    = Ingestion()
_sql_engine   = Text2SQLEngine()
_judge        = HallucinationJudge()
_factory      = ToolFactory(_ingestion)
_memory_store = UserMemoryStore(_ingestion._embeddings)
_limiter      = RateLimiter(max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW)
_history      = ConversationStore(db_path=HISTORY_DB_PATH)
_study_store  = StudyStore(db_path=STUDY_DB_PATH)

from langgraph.checkpoint.memory import InMemorySaver
_checkpointer = InMemorySaver()
_graph        = build_graph(_llm, _factory, _sql_engine, _judge, _checkpointer)
_study_graph  = build_study_graph(_llm, _factory, _checkpointer)
logger.info("Agent ready (InMemorySaver — upgrading to persistent checkpointer on startup).")


@cl.on_app_startup
async def _on_app_startup():
    """Upgrade checkpointer from InMemorySaver to AsyncSqliteSaver.

    Runs once before any user connects.  Must be async so aiosqlite can open
    the connection in the event loop — this is why we can't do it at module level.
    """
    global _checkpointer, _graph, _study_graph
    _checkpointer = await create_checkpointer()
    _graph        = build_graph(_llm, _factory, _sql_engine, _judge, _checkpointer)
    _study_graph  = build_study_graph(_llm, _factory, _checkpointer)
    logger.info("Persistent checkpointer active — conversation state survives container restarts.")


# ── Auth ───────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    admin_user = os.environ.get("ADMIN_USERNAME", "admin")
    admin_pass = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_pass:
        return cl.User(identifier=username, metadata={"role": "user"})
    if username == admin_user and password == admin_pass:
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None


# ── Chat lifecycle ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    thread_id = cl.context.session.id
    user      = cl.context.session.user
    user_id   = user.identifier if user else "default"

    cl.user_session.set("thread_id",              thread_id)
    cl.user_session.set("user_id",                user_id)
    cl.user_session.set("awaiting_clarification",  False)
    cl.user_session.set("last_answer",            "")
    cl.user_session.set("last_sources",           [])
    cl.user_session.set("last_question",          "")
    cl.user_session.set("answer_language",        "english")
    cl.user_session.set("study_active",           False)
    cl.user_session.set("study_thread_id",        "")
    cl.user_session.set("compare_mode",           False)
    cl.user_session.set("compare_doc_a",          "")
    cl.user_session.set("compare_doc_b",          "")
    cl.user_session.set("web_search_enabled",     False)

    _history.upsert(thread_id, user_id)

    await cl.Message(
        content=_welcome_text(user_id),
        actions=[
            cl.Action(name="upload_pdf",        payload={}, label="📎 Upload File"),
            cl.Action(name="ingest_all",        payload={}, label="📂 Ingest All Docs"),
            cl.Action(name="toggle_hindi",      payload={}, label="🇮🇳 Hindi"),
            cl.Action(name="study_mode",        payload={}, label="📚 Study"),
            cl.Action(name="compare_mode",      payload={}, label="🔀 Compare"),
            cl.Action(name="toggle_web_search", payload={}, label="🌐 Web Search"),
        ],
    ).send()

    await _send_history(user_id, thread_id)
    await _send_study_history(user_id)


@cl.on_message
async def on_message(message: cl.Message):
    user_id   = cl.user_session.get("user_id", "default")
    thread_id = cl.user_session.get("thread_id", cl.context.session.id)

    # ── Special command: "what do you know about me" ──────────────────────
    cmd = message.content.strip().lower().rstrip("?!")
    if cmd in ("memories", "what do you know about me"):
        mems = await asyncio.to_thread(_memory_store.get_all, user_id)
        if not mems:
            await cl.Message(
                content="I don't have any memories about you yet. Chat more and I'll start learning your preferences!"
            ).send()
        else:
            _TYPE_LABELS = {
                "topic_interest": "🎯 Interest",
                "preference":     "⚙️ Preference",
                "knowledge_gap":  "📖 Gap",
                "doc_affinity":   "📄 Docs",
            }
            lines = [
                f"**{_TYPE_LABELS.get(m['memory_type'], '💡')}:** {m['content']}"
                for m in mems
            ]
            await cl.Message(
                content="**What I know about you:**\n\n" + "\n\n".join(lines)
            ).send()
        return

    # Files attached directly in chat
    if message.elements:
        file_elements = [
            e for e in message.elements
            if str(e.name).lower().endswith(_SUPPORTED_EXT)
        ]
        if file_elements:
            await _handle_file_elements(file_elements, user_id=user_id)
            return

    # Rate limiting (keyed per session thread_id)
    if not _limiter.is_allowed(thread_id):
        await cl.Message(
            content="⏳ Rate limit reached. You can send up to 10 messages per minute. Please wait a moment."
        ).send()
        return

    _history.set_title(thread_id, message.content)
    _history.touch(thread_id)
    cl.user_session.set("last_question", message.content)

    config = {"configurable": {"thread_id": thread_id}}

    # ── Study Mode: route answer to study graph ───────────────────────────
    study_active    = cl.user_session.get("study_active", False)
    study_thread_id = cl.user_session.get("study_thread_id", "")
    if study_active and study_thread_id:
        study_config = {"configurable": {"thread_id": study_thread_id}}
        try:
            study_state = _study_graph.get_state(study_config)
            if study_state.next and "evaluate_answer" in study_state.next:
                _study_graph.update_state(
                    study_config,
                    {"messages": [HumanMessage(content=message.content)]},
                )
                await _run_study_graph(study_config, user_id=user_id)
                return
            else:
                await cl.Message(
                    content=(
                        "📚 **Study session isn't waiting for an answer right now.** "
                        "The session may be between questions or in an unexpected state.\n\n"
                        "Use the **Exit Study** button to end the session cleanly, then start a new one."
                    )
                ).send()
                return
        except Exception as e:
            logger.error(f"Study mode routing failed: {e}", exc_info=True)
            cl.user_session.set("study_active", False)

    # ── Compare Mode: send next message as comparison topic ───────────────
    compare_mode_active = cl.user_session.get("compare_mode", False)
    if compare_mode_active:
        doc_a = cl.user_session.get("compare_doc_a", "")
        doc_b = cl.user_session.get("compare_doc_b", "")
        if doc_a and doc_b:
            user_memories = await asyncio.to_thread(
                _memory_store.fetch_relevant, user_id, message.content
            )
            try:
                await _run_graph(
                    user_input=message.content,
                    config=config,
                    user_id=user_id,
                    answer_language=cl.user_session.get("answer_language", "english"),
                    compare_doc_a=doc_a,
                    compare_doc_b=doc_b,
                    user_memories=user_memories,
                )
            finally:
                cl.user_session.set("compare_mode", False)
            return

    # ── Normal flow ───────────────────────────────────────────────────────
    user_memories = await asyncio.to_thread(
        _memory_store.fetch_relevant, user_id, message.content
    )

    if cl.user_session.get("awaiting_clarification"):
        _graph.update_state(config, {"messages": [HumanMessage(content=message.content)]})
        cl.user_session.set("awaiting_clarification", False)
        await _run_graph(
            resume=True, config=config, user_id=user_id,
            answer_language=cl.user_session.get("answer_language", "english"),
            user_memories=user_memories,
        )
    else:
        await _run_graph(
            user_input=message.content, config=config, user_id=user_id,
            answer_language=cl.user_session.get("answer_language", "english"),
            user_memories=user_memories,
        )


# ── Action callbacks ───────────────────────────────────────────────────────────

@cl.action_callback("upload_pdf")
async def on_upload_action(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    files = await cl.AskFileMessage(
        content="Upload up to 5 files — PDF, Word (.docx), or text files. Select multiple with Shift+click.",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
        max_size_mb=MAX_UPLOAD_SIZE_MB,
        max_files=5,
        timeout=300,
    ).send()
    if files:
        await _handle_file_elements(files, user_id=user_id)


@cl.action_callback("ingest_all")
async def on_ingest_all_action(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    await cl.Message(content="Scanning and ingesting all documents…").send()
    async with cl.Step(name="📚 Ingesting all documents", type="tool") as step:
        step.input = f"Scanning: {DOCS_DIR}"
        stats = await asyncio.to_thread(_ingestion.ingest_all, DOCS_DIR, user_id)
        await asyncio.to_thread(_factory.rebuild_bm25)
        ingested = len(stats["ingested"])
        skipped  = len(stats["skipped"])
        step.output = f"✓ {ingested} ingested | {skipped} already done"

    await _send_doc_list(user_id)
    if ingested:
        await cl.Message(content=f"✅ {ingested} document(s) ingested. Ready to answer questions!").send()
    else:
        await cl.Message(content="No new documents found. Upload files first.").send()


# ── Feature 1: Hindi Mode toggle ───────────────────────────────────────────────

@cl.action_callback("toggle_hindi")
async def on_toggle_hindi(action: cl.Action):
    current  = cl.user_session.get("answer_language", "english")
    new_lang = "hindi" if current == "english" else "english"
    cl.user_session.set("answer_language", new_lang)

    if new_lang == "hindi":
        await cl.Message(
            content="🇮🇳 **Hindi Mode** enabled. All answers will be in Hindi (Devanagari script).",
            actions=[cl.Action(name="toggle_hindi", payload={}, label="🇮🇳 Hindi ON — click to switch back")],
        ).send()
    else:
        await cl.Message(
            content="🇬🇧 **English Mode** enabled.",
            actions=[cl.Action(name="toggle_hindi", payload={}, label="🇮🇳 Hindi")],
        ).send()


# ── Feature 2: Study Mode ──────────────────────────────────────────────────────

@cl.action_callback("study_mode")
async def on_study_mode(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    docs    = _ingestion.list_ingested_files(user_id=user_id)

    if not docs:
        await cl.Message(content="No documents ingested yet. Upload files first.").send()
        return

    doc_actions = [
        cl.Action(name="start_study", payload={"doc": d}, label=f"📖 {d}")
        for d in docs[:8]
    ]
    await cl.Message(
        content="**📚 Study Mode** — Select a document to quiz yourself on:",
        actions=doc_actions,
    ).send()


@cl.action_callback("start_study")
async def on_start_study(action: cl.Action):
    doc_name = action.payload.get("doc", "")
    if not doc_name:
        await cl.Message(content="No document selected.").send()
        return

    user_id         = cl.user_session.get("user_id", "default")
    session_id      = cl.context.session.id
    study_thread_id = f"study_{user_id}_{session_id}"

    cl.user_session.set("study_active",    True)
    cl.user_session.set("study_thread_id", study_thread_id)

    _study_store.create_session(user_id, study_thread_id, doc_name)

    await cl.Message(
        content=f"Starting study session for **{doc_name}**... Extracting topics.",
        actions=[cl.Action(name="exit_study", payload={}, label="❌ Exit Study Mode")],
    ).send()

    initial_state = {
        "doc_name":           doc_name,
        "topics":             [],
        "topic_idx":          0,
        "q_idx":              0,
        "questions_per_topic": 3,
        "current_question":   "",
        "current_context":    "",
        "score":              0.0,
        "total":              0,
        "messages":           [],
    }
    study_config = {"configurable": {"thread_id": study_thread_id}}
    await _run_study_graph(study_config, initial_state=initial_state, user_id=user_id)


@cl.action_callback("exit_study")
async def on_exit_study(action: cl.Action):
    cl.user_session.set("study_active",    False)
    cl.user_session.set("study_thread_id", "")
    await cl.Message(content="📚 Study mode ended. Back to normal chat.").send()


@cl.action_callback("resume_study")
async def on_resume_study(action: cl.Action):
    """Resume an in-progress study session from its checkpoint."""
    past_thread_id = action.payload.get("study_thread_id", "")
    doc_name       = action.payload.get("doc_name", "unknown")
    user_id        = cl.user_session.get("user_id", "default")

    if not past_thread_id:
        await cl.Message(content="Could not resume — session ID missing.").send()
        return

    study_config = {"configurable": {"thread_id": past_thread_id}}
    try:
        state = _study_graph.get_state(study_config)
    except Exception as e:
        logger.error(f"resume_study get_state failed: {e}", exc_info=True)
        await cl.Message(content="Could not load study session state. It may have expired.").send()
        return

    cl.user_session.set("study_active",    True)
    cl.user_session.set("study_thread_id", past_thread_id)

    current_q = state.values.get("current_question", "")
    score     = state.values.get("score",  0.0)
    total     = state.values.get("total",  0)

    resume_text = (
        f"📚 **Study session resumed** — *{doc_name}*\n\n"
        f"Score so far: **{score:.0f} / {total}**\n\n"
    )
    if current_q:
        resume_text += f"**Current question:**\n\n{current_q}"
    else:
        resume_text += "Type your answer to continue, or use **Exit Study** to end the session."

    await cl.Message(
        content=resume_text,
        actions=[cl.Action(name="exit_study", payload={}, label="❌ Exit Study Mode")],
    ).send()


# ── Feature 3: Compare Mode ────────────────────────────────────────────────────

@cl.action_callback("compare_mode")
async def on_compare_mode(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    docs    = _ingestion.list_ingested_files(user_id=user_id)

    if len(docs) < 2:
        await cl.Message(
            content="Need at least **2 ingested documents** to use Compare Mode. Upload more files first."
        ).send()
        return

    cl.user_session.set("compare_doc_a", "")
    cl.user_session.set("compare_doc_b", "")
    cl.user_session.set("compare_mode",  False)

    doc_actions = [
        cl.Action(name="compare_pick_a", payload={"doc": d}, label=f"📄 {d}")
        for d in docs[:6]
    ]
    await cl.Message(
        content="**🔀 Compare Mode** — Select **Document A**:",
        actions=doc_actions,
    ).send()


@cl.action_callback("compare_pick_a")
async def on_compare_pick_a(action: cl.Action):
    doc_a   = action.payload.get("doc", "")
    user_id = cl.user_session.get("user_id", "default")
    cl.user_session.set("compare_doc_a", doc_a)

    docs       = _ingestion.list_ingested_files(user_id=user_id)
    other_docs = [d for d in docs if d != doc_a]

    if not other_docs:
        await cl.Message(content="No other documents available to compare with.").send()
        return

    doc_actions = [
        cl.Action(name="compare_pick_b", payload={"doc": d}, label=f"📄 {d}")
        for d in other_docs[:6]
    ]
    await cl.Message(
        content=f"**Document A:** `{doc_a}`\n\nNow select **Document B**:",
        actions=doc_actions,
    ).send()


@cl.action_callback("compare_pick_b")
async def on_compare_pick_b(action: cl.Action):
    doc_b = action.payload.get("doc", "")
    doc_a = cl.user_session.get("compare_doc_a", "")
    cl.user_session.set("compare_doc_b", doc_b)
    cl.user_session.set("compare_mode",  True)

    await cl.Message(
        content=(
            f"**🔀 Compare Mode Active**\n\n"
            f"**A:** `{doc_a}`  vs  **B:** `{doc_b}`\n\n"
            f"Type your comparison topic, e.g. *agriculture budget*, *monetary policy changes*, "
            f"*scheme beneficiaries*"
        ),
        actions=[cl.Action(name="exit_compare", payload={}, label="❌ Cancel Compare")],
    ).send()


@cl.action_callback("exit_compare")
async def on_exit_compare(action: cl.Action):
    cl.user_session.set("compare_mode",  False)
    cl.user_session.set("compare_doc_a", "")
    cl.user_session.set("compare_doc_b", "")
    await cl.Message(content="🔀 Compare mode cancelled. Back to normal chat.").send()


# ── Feature 4: Web Search toggle ───────────────────────────────────────────────

@cl.action_callback("toggle_web_search")
async def on_toggle_web_search(action: cl.Action):
    current = cl.user_session.get("web_search_enabled", False)
    new_val = not current
    cl.user_session.set("web_search_enabled", new_val)

    if new_val:
        await cl.Message(
            content=(
                "🌐 **Web Search ON** — I'll search your documents first, then the web if needed.\n\n"
                "*Note: Web results are real-time but less reliable than your documents.*"
            ),
            actions=[cl.Action(name="toggle_web_search", payload={}, label="🌐 Web Search ON — click to disable")],
        ).send()
    else:
        await cl.Message(
            content="🌐 **Web Search OFF** — Answering from your uploaded documents only.",
            actions=[cl.Action(name="toggle_web_search", payload={}, label="🌐 Web Search")],
        ).send()


# ── Conversation history callbacks ─────────────────────────────────────────────

@cl.action_callback("resume_conversation")
async def on_resume_conversation(action: cl.Action):
    past_thread_id = action.payload.get("thread_id", "")
    user_id        = cl.user_session.get("user_id", "default")

    owned = {c["thread_id"] for c in _history.list_user(user_id)}
    if past_thread_id not in owned:
        await cl.Message(content="Conversation not found.").send()
        return

    cl.user_session.set("thread_id", past_thread_id)
    _history.upsert(past_thread_id, user_id)

    config = {"configurable": {"thread_id": past_thread_id}}
    try:
        state  = _graph.get_state(config)
        msgs   = state.values.get("messages", [])
        last_q, last_a = "", ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content and not last_a:
                last_a = m.content[:400]
            if isinstance(m, HumanMessage) and not last_q:
                last_q = m.content
            if last_q and last_a:
                break
        preview = (
            f"**Conversation resumed.**\n\n"
            f"**Last question:** *{last_q}*\n\n"
            f"**Last answer:** {last_a}{'…' if len(last_a) == 400 else ''}"
        )
    except Exception:
        preview = "**Conversation resumed.** Continue from where you left off."

    await cl.Message(content=preview).send()


# ── User feedback callbacks ────────────────────────────────────────────────────

@cl.action_callback("rate_good")
async def on_rate_good(action: cl.Action):
    user_id  = cl.user_session.get("user_id", "default")
    question = cl.user_session.get("last_question", "")
    if question:
        await asyncio.to_thread(
            _memory_store.save_direct,
            user_id,
            f"User confirmed accurate answer on: {question[:80]}",
            "preference", 2,
        )
    await cl.Message(content="✅ Marked as accurate.", author="System").send()


@cl.action_callback("rate_review")
async def on_rate_review(action: cl.Action):
    user_id  = cl.user_session.get("user_id", "default")
    question = cl.user_session.get("last_question", "")
    if question:
        await asyncio.to_thread(
            _memory_store.save_direct,
            user_id,
            f"Partially correct response on: {question[:80]} — be more thorough next time",
            "knowledge_gap", 2,
        )
    await cl.Message(content="⚠️ Noted — will be more thorough on similar topics.", author="System").send()


@cl.action_callback("rate_wrong")
async def on_rate_wrong(action: cl.Action):
    user_id  = cl.user_session.get("user_id", "default")
    question = cl.user_session.get("last_question", "")
    if question:
        await asyncio.to_thread(
            _memory_store.save_direct,
            user_id,
            f"Wrong answer flagged for: {question[:80]} — verify carefully before answering similar queries",
            "knowledge_gap", 3,
        )
    await cl.Message(content="🚫 Feedback saved — will be more careful on this topic.", author="System").send()


# ── Core graph runner ──────────────────────────────────────────────────────────

async def _run_graph(
    config: dict,
    user_input: str = None,
    resume: bool = False,
    user_id: str = "default",
    answer_language: str = "english",
    compare_doc_a: str = "",
    compare_doc_b: str = "",
    user_memories: list = None,
):
    if user_memories is None:
        user_memories = []

    web_search_enabled = cl.user_session.get("web_search_enabled", False)

    if resume:
        graph_input = None
    else:
        graph_input: dict = {
            "messages":          [HumanMessage(content=user_input)],
            "answer_language":   answer_language,
            "user_memories":     user_memories,
            "web_search_enabled": web_search_enabled,
        }
        if compare_doc_a and compare_doc_b:
            graph_input["query_type"]    = "compare"
            graph_input["compare_doc_a"] = compare_doc_a
            graph_input["compare_doc_b"] = compare_doc_b

    _token = user_id_ctx.set(user_id)

    response_msg = cl.Message(content="")
    await response_msg.send()

    _sources: set[str] = set()
    _reasoning_steps: list = []
    _ANSWER_NODES = {
        "aggregate_answers", "fallback_response",
        "text2sql_node", "diff_synthesizer", "reasoning_synthesizer",
    }

    try:
        async for event in _graph.astream_events(graph_input, config=config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Word-by-word token streaming for all final answer nodes
            if kind == "on_chat_model_stream" and node in _ANSWER_NODES:
                token = event["data"]["chunk"].content
                if token:
                    await response_msg.stream_token(token)

            # Show searching step the moment a tool call begins
            elif kind == "on_tool_start" and event.get("name") == "search_chunks":
                tool_input = event["data"].get("input") or {}
                query_str  = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                if query_str:
                    async with cl.Step(name=f"🔍 Searching: {query_str[:60]}", type="tool") as step:
                        step.input  = query_str
                        step.output = "Hybrid search (dense + BM25 + reranking)..."

            elif kind == "on_tool_start" and event.get("name") == "web_search":
                tool_input = event["data"].get("input") or {}
                query_str  = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                if query_str:
                    async with cl.Step(name=f"🌐 Web search: {query_str[:60]}", type="tool") as step:
                        step.input  = query_str
                        step.output = "Searching the web..."

            # Extract sources and show content preview when tool finishes
            elif kind == "on_tool_end" and event.get("name") == "search_chunks":
                raw    = event["data"].get("output", "")
                output = raw.content if hasattr(raw, "content") else str(raw) if raw else ""
                for hit in _SOURCE_PATTERN.finditer(output):
                    src = hit.group(1).strip()
                    if src:
                        _sources.add(src)
                if output and not output.startswith("NO_"):
                    preview = output[:400] + "..." if len(output) > 400 else output
                    async with cl.Step(name="📄 Retrieved parent chunks", type="tool") as step:
                        step.output = preview

            # Multi-hop: capture reasoning plan when planner node finishes
            elif kind == "on_chain_end" and node == "reasoning_planner":
                out = event["data"].get("output") or {}
                _reasoning_steps = out.get("reasoning_steps", [])

            # Multi-hop: show each completed reasoning step
            elif kind == "on_chain_end" and node == "execute_reasoning_step":
                out      = event["data"].get("output") or {}
                step_num = out.get("current_step_index", 1)
                total    = len(_reasoning_steps)
                step_q   = (
                    _reasoning_steps[step_num - 1]
                    if _reasoning_steps and 0 < step_num <= len(_reasoning_steps)
                    else f"Step {step_num}"
                )
                results = out.get("step_results", [])
                preview = results[-1][:300] if results else "No content retrieved"
                async with cl.Step(name=f"🔗 Step {step_num}/{total}: {step_q[:50]}", type="tool") as step:
                    step.output = preview

        final_state = _graph.get_state(config)

        if final_state.next and "request_clarification" in final_state.next:
            cl.user_session.set("awaiting_clarification", True)
            clarification = _extract_last_ai_message(final_state.values)
            try:
                if not response_msg.content:
                    response_msg.content = "..."
                await response_msg.update()
            except Exception:
                pass
            if clarification:
                await cl.Message(
                    content=f"**Clarification needed:**\n\n{clarification}"
                ).send()
            return

        if not response_msg.content:
            last_ai = _extract_last_ai_message(final_state.values)
            response_msg.content = last_ai or "The agent could not generate a response. Please try rephrasing."

        # Save clean answer (before any decoration)
        cl.user_session.set("last_answer",  response_msg.content)
        cl.user_session.set("last_sources", sorted(_sources))

        # Attach InsightCard inline — judge badge + source chips + Copy/Export buttons
        try:
            judge_badge  = final_state.values.get("judge_badge",  "")
            judge_reason = final_state.values.get("judge_reason", "")
            card = cl.CustomElement(
                name="InsightCard",
                display="inline",
                props={
                    "answer":      response_msg.content,
                    "sources":     sorted(_sources),
                    "judgebadge":  judge_badge,
                    "judgereason": judge_reason,
                },
            )
            response_msg.elements = [card]
        except Exception as _card_err:
            logger.warning(f"InsightCard element skipped: {_card_err}")

        await response_msg.update()

        # User feedback buttons
        if response_msg.content:
            await cl.Message(
                content="",
                actions=[
                    cl.Action(name="rate_good",   payload={}, label="✅ Accurate"),
                    cl.Action(name="rate_review", payload={}, label="⚠️ Partly right"),
                    cl.Action(name="rate_wrong",  payload={}, label="🚫 Wrong"),
                ],
            ).send()

        # Fire-and-forget memory extraction
        msgs_for_memory = list(final_state.values.get("messages", []))
        if msgs_for_memory and not final_state.next:
            asyncio.create_task(
                asyncio.to_thread(_memory_store.extract_and_save, user_id, msgs_for_memory, _llm)
            )

    except Exception as e:
        logger.error(f"_run_graph error: {type(e).__name__}: {e}", exc_info=True)
        err_map = {
            "RateLimitError": "The AI service is busy. Please try again in a moment.",
            "TimeoutError": "Request timed out. Please try a shorter question.",
            "GraphInterruptException": "Agent paused for clarification.",
        }
        friendly = err_map.get(type(e).__name__, "Something went wrong. Please try again.")
        response_msg.content = f"⚠️ {friendly}"
        await response_msg.update()
    finally:
        user_id_ctx.reset(_token)


# ── Study graph runner ─────────────────────────────────────────────────────────

async def _run_study_graph(
    config: dict,
    initial_state: dict = None,
    user_id: str = "default",
):
    _token = user_id_ctx.set(user_id)
    try:
        async for chunk in _study_graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, output in chunk.items():
                msgs = output.get("messages", [])
                for m in msgs:
                    if isinstance(m, AIMessage) and m.content:
                        await cl.Message(content=m.content).send()

        final_state = _study_graph.get_state(config)

        if not final_state.next:
            cl.user_session.set("study_active",    False)
            study_thread_id = config["configurable"]["thread_id"]
            score = final_state.values.get("score", 0.0)
            total = final_state.values.get("total",  0)
            _study_store.update_score(study_thread_id, score, total)
            _study_store.complete_session(study_thread_id)

    except Exception as e:
        logger.error(f"_run_study_graph error: {type(e).__name__}: {e}", exc_info=True)
        await cl.Message(content=f"Study session error: {str(e)[:200]}").send()
        cl.user_session.set("study_active", False)
    finally:
        user_id_ctx.reset(_token)


# ── File handling ──────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    safe = Path(name).name
    safe = safe.replace("..", "").strip()
    return safe or "upload.bin"


def _is_valid_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


async def _handle_file_elements(elements, user_id: str = "default") -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingested_names = []
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for element in elements:
        filename = _sanitize_filename(
            element.name if hasattr(element, "name") else Path(element.path).name
        )
        if not filename.lower().endswith(_SUPPORTED_EXT):
            await cl.Message(content=f"**{filename}** — unsupported type. Use PDF, DOCX, or TXT.").send()
            continue

        if hasattr(element, "path") and element.path:
            content = Path(element.path).read_bytes()
        elif hasattr(element, "content") and element.content:
            content = element.content
        else:
            await cl.Message(content=f"Could not read file: {filename}").send()
            continue

        if len(content) > max_bytes:
            await cl.Message(content=f"**{filename}** exceeds {MAX_UPLOAD_SIZE_MB}MB limit.").send()
            continue

        if filename.lower().endswith(".pdf") and not _is_valid_pdf(content):
            await cl.Message(content=f"**{filename}** does not appear to be a valid PDF.").send()
            continue

        dest = Path(DOCS_DIR) / filename
        dest.write_bytes(content)

        await cl.Message(content=f"Processing **{filename}**… this may take 1-2 minutes for large files.").send()
        async with cl.Step(name=f"📄 Ingesting {filename}", type="tool") as step:
            step.input = str(dest)
            stats = await asyncio.to_thread(_ingestion.ingest_single, str(dest), user_id)
            await asyncio.to_thread(_factory.rebuild_bm25)
            if "error" in stats:
                step.output = f"✗ Error: {stats['error']}"
            else:
                step.output = (
                    f"✓ {stats.get('parent_chunks', 0)} parent chunks | "
                    f"{stats.get('child_chunks', 0)} child chunks"
                )
                ingested_names.append(filename)

    await _send_doc_list(user_id)

    if ingested_names:
        await cl.Message(
            content=f"✅ **Ingested:** {', '.join(ingested_names)}\n\nYou can now ask questions about these documents."
        ).send()


# ── UI helpers ─────────────────────────────────────────────────────────────────

async def _send_doc_list(user_id: str = "default") -> None:
    docs = _ingestion.list_ingested_files(user_id=user_id)
    if docs:
        content = f"**Library ({len(docs)} document{'s' if len(docs) != 1 else ''})** — ask anything about them below."
    else:
        content = "📂 No documents yet. Drop a PDF, Word doc, or text file here to get started."
    try:
        card = cl.CustomElement(name="DocLibraryCard", display="inline", props={"docs": docs})
        await cl.Message(content=content, author="📚 Library", elements=[card]).send()
    except Exception:
        await cl.Message(content=content, author="📚 Library").send()


async def _send_history(user_id: str, current_thread_id: str) -> None:
    convs = _history.list_user(user_id, limit=3)
    past  = [c for c in convs if c["thread_id"] != current_thread_id and c["message_count"] > 0]
    if not past:
        return

    actions = [
        cl.Action(
            name="resume_conversation",
            payload={"thread_id": c["thread_id"]},
            label=f"↩ {c['title'][:45]}",
        )
        for c in past[:3]
    ]
    await cl.Message(
        content="**Recent conversations:**",
        actions=actions,
        author="History",
    ).send()


async def _send_study_history(user_id: str) -> None:
    """Show past study sessions — active ones get a Resume button."""
    sessions = await asyncio.to_thread(_study_store.list_user, user_id, 5)
    if not sessions:
        return

    active   = [s for s in sessions if s["status"] == "active"]
    finished = [s for s in sessions if s["status"] == "completed"]

    lines = []
    actions = []

    for s in active:
        pct = f"{s['score']:.0f}/{s['total']}" if s["total"] else "0/0"
        lines.append(f"📖 **{s['doc_name']}** — {pct} pts *(in progress)*")
        actions.append(
            cl.Action(
                name="resume_study",
                payload={"study_thread_id": s["study_thread_id"], "doc_name": s["doc_name"]},
                label=f"▶ Resume: {s['doc_name'][:30]}",
            )
        )

    for s in finished[:3]:
        pct = f"{s['score']:.0f}/{s['total']}" if s["total"] else "—"
        lines.append(f"✅ **{s['doc_name']}** — {pct} pts")

    if not lines:
        return

    await cl.Message(
        content="**Past study sessions:**\n\n" + "\n\n".join(lines),
        actions=actions or None,
        author="Study History",
    ).send()


def _extract_last_ai_message(state_values: dict) -> str:
    for msg in reversed(state_values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _welcome_text(user_id: str = "default") -> str:
    docs = _ingestion.list_ingested_files(user_id=user_id)
    doc_count = len(docs)
    if doc_count:
        doc_line = f"**{doc_count} document{'s' if doc_count > 1 else ''} ready**"
    else:
        doc_line = "**No documents yet**"
    return (
        f"## InsightEngine AI\n\n"
        f"{doc_line} · [Budget Dashboard](/public/budget.html)\n\n"
        f"Ask anything from your documents, or toggle web search for live information. "
        f"Drop a PDF, Word doc, or text file here to get started.\n\n"
        f"*Tip: Mention a filename to search a specific document — \"summarize Unit1_PartA.pdf\"*"
    )
