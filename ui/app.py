"""
India Policy Intelligence Agent — Chainlit UI

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
    os.environ.setdefault("LANGCHAIN_PROJECT", "india-policy-agent")

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

# ── Global singletons ──────────────────────────────────────────────────────────
logger.info("Starting India Policy Intelligence Agent...")

_llm          = get_llm()
_ingestion    = Ingestion()
_sql_engine   = Text2SQLEngine()
_judge        = HallucinationJudge()
_factory      = ToolFactory(_ingestion)
_checkpointer = create_checkpointer()
_graph        = build_graph(_llm, _factory, _sql_engine, _judge, _checkpointer)
_study_graph  = build_study_graph(_llm, _factory, _checkpointer)
_memory_store = UserMemoryStore(_ingestion._embeddings)
_limiter      = RateLimiter(max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW)
_history      = ConversationStore(db_path=HISTORY_DB_PATH)
_study_store  = StudyStore(db_path=STUDY_DB_PATH)

logger.info("Agent ready. Launching UI...")


# ── Auth — Google OAuth ────────────────────────────────────────────────────────

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


# ── Chat lifecycle ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    thread_id = cl.context.session.id
    user      = cl.context.current_user
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

    _history.upsert(thread_id, user_id)

    await cl.Message(
        content=_welcome_text(user_id),
        actions=[
            cl.Action(name="upload_pdf",    payload={"action": "upload"}, label="📤 Upload PDF"),
            cl.Action(name="ingest_all",    payload={"action": "ingest"}, label="🔄 Ingest All Docs"),
            cl.Action(name="toggle_hindi",  payload={},                   label="🇮🇳 Hindi Mode"),
            cl.Action(name="study_mode",    payload={},                   label="📚 Study Mode"),
            cl.Action(name="compare_mode",  payload={},                   label="📊 Compare Mode"),
        ],
    ).send()

    await _send_doc_list(user_id)
    await _send_history(user_id, thread_id)


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

    # PDF attached directly in chat
    if message.elements:
        pdf_elements = [e for e in message.elements if str(e.name).lower().endswith(".pdf")]
        if pdf_elements:
            await _handle_pdf_elements(pdf_elements, user_id=user_id)
            return

    # Rate limiting (keyed per session thread_id)
    if not _limiter.is_allowed(thread_id):
        await cl.Message(
            content="Rate limit reached. Please wait a moment before asking again."
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
        except Exception as e:
            logger.error(f"Study mode routing failed: {e}", exc_info=True)
            cl.user_session.set("study_active", False)

    # ── Compare Mode: send next message as comparison topic ───────────────
    compare_mode_active = cl.user_session.get("compare_mode", False)
    if compare_mode_active:
        doc_a = cl.user_session.get("compare_doc_a", "")
        doc_b = cl.user_session.get("compare_doc_b", "")
        cl.user_session.set("compare_mode", False)
        if doc_a and doc_b:
            user_memories = await asyncio.to_thread(
                _memory_store.fetch_relevant, user_id, message.content
            )
            await _run_graph(
                user_input=message.content,
                config=config,
                user_id=user_id,
                answer_language=cl.user_session.get("answer_language", "english"),
                compare_doc_a=doc_a,
                compare_doc_b=doc_b,
                user_memories=user_memories,
            )
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
        content="Upload a government policy PDF (RBI report, Budget speech, Economic Survey, etc.)",
        accept=["application/pdf"],
        max_size_mb=MAX_UPLOAD_SIZE_MB,
        timeout=180,
    ).send()
    if files:
        await _handle_pdf_elements(files, user_id=user_id)


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
        await cl.Message(content="No new documents found. Upload PDFs first.").send()


# ── Feature 1: Hindi Mode toggle ───────────────────────────────────────────────

@cl.action_callback("toggle_hindi")
async def on_toggle_hindi(action: cl.Action):
    current  = cl.user_session.get("answer_language", "english")
    new_lang = "hindi" if current == "english" else "english"
    cl.user_session.set("answer_language", new_lang)

    if new_lang == "hindi":
        await cl.Message(
            content="🇮🇳 **Hindi Mode** enabled. All answers will be in Hindi (Devanagari script).\n\n"
                    "Policy terms like PM-KISAN, RBI, repo rate will remain in English.",
            actions=[cl.Action(name="toggle_hindi", payload={}, label="🇬🇧 Switch to English")],
        ).send()
    else:
        await cl.Message(
            content="🇬🇧 **English Mode** enabled.",
            actions=[cl.Action(name="toggle_hindi", payload={}, label="🇮🇳 Switch to Hindi")],
        ).send()


# ── Feature 2: Study Mode ──────────────────────────────────────────────────────

@cl.action_callback("study_mode")
async def on_study_mode(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    docs    = _ingestion.list_ingested_files(user_id=user_id)

    if not docs:
        await cl.Message(content="No documents ingested yet. Upload and ingest PDFs first.").send()
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


# ── Feature 3: Compare Mode ────────────────────────────────────────────────────

@cl.action_callback("compare_mode")
async def on_compare_mode(action: cl.Action):
    user_id = cl.user_session.get("user_id", "default")
    docs    = _ingestion.list_ingested_files(user_id=user_id)

    if len(docs) < 2:
        await cl.Message(
            content="Need at least **2 ingested documents** to use Compare Mode. Upload more PDFs first."
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
        content="**📊 Compare Mode** — Select **Document A**:",
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
            f"**📊 Compare Mode Active**\n\n"
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
    await cl.Message(content="📊 Compare mode cancelled. Back to normal chat.").send()


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


@cl.action_callback("copy_answer")
async def on_copy_answer(action: cl.Action):
    answer   = cl.user_session.get("last_answer", "")
    question = cl.user_session.get("last_question", "")
    sources  = cl.user_session.get("last_sources", [])
    if not answer:
        await cl.Message(content="No answer to copy.").send()
        return
    src_block = "\n".join(f"- {s}" for s in sources) if sources else "_No sources_"
    copyable  = f"**Q:** {question}\n\n**A:** {answer}\n\n**Sources:**\n{src_block}"
    await cl.Message(content=f"```markdown\n{copyable}\n```", author="Copy").send()


@cl.action_callback("export_answer")
async def on_export_answer(action: cl.Action):
    answer   = cl.user_session.get("last_answer", "")
    question = cl.user_session.get("last_question", "")
    sources  = cl.user_session.get("last_sources", [])
    if not answer:
        await cl.Message(content="No answer to export.").send()
        return

    from datetime import datetime, timezone
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    src = "\n".join(f"- 📄 {s}" for s in sources) if sources else "_No sources_"
    md  = (
        f"# Policy Research Export\n\n"
        f"**Date:** {ts}  \n"
        f"**System:** India Policy Intelligence Agent\n\n"
        f"---\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Answer\n\n{answer}\n\n"
        f"## Sources\n\n{src}\n\n"
        f"---\n*Generated by India Policy Intelligence Agent*"
    )

    export_path = Path(DOCS_DIR).parent / "data" / "export.md"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(md, encoding="utf-8")

    await cl.Message(
        content="Export ready:",
        elements=[cl.File(name="policy_answer.md", path=str(export_path), display="inline")],
    ).send()


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

    if resume:
        graph_input = None
    else:
        graph_input: dict = {
            "messages":       [HumanMessage(content=user_input)],
            "answer_language": answer_language,
            "user_memories":   user_memories,
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

            # Show searching step the moment a tool call begins (query available immediately)
            elif kind == "on_tool_start" and event.get("name") == "search_chunks":
                tool_input = event["data"].get("input") or {}
                query_str  = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                if query_str:
                    async with cl.Step(name=f"🔍 Searching: {query_str[:60]}", type="tool") as step:
                        step.input  = query_str
                        step.output = "Hybrid search (dense + BM25 + reranking)..."

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
            if response_msg.content:
                await response_msg.update()
            else:
                await response_msg.remove()
            if clarification:
                await cl.Message(
                    content=f"**Clarification needed:**\n\n{clarification}"
                ).send()
            return

        if response_msg.content:
            await response_msg.update()
        else:
            last_ai = _extract_last_ai_message(final_state.values)
            if last_ai:
                response_msg.content = last_ai
                await response_msg.update()
            else:
                response_msg.content = "The agent could not generate a response. Please try rephrasing."
                await response_msg.update()

        cl.user_session.set("last_answer",  response_msg.content)
        cl.user_session.set("last_sources", sorted(_sources))

        if _sources:
            pills = "  ".join(f"📄 `{s}`" for s in sorted(_sources))
            await cl.Message(content=pills, author="Sources").send()

        await _send_metadata_badges(final_state.values)

        if response_msg.content:
            await cl.Message(
                content="",
                actions=[
                    cl.Action(name="copy_answer",  payload={}, label="📋 Copy as Markdown"),
                    cl.Action(name="export_answer", payload={}, label="📥 Export .md"),
                ],
            ).send()

        # Fire-and-forget memory extraction — runs in background after response delivered
        msgs_for_memory = list(final_state.values.get("messages", []))
        if msgs_for_memory and not final_state.next:
            asyncio.create_task(
                asyncio.to_thread(_memory_store.extract_and_save, user_id, msgs_for_memory, _llm)
            )

    except Exception as e:
        logger.error(f"_run_graph error: {type(e).__name__}: {e}", exc_info=True)
        response_msg.content = f"**Error:** {type(e).__name__}: {str(e)[:300]}"
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
            # Session reached END (show_final_score ran)
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
    return safe or "upload.pdf"


def _is_valid_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


async def _handle_pdf_elements(elements, user_id: str = "default") -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingested_names = []
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for element in elements:
        filename = _sanitize_filename(
            element.name if hasattr(element, "name") else Path(element.path).name
        )
        if not filename.lower().endswith(".pdf"):
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

        if not _is_valid_pdf(content):
            await cl.Message(content=f"**{filename}** does not appear to be a valid PDF.").send()
            continue

        dest = Path(DOCS_DIR) / filename
        dest.write_bytes(content)

        await cl.Message(content=f"Processing **{filename}**… embedding may take 1-2 minutes for large PDFs.").send()
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
        lines   = "\n".join(f"• {d}" for d in docs)
        content = f"**📚 Your Documents ({len(docs)}):**\n{lines}"
    else:
        content = "📂 No documents ingested yet. Upload PDFs to get started."
    await cl.Message(content=content, author="Documents").send()


async def _send_history(user_id: str, current_thread_id: str) -> None:
    convs = _history.list_user(user_id, limit=8)
    past  = [c for c in convs if c["thread_id"] != current_thread_id and c["message_count"] > 0]
    if not past:
        return

    actions = [
        cl.Action(
            name="resume_conversation",
            payload={"thread_id": c["thread_id"]},
            label=f"↩ {c['title'][:45]}",
        )
        for c in past[:6]
    ]
    await cl.Message(
        content="**Past conversations** — click to resume:",
        actions=actions,
        author="History",
    ).send()


async def _send_metadata_badges(state_values: dict) -> None:
    badge  = state_values.get("judge_badge", "")
    reason = state_values.get("judge_reason", "")
    qtype  = state_values.get("query_type", "rag")
    score  = state_values.get("judge_score", 0)

    if not badge:
        return

    _VERDICT_ICON = {"🟢 Verified": "✅", "🟡 Review": "⚠️", "🔴 Warning": "🚫"}
    icon        = _VERDICT_ICON.get(badge, "ℹ️")
    qtype_label = {
        "rag":       "RAG · Document Search",
        "sql":       "SQL · Budget Data",
        "compare":   "Compare · Two Documents",
        "multi_hop": "Multi-Hop · Chained Reasoning",
    }.get(qtype, "RAG · Document Search")

    lines = [f"{icon} **{badge}** · Score {score}/5 · {qtype_label}"]
    if reason:
        lines.append(f"> *{reason}*")

    await cl.Message(content="\n\n".join(lines), author="Evaluation").send()


def _extract_last_ai_message(state_values: dict) -> str:
    for msg in reversed(state_values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _welcome_text(user_id: str = "default") -> str:
    docs      = _ingestion.list_ingested_files(user_id=user_id)
    doc_count = len(docs)
    if doc_count:
        doc_status = f"**{doc_count} document{'s' if doc_count > 1 else ''} ready**"
    else:
        doc_status = "No documents yet — upload PDFs to get started"

    return f"""\
## 🏛️ India Policy Intelligence Agent

{doc_status} · [📊 Budget Dashboard](/public/budget.html)

*LangGraph multi-agent · Hybrid RAG · Hallucination judge · Per-user isolation*

---

**Get started:**
1. Click **📤 Upload PDF** to add policy documents
2. Click **🔄 Ingest All Docs** to process the docs/ folder
3. Ask your question below

**Features:**
- 🇮🇳 **Hindi Mode** — toggle to get answers in Hindi
- 📚 **Study Mode** — quiz yourself on any uploaded document
- 📊 **Compare Mode** — compare two documents side-by-side on any topic
- 🔗 **Multi-Hop** — automatically activated for complex chained questions
- 🧠 **Memory** — type *"what do you know about me?"* to see your preferences

**Sample questions:**

| | Example |
|--|---------|
| 📄 | *What is RBI's inflation targeting framework?* |
| 📄 | *Explain PM-KISAN eligibility criteria* |
| 📊 | *Which ministry had the highest allocation in 2024?* |
| 🔗 | *Did FRBM fiscal deficit target match Budget 2024 actuals?* |

Each answer shows a **✅ Verified / ⚠️ Review / 🚫 Warning** badge, source citations, and copy/export buttons.\
"""
