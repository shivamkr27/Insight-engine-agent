"""
India Policy Intelligence Agent — Chainlit UI

Startup sequence (once per container):
  1. Load .env (GROQ_API_KEY)
  2. Init Ingestion   → loads ChromaDB + parent store
  3. Init Text2SQL    → loads budget_data.csv → SQLite
  4. Init ToolFactory → loads CrossEncoder + builds BM25 index
  5. Build LangGraph  → compile the full agent graph

Per-session (each browser tab / user):
  - Unique thread_id → separate conversation memory in InMemorySaver
  - HITL state flag  → tracks if graph is waiting for clarification
"""

import asyncio
import os
import sys
import io
import shutil
from pathlib import Path

# Make 'core' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with emoji in logs)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from core.config import GROQ_MODEL, DOCS_DIR
from core.ingestion import Ingestion
from core.tools import ToolFactory
from core.text2sql import Text2SQLEngine
from core.judge import HallucinationJudge
from core.graph import build_graph


# ── Global singletons — initialised once at startup ───────────────────────────
print("\n🏛️  Starting India Policy Intelligence Agent...\n")

_llm        = ChatGroq(model=GROQ_MODEL, temperature=0)
_ingestion  = Ingestion()
_sql_engine = Text2SQLEngine()
_judge      = HallucinationJudge()
_factory    = ToolFactory(_ingestion)
_graph      = build_graph(_llm, _factory, _sql_engine, _judge)

print("\n✅ Agent ready. Launching UI...\n")


# ── Chat lifecycle ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    # Each session gets its own thread_id for conversation memory
    thread_id = cl.context.session.id
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("awaiting_clarification", False)

    await cl.Message(
        content=_welcome_text(),
        actions=[
            cl.Action(name="upload_pdf", payload={"action": "upload"}, label="📤 Upload PDF"),
            cl.Action(name="ingest_all", payload={"action": "ingest"}, label="🔄 Ingest All Docs"),
        ],
    ).send()

    await _send_doc_list()


@cl.on_message
async def on_message(message: cl.Message):
    # PDF file attached directly in chat
    if message.elements:
        pdf_elements = [e for e in message.elements if str(e.name).lower().endswith(".pdf")]
        if pdf_elements:
            await _handle_pdf_elements(pdf_elements)
            return

    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}

    if cl.user_session.get("awaiting_clarification"):
        # Resume graph — add user clarification to state, then continue
        _graph.update_state(config, {"messages": [HumanMessage(content=message.content)]})
        cl.user_session.set("awaiting_clarification", False)
        await _run_graph(resume=True, config=config)
    else:
        await _run_graph(user_input=message.content, config=config)


# ── Sidebar button callbacks ───────────────────────────────────────────────────

@cl.action_callback("upload_pdf")
async def on_upload_action(action: cl.Action):
    files = await cl.AskFileMessage(
        content="Upload a government policy PDF (RBI report, Budget speech, Economic Survey, etc.)",
        accept=["application/pdf"],
        max_size_mb=50,
        timeout=180,
    ).send()
    if files:
        await _handle_pdf_elements(files)


@cl.action_callback("ingest_all")
async def on_ingest_all_action(action: cl.Action):
    await cl.Message(content="⏳ Scanning and ingesting all documents... this may take a minute.").send()
    async with cl.Step(name="📚 Ingesting all documents", type="tool") as step:
        step.input = f"Scanning: {DOCS_DIR}"
        stats = await asyncio.to_thread(_ingestion.ingest_all)
        await asyncio.to_thread(_factory.rebuild_bm25)
        ingested = len(stats["ingested"])
        skipped  = len(stats["skipped"])
        step.output = f"✓ {ingested} ingested | {skipped} already done"

    await _send_doc_list()
    if ingested:
        await cl.Message(content=f"✅ {ingested} document(s) ingested. Ready to answer questions!").send()
    else:
        await cl.Message(content="ℹ️ No new documents found. Upload PDFs first.").send()


# ── Core graph runner ──────────────────────────────────────────────────────────

async def _run_graph(config: dict, user_input: str = None, resume: bool = False):
    """
    Run the LangGraph agent and stream the response.

    Args:
        config:     LangGraph config with thread_id.
        user_input: New user message (None when resuming after clarification).
        resume:     True when continuing from a HITL interrupt.
    """
    graph_input = (
        None
        if resume
        else {"messages": [HumanMessage(content=user_input)]}
    )

    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Stream node-level updates from the graph
        async for chunk in _graph.astream(graph_input, config=config, stream_mode="updates"):
            for node_name, node_output in chunk.items():

                # ── Show tool call steps ──────────────────────────────
                if node_name == "orchestrator":
                    msgs = node_output.get("messages", [])
                    for m in msgs:
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                query_str = tc.get("args", {}).get("query", str(tc.get("args", {})))
                                async with cl.Step(name=f"🔍 Searching: {query_str[:60]}", type="tool") as step:
                                    step.input = query_str
                                    step.output = "Searching..."

                # ── Show search results as collapsed step ─────────────
                if node_name == "tools":
                    msgs = node_output.get("messages", [])
                    for m in msgs:
                        if hasattr(m, "content") and m.content:
                            preview = m.content[:400] + "..." if len(m.content) > 400 else m.content
                            async with cl.Step(name="📄 Retrieved context", type="tool") as step:
                                step.output = preview

                # ── Stream the main answer (RAG path) ─────────────────
                if node_name == "aggregate_answers":
                    msgs = node_output.get("messages", [])
                    for m in msgs:
                        if isinstance(m, AIMessage) and m.content:
                            await response_msg.stream_token(m.content)

                # ── Stream SQL result (SQL path) ──────────────────────
                if node_name == "text2sql_node":
                    msgs = node_output.get("messages", [])
                    for m in msgs:
                        if isinstance(m, AIMessage) and m.content:
                            await response_msg.stream_token(m.content)

        # Graph run complete — check final state
        final_state = _graph.get_state(config)

        # ── HITL interrupt: graph waiting for clarification ───────────
        if final_state.next and "request_clarification" in final_state.next:
            cl.user_session.set("awaiting_clarification", True)
            clarification = _extract_last_ai_message(final_state.values)
            if response_msg.content:
                await response_msg.update()
            else:
                await response_msg.remove()
            if clarification:
                await cl.Message(
                    content=f"🤔 **Clarification needed:**\n\n{clarification}"
                ).send()
            return

        # Finalize the response message
        if response_msg.content:
            await response_msg.update()
        else:
            # Graph ran but produced no streamed content — check state for fallback
            last_ai = _extract_last_ai_message(final_state.values)
            if last_ai:
                response_msg.content = last_ai
                await response_msg.update()
            else:
                response_msg.content = "⚠️ The agent could not generate a response. Please try rephrasing your question."
                await response_msg.update()

        # ── Show metadata badges ──────────────────────────────────────
        await _send_metadata_badges(final_state.values)

    except Exception as e:
        error_text = f"⚠️ **Error:** {type(e).__name__}: {str(e)[:300]}"
        response_msg.content = error_text
        await response_msg.update()
        print(f"[_run_graph ERROR] {type(e).__name__}: {e}")


# ── File handling ──────────────────────────────────────────────────────────────

async def _handle_pdf_elements(elements) -> None:
    """Save uploaded PDFs to docs/ and ingest them."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingested_names = []

    for element in elements:
        filename = element.name if hasattr(element, "name") else Path(element.path).name
        if not filename.lower().endswith(".pdf"):
            continue

        dest = os.path.join(DOCS_DIR, filename)

        # Copy from temp path or write bytes
        if hasattr(element, "path") and element.path:
            shutil.copy(element.path, dest)
        elif hasattr(element, "content") and element.content:
            with open(dest, "wb") as f:
                f.write(element.content)
        else:
            await cl.Message(content=f"⚠️ Could not read file: {filename}").send()
            continue

        await cl.Message(content=f"⏳ Processing **{filename}**... embedding may take 1-2 minutes for large PDFs.").send()
        async with cl.Step(name=f"📄 Ingesting {filename}", type="tool") as step:
            step.input = dest
            stats = await asyncio.to_thread(_ingestion.ingest_single, dest)
            await asyncio.to_thread(_factory.rebuild_bm25)
            if "error" in stats:
                step.output = f"✗ Error: {stats['error']}"
            else:
                step.output = (
                    f"✓ {stats.get('parent_chunks', 0)} parent chunks | "
                    f"{stats.get('child_chunks', 0)} child chunks"
                )
                ingested_names.append(filename)

    await _send_doc_list()

    if ingested_names:
        await cl.Message(
            content=f"✅ **Ingested:** {', '.join(ingested_names)}\n\nYou can now ask questions about these documents."
        ).send()


# ── UI helpers ─────────────────────────────────────────────────────────────────

async def _send_doc_list() -> None:
    """Show currently ingested documents."""
    docs = _ingestion.list_ingested_files()
    if docs:
        lines = "\n".join(f"• {d}" for d in docs)
        content = f"**📚 Ingested Documents ({len(docs)}):**\n{lines}"
    else:
        content = "📂 No documents ingested yet."
    await cl.Message(content=content, author="📚 Documents").send()


async def _send_metadata_badges(state_values: dict) -> None:
    """Show hallucination score badge and query type badge after each response."""
    badge  = state_values.get("judge_badge", "")
    reason = state_values.get("judge_reason", "")
    qtype  = state_values.get("query_type", "rag")
    score  = state_values.get("judge_score", 0)

    if not badge:
        return

    qtype_label = "📄 RAG — Document Search" if qtype == "rag" else "📊 SQL — Budget Data"
    score_label = f"Score: {score}/5"

    lines = [
        f"**Confidence:** {badge} ({score_label})  |  **Source:** {qtype_label}",
    ]
    if reason:
        lines.append(f"*{reason}*")

    await cl.Message(content="\n".join(lines), author="🔎 Evaluation").send()


def _extract_last_ai_message(state_values: dict) -> str:
    """Pull the most recent AIMessage content from state (for clarification display)."""
    for msg in reversed(state_values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _welcome_text() -> str:
    return """\
# 🏛️ India Policy Intelligence Agent

*Multi-agent RAG system for Indian government policies, RBI circulars, Union Budget & PM schemes.*

---

**To get started:**
1. Click **📤 Upload PDF** to add policy documents (RBI Annual Report, Budget Speech, Economic Survey, etc.)
2. Click **🔄 Ingest All Docs** to process everything in the docs/ folder
3. Ask your question below

---

**Sample questions you can ask:**

| Type | Example |
|------|---------|
| 📄 Policy | *What is RBI's inflation targeting framework?* |
| 📄 Scheme  | *Explain eligibility criteria for PM-KISAN* |
| 📊 Budget  | *Which ministry had the highest allocation in 2024?* |
| 📊 Trend   | *Compare MGNREGS spending in 2023 vs 2024* |

---

*Each response shows a **🟢 Verified / 🟡 Review / 🔴 Warning** badge based on hallucination scoring.*\
"""
