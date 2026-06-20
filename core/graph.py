"""
LangGraph orchestration for the India Policy Intelligence Agent.

Pipeline:
  User question
    │
    ├─ summarize_history     Keep conversation context compact; inject user memories
    ├─ rewrite_query         Clarify + split into sub-questions (structured output)
    │   └─[unclear?]─► request_clarification  (HITL interrupt)
    ├─ route_query           Classify: rag | sql | multi_hop | compare
    │
    ├─[RAG]──► agent subgraph × N  (parallel, one per rewritten question)
    │           orchestrator → search_chunks → retrieval_grader → compress_context → collect_answer
    │           CRAG: if grade=irrelevant (<2 retries) → query_rewriter_loop → orchestrator
    │           └─► after_agents (router) ──► aggregate_answers (RAG)
    │                                    └──► diff_synthesizer  (compare)
    │
    ├─[SQL]──► text2sql_node   NL → SQL → SQLite → result string
    │
    ├─[multi_hop]──► reasoning_planner → execute_reasoning_step (self-loop) → reasoning_synthesizer
    │
    ├─[compare]──► two parallel agents, each locked to one doc → diff_synthesizer
    │
    ├─ hallucination_judge   Score answer 1-5; add badge to state
    └─ END

Checkpointer (SqliteSaver) is created via create_checkpointer() and shared
with the study graph so both use the same checkpoints.db.
"""

import operator
import sqlite3 as _sqlite3
from functools import partial
from typing import List, Set, Annotated, Literal, Optional

import tiktoken
from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, RemoveMessage, ToolMessage,
)
from langchain_core.globals import set_llm_cache
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.cache import SQLiteCache
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Command, Send
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import InMemorySaver

from .config import (
    MAX_TOOL_CALLS, MAX_ITERATIONS, GRAPH_RECURSION_LIMIT,
    BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR,
    SQLITE_CHECKPOINT_PATH,
)
from .prompts import (
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
    get_query_router_prompt,
    get_orchestrator_prompt,
    get_fallback_prompt,
    get_compress_prompt,
    get_aggregation_prompt,
    get_diff_synthesizer_prompt,
    get_multi_hop_synthesizer_prompt,
)
from .judge import HallucinationJudge
from .tools import ToolFactory, _format_search_results
from .retrieval_grader import RetrievalGrader
from .text2sql import Text2SQLEngine
from .utils import invoke_with_retry
from .logging_config import get_logger

logger = get_logger(__name__)

# Semantic LLM response cache — avoids redundant API calls for identical inputs
set_llm_cache(SQLiteCache(database_path=str(SQLITE_CHECKPOINT_PATH).replace("checkpoints.db", "llm_cache.db")))

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class QueryAnalysis(BaseModel):
    questions: List[str] = Field(
        description="One or more rewritten, self-contained search queries."
    )
    is_clear: bool = Field(
        description="True if the query intent is clear enough to search."
    )
    clarification_needed: Optional[str] = Field(
        default=None,
        description="What clarification is needed if is_clear is False."
    )

class QueryRoute(BaseModel):
    route: Literal["rag", "sql", "multi_hop"] = Field(
        description=(
            "rag for policy/text questions, sql for budget number queries, "
            "multi_hop for questions requiring chaining multiple distinct facts."
        )
    )

class ReasoningPlan(BaseModel):
    steps: List[str] = Field(
        description="2-4 ordered, specific search queries to answer the question step by step."
    )
    can_parallelize: bool = Field(
        default=False,
        description="True if steps are independent (rare — most multi-hop steps are sequential)."
    )


# ── State definitions ──────────────────────────────────────────────────────────

def _accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new

def _set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


class State(MessagesState):
    question_is_clear:   bool  = False
    conversation_summary: str  = ""
    original_query:      str   = ""
    rewritten_questions: List[str] = []
    agent_answers:       Annotated[List[dict], _accumulate_or_reset] = []
    query_type:          str   = "rag"
    sql_result:          str   = ""
    judge_score:         int   = 0
    judge_reason:        str   = ""
    judge_is_safe:       bool  = True
    judge_badge:         str   = ""
    # Feature: Hindi Answer Mode
    answer_language:     str   = "english"
    # Feature: Document Comparison Mode
    compare_doc_a:       str   = ""
    compare_doc_b:       str   = ""
    # Feature: Persistent Semantic Memory
    user_memories:       List[str] = []
    # Feature: Multi-Hop Reasoning
    reasoning_steps:     List[str] = []
    current_step_index:  int   = 0
    step_results:        List[str] = []
    # Feature: Web Search
    web_search_enabled:  bool  = False


class AgentState(MessagesState):
    question:       str   = ""
    question_index: int   = 0
    context_summary: str  = ""
    retrieval_keys: Annotated[Set[str], _set_union] = set()
    final_answer:   str   = ""
    agent_answers:  List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    # Propagated from main State
    doc_filter:         str  = ""
    answer_language:    str  = "english"
    user_memories:      List[str] = []
    web_search_enabled: bool = False
    # Feature: CRAG
    retrieval_attempts:   Annotated[int, operator.add] = 0
    last_retrieval_grade: str = ""


# ── Token estimation ───────────────────────────────────────────────────────────

def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        total += len(_TOKENIZER.encode(str(content)))
    return total


# ── Main graph nodes ───────────────────────────────────────────────────────────

def summarize_history(state: State, llm) -> dict:
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant = [
        m for m in state["messages"][:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ]
    if not relevant:
        return {"conversation_summary": ""}

    history_text = "Conversation history:\n"
    for m in relevant[-6:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_text += f"{role}: {m.content}\n"

    try:
        response = llm.invoke([
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=history_text),
        ])
        summary = response.content

        # Append user memories to the conversation summary so orchestrator sees them
        memories = state.get("user_memories", [])
        if memories:
            mem_block = "\n".join(f"- {m}" for m in memories)
            summary += f"\n\n[USER MEMORY]\n{mem_block}"

        return {
            "conversation_summary": summary,
            "agent_answers": [{"__reset__": True}],
        }
    except Exception as e:
        logger.error(f"summarize_history failed: {e}", exc_info=True)
        return {
            "conversation_summary": "",
            "agent_answers": [{"__reset__": True}],
        }


def rewrite_query(state: State, llm) -> dict:
    last_message = state["messages"][-1]
    summary = state.get("conversation_summary", "")

    context_block = ""
    if summary.strip():
        context_block = f"Conversation Context:\n{summary}\n\n"
    context_block += f"User Query:\n{last_message.content}"

    try:
        structured_llm = llm.with_structured_output(QueryAnalysis)
        result: QueryAnalysis = structured_llm.invoke([
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=context_block),
        ])

        if result.is_clear:
            delete_msgs = [
                RemoveMessage(id=m.id) for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            return {
                "question_is_clear": True,
                "messages": delete_msgs,
                "original_query": last_message.content,
                "rewritten_questions": result.questions,
            }

        clarification = (
            result.clarification_needed
            if result.clarification_needed and len(result.clarification_needed.strip()) > 10
            else "Could you clarify your question? More details will help me give a better answer."
        )
        return {
            "question_is_clear": False,
            "messages": [AIMessage(content=clarification)],
        }

    except Exception as e:
        logger.error(f"rewrite_query failed: {e}", exc_info=True)
        original = last_message.content
        return {
            "question_is_clear": True,
            "rewritten_questions": [original],
            "original_query": original,
            "messages": [],
        }


def request_clarification(state: State) -> dict:
    """
    HITL pause point. Graph compiles with interrupt_before=["request_clarification"],
    so execution pauses here and the UI shows the clarification message.
    """
    return {}


def route_query_node(state: State, llm) -> dict:
    # If compare mode was set from the UI, preserve it — don't let the LLM overwrite it
    if state.get("query_type") == "compare":
        return {"query_type": "compare"}

    combined_q = " | ".join(state["rewritten_questions"])
    try:
        structured_llm = llm.with_structured_output(QueryRoute)
        result: QueryRoute = structured_llm.invoke([
            SystemMessage(content=get_query_router_prompt()),
            HumanMessage(content=combined_q),
        ])
        return {"query_type": result.route}
    except Exception as e:
        logger.error(f"route_query_node failed: {e}", exc_info=True)
        return {"query_type": "rag"}


def text2sql_node(state: State, llm, sql_engine: Text2SQLEngine) -> dict:
    question = state["rewritten_questions"][0]
    lang     = state.get("answer_language", "english")
    try:
        sql_result = sql_engine.query(question, llm)
        label      = "बजट डेटा परिणाम" if lang == "hindi" else "Budget Data Result"
        formatted  = f"**{label}:**\n\n```\n{sql_result}\n```"
        return {
            "sql_result": sql_result,
            "messages":   [AIMessage(content=formatted)],
            "agent_answers": [{"index": 0, "question": question, "answer": sql_result}],
        }
    except Exception as e:
        logger.error(f"text2sql_node failed: {e}", exc_info=True)
        msg = "Unable to query budget data at this time. Please try again."
        return {
            "sql_result": "",
            "messages":   [AIMessage(content=msg)],
            "agent_answers": [{"index": 0, "question": question, "answer": msg}],
        }


def aggregate_answers(state: State, llm) -> dict:
    lang    = state.get("answer_language", "english")
    answers = [a for a in state.get("agent_answers", []) if not a.get("__reset__")]
    if not answers:
        return {"messages": [AIMessage(content="No answers were generated from the documents.")]}

    sorted_answers = sorted(answers, key=lambda x: x.get("index", 0))
    combined = "\n\n".join(
        f"Answer {i+1}:\n{a['answer']}"
        for i, a in enumerate(sorted_answers)
    )

    try:
        user_msg = HumanMessage(
            content=f"Original question: {state['original_query']}\n\nRetrieved answers:\n{combined}"
        )
        response = llm.invoke([
            SystemMessage(content=get_aggregation_prompt(language=lang)),
            user_msg,
        ])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"aggregate_answers failed: {e}", exc_info=True)
        fallback = sorted_answers[0]["answer"] if sorted_answers else "Unable to generate a response."
        return {"messages": [AIMessage(content=fallback)]}


def diff_synthesizer_node(state: State, llm) -> dict:
    """Synthesize a structured comparison when two agents searched different documents."""
    lang    = state.get("answer_language", "english")
    answers = [a for a in state.get("agent_answers", []) if not a.get("__reset__")]

    if len(answers) < 2:
        return aggregate_answers(state, llm)

    sorted_answers = sorted(answers, key=lambda x: x.get("index", 0))
    doc_a  = state.get("compare_doc_a", "Document A")
    doc_b  = state.get("compare_doc_b", "Document B")
    topic  = state.get("original_query", "the topic")

    combined = (
        f"Document A ({doc_a}):\n{sorted_answers[0]['answer']}\n\n"
        f"Document B ({doc_b}):\n{sorted_answers[1]['answer']}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=get_diff_synthesizer_prompt(language=lang)),
            HumanMessage(content=f"Comparison topic: {topic}\n\n{combined}"),
        ])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"diff_synthesizer_node failed: {e}", exc_info=True)
        return {"messages": [AIMessage(content=combined)]}


def hallucination_judge_node(state: State, llm, judge: HallucinationJudge) -> dict:
    question = state.get("original_query", "")

    final_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    if state.get("query_type") == "sql":
        context = state.get("sql_result", "")
    else:
        answers = [a for a in state.get("agent_answers", []) if not a.get("__reset__")]
        context = "\n\n---\n\n".join(a.get("answer", "") for a in answers)

    try:
        result = judge.score(question, context, final_answer, llm)
        return {
            "judge_score":   result["score"],
            "judge_reason":  result["reason"],
            "judge_is_safe": result["is_safe"],
            "judge_badge":   result["badge"],
        }
    except Exception as e:
        logger.error(f"hallucination_judge_node failed: {e}", exc_info=True)
        return {
            "judge_score":   1,
            "judge_reason":  "Judge unavailable.",
            "judge_is_safe": True,
            "judge_badge":   "🟢 Verified",
        }


# ── Multi-Hop Reasoning nodes ──────────────────────────────────────────────────

def reasoning_planner(state: State, llm) -> dict:
    """Break a complex question into 2-4 ordered search queries."""
    question = state.get("original_query", "") or (state.get("rewritten_questions") or [""])[0]

    try:
        planner_llm = llm.with_structured_output(ReasoningPlan)
        plan: ReasoningPlan = planner_llm.invoke([
            SystemMessage(content=(
                "Break this complex question into 2-4 sequential research steps. "
                "Each step must be a specific, focused search query. "
                "Steps must be ordered — later steps may reference findings from earlier steps. "
                "Return steps as a JSON list of strings."
            )),
            HumanMessage(content=f"Question: {question}"),
        ])
        steps = plan.steps[:4] if plan.steps else [question]
    except Exception as e:
        logger.error(f"reasoning_planner failed: {e}", exc_info=True)
        steps = [question]

    logger.info(f"Multi-hop plan: {len(steps)} steps for: {question[:60]}")
    return {
        "reasoning_steps":    steps,
        "current_step_index": 0,
        "step_results":       [],
    }


def execute_reasoning_step(state: State, tool_factory: ToolFactory) -> dict:
    """Execute one step of the multi-hop reasoning chain via direct hybrid search."""
    steps        = state.get("reasoning_steps", [])
    idx          = state.get("current_step_index", 0)
    prior        = list(state.get("step_results", []))

    if idx >= len(steps):
        return {"current_step_index": idx}

    query   = steps[idx]
    results = tool_factory._hybrid_search(query)
    content = _format_search_results(results, tool_factory._ingestion) if results else "No results found."

    logger.info(f"Multi-hop step {idx + 1}/{len(steps)}: '{query[:60]}'")
    return {
        "step_results":       prior + [content],
        "current_step_index": idx + 1,
    }


def reasoning_synthesizer(state: State, llm) -> dict:
    """Synthesize findings from all reasoning steps into a final answer."""
    lang     = state.get("answer_language", "english")
    question = state.get("original_query", "")
    steps    = state.get("reasoning_steps", [])
    results  = state.get("step_results", [])

    chain_text = "\n\n".join(
        f"**Step {i+1} — {steps[i] if i < len(steps) else ''}:**\n{r}"
        for i, r in enumerate(results)
    )

    try:
        response = llm.invoke([
            SystemMessage(content=get_multi_hop_synthesizer_prompt(language=lang)),
            HumanMessage(content=(
                f"Original question: {question}\n\n"
                f"Step-by-step research findings:\n\n{chain_text}"
            )),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"reasoning_synthesizer failed: {e}", exc_info=True)
        answer = chain_text or "Unable to synthesize multi-hop answer."

    return {
        "messages":    [AIMessage(content=answer)],
        "agent_answers": [{"index": 0, "question": question, "answer": answer}],
    }


# ── Agent subgraph nodes ───────────────────────────────────────────────────────

def orchestrator(state: AgentState, llm_no_web, llm_with_web) -> dict:
    lang               = state.get("answer_language", "english")
    doc_filter         = state.get("doc_filter", "")
    user_memories      = state.get("user_memories", [])
    web_search_enabled = state.get("web_search_enabled", False)

    # Hard enforcement: bind only the tools the toggle actually allows.
    # llm_no_web has search_chunks only; llm_with_web adds web_search.
    # Selecting here prevents the LLM from calling web_search even if it ignores the prompt.
    active_llm = llm_with_web if web_search_enabled else llm_no_web

    memory_context = "\n".join(f"- {m}" for m in user_memories) if user_memories else ""
    prompt_text    = get_orchestrator_prompt(
        language=lang,
        memory_context=memory_context,
        web_search_enabled=web_search_enabled,
    )

    if doc_filter:
        prompt_text += (
            f"\n\nDOCUMENT SCOPE: This agent is analysing ONLY the document '{doc_filter}'. "
            f"You MUST pass source_filter='{doc_filter}' in EVERY call to search_chunks."
        )

    sys_msg = SystemMessage(content=prompt_text)

    compressed = state.get("context_summary", "").strip()
    compressed_msgs = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{compressed}")]
        if compressed else []
    )

    try:
        if not state.get("messages"):
            human_msg = HumanMessage(content=state["question"])
            force_msg = HumanMessage(
                content="You MUST call search_chunks as your first action to find relevant documents."
            )
            response = invoke_with_retry(
                active_llm, [sys_msg] + compressed_msgs + [human_msg, force_msg]
            )
            return {
                "messages":        [human_msg, response],
                "tool_call_count": len(response.tool_calls or []),
                "iteration_count": 1,
            }

        response = invoke_with_retry(active_llm, [sys_msg] + compressed_msgs + state["messages"])
        return {
            "messages":        [response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }
    except Exception as e:
        logger.error(f"orchestrator failed: {e}", exc_info=True)
        return {
            "messages":        [AIMessage(content="Unable to process this request.")],
            "tool_call_count": 0,
            "iteration_count": MAX_ITERATIONS,
        }


def retrieval_grader_node(state: AgentState, grader: RetrievalGrader) -> dict:
    """Grade the most recent tool retrieval for relevance to the question."""
    last_tool_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            last_tool_content = msg.content
            break

    question = state.get("question", "")

    # Immediately flag empty / no-results responses as irrelevant
    if not last_tool_content or last_tool_content.startswith("NO_"):
        return {"last_retrieval_grade": "irrelevant"}

    result = grader.grade(question, last_tool_content)
    return {"last_retrieval_grade": result.grade}


def query_rewriter_loop(state: AgentState, llm) -> dict:
    """Rewrite the last failed search query and inject a hint for the orchestrator to retry."""
    question   = state.get("question", "")
    last_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                q = tc["args"].get("query", "")
                if q:
                    last_query = q
            break

    try:
        response = llm.invoke([
            SystemMessage(content=(
                "Rewrite a failed search query with different keywords. "
                "Return ONLY the new query — no quotes, no explanation."
            )),
            HumanMessage(content=(
                f"Original question: {question}\n"
                f"Failed query: {last_query}\n\n"
                f"Rewrite with different keywords to find this information:"
            )),
        ])
        new_query = response.content.strip()
    except Exception:
        new_query = question  # fallback to original question

    logger.info(f"CRAG rewrite: '{last_query[:50]}' → '{new_query[:50]}'")
    return {
        "messages":           [HumanMessage(content=(
            f"Previous search did not find relevant content. "
            f"Please search again using this query: {new_query}"
        ))],
        "retrieval_attempts": 1,  # accumulated via operator.add
    }


def should_compress_context(
    state: AgentState,
) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_keys: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                query = tc["args"].get("query", "")
                if query:
                    new_keys.add(f"search::{query}")
            break

    updated_keys = state.get("retrieval_keys", set()) | new_keys

    tokens_msgs    = _estimate_tokens(messages)
    tokens_summary = _estimate_tokens(
        [HumanMessage(content=state.get("context_summary", ""))]
    )
    total_tokens = tokens_msgs + tokens_summary
    max_allowed  = BASE_TOKEN_THRESHOLD + int(tokens_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if total_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_keys}, goto=goto)


def compress_context(state: AgentState, llm) -> dict:
    messages  = state["messages"]
    existing  = state.get("context_summary", "").strip()
    done_keys = state.get("retrieval_keys", set())

    conv_text = f"USER QUESTION:\n{state.get('question')}\n\n"
    if existing:
        conv_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls_info = " | Calls: " + ", ".join(
                    f"{tc['name']}({tc['args']})" for tc in msg.tool_calls
                )
            conv_text += f"[ASSISTANT{calls_info}]\n{msg.content or '(tool call)'}\n\n"
        elif isinstance(msg, ToolMessage):
            conv_text += f"[TOOL: {getattr(msg, 'name', 'tool')}]\n{msg.content}\n\n"

    try:
        response = llm.invoke([
            SystemMessage(content=get_compress_prompt()),
            HumanMessage(content=conv_text),
        ])
        new_summary = response.content
    except Exception as e:
        logger.error(f"compress_context failed: {e}", exc_info=True)
        new_summary = existing or conv_text[:2000]

    if done_keys:
        searches = sorted(k.replace("search::", "") for k in done_keys if k.startswith("search::"))
        if searches:
            new_summary += (
                "\n\n---\n**Already searched (do NOT repeat):**\n"
                + "\n".join(f"- {q}" for q in searches)
            )

    return {
        "context_summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }


def fallback_response(state: AgentState, llm) -> dict:
    lang = state.get("answer_language", "english")

    seen, unique_tool_outputs = set(), []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_tool_outputs.append(m.content)
            seen.add(m.content)

    compressed = state.get("context_summary", "").strip()
    parts = []
    if compressed:
        parts.append(f"## Compressed Research Context\n\n{compressed}")
    if unique_tool_outputs:
        parts.append(
            "## Retrieved Data\n\n"
            + "\n\n".join(f"--- Source {i} ---\n{c}" for i, c in enumerate(unique_tool_outputs, 1))
        )

    context_text = "\n\n".join(parts) or "No data was retrieved."
    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        "INSTRUCTION: Provide the best possible answer using only the data above."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=get_fallback_prompt(language=lang)),
            HumanMessage(content=prompt_content),
        ])
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"fallback_response failed: {e}", exc_info=True)
        return {"messages": [AIMessage(content="Unable to generate a response from retrieved data.")]}


def collect_answer(state: AgentState) -> dict:
    last = state["messages"][-1]
    is_valid = isinstance(last, AIMessage) and last.content and not getattr(last, "tool_calls", None)
    answer = last.content if is_valid else "Unable to generate an answer from the available documents."
    return {
        "final_answer": answer,
        "agent_answers": [{
            "index":    state["question_index"],
            "question": state["question"],
            "answer":   answer,
        }],
    }


# ── Edge routing functions ─────────────────────────────────────────────────────

def route_after_rewrite(state: State) -> str:
    if not state.get("question_is_clear", False):
        return "request_clarification"
    return "route_query"


def route_after_route_query(state: State):
    qt                 = state.get("query_type", "rag")
    lang               = state.get("answer_language", "english")
    web_search_enabled = state.get("web_search_enabled", False)

    if qt == "sql":
        return "text2sql_node"

    if qt == "multi_hop":
        return "reasoning_planner"

    if qt == "compare":
        topic = state["rewritten_questions"][0]
        doc_a = state.get("compare_doc_a", "")
        doc_b = state.get("compare_doc_b", "")
        return [
            Send("agent", {
                "question":            f"What does the document say about: {topic}",
                "question_index":      0,
                "doc_filter":          doc_a,
                "answer_language":     lang,
                "user_memories":       state.get("user_memories", []),
                "web_search_enabled":  web_search_enabled,
                "messages":            [],
                "context_summary":     "",
                "retrieval_keys":      set(),
                "tool_call_count":     0,
                "iteration_count":     0,
                "retrieval_attempts":  0,
            }),
            Send("agent", {
                "question":            f"What does the document say about: {topic}",
                "question_index":      1,
                "doc_filter":          doc_b,
                "answer_language":     lang,
                "user_memories":       state.get("user_memories", []),
                "web_search_enabled":  web_search_enabled,
                "messages":            [],
                "context_summary":     "",
                "retrieval_keys":      set(),
                "tool_call_count":     0,
                "iteration_count":     0,
                "retrieval_attempts":  0,
            }),
        ]

    # Default RAG
    return [
        Send("agent", {
            "question":            q,
            "question_index":      i,
            "doc_filter":          "",
            "answer_language":     lang,
            "user_memories":       state.get("user_memories", []),
            "web_search_enabled":  web_search_enabled,
            "messages":            [],
            "context_summary":     "",
            "retrieval_keys":      set(),
            "tool_call_count":     0,
            "iteration_count":     0,
            "retrieval_attempts":  0,
        })
        for i, q in enumerate(state["rewritten_questions"])
    ]


def route_after_orchestrator(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "fallback_response"
    if state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "fallback_response"

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "collect_answer"
    return "tools"


def route_after_grader(state: AgentState) -> str:
    grade    = state.get("last_retrieval_grade", "relevant")
    attempts = state.get("retrieval_attempts", 0)
    if grade == "relevant":
        return "should_compress_context"
    if grade == "irrelevant" and attempts < 2:
        return "query_rewriter_loop"
    return "should_compress_context"  # partial or max retries — proceed anyway


def route_reasoning_steps(state: State) -> str:
    if state.get("current_step_index", 0) < len(state.get("reasoning_steps", [])):
        return "execute_reasoning_step"
    return "reasoning_synthesizer"


def _after_agents_router(state: State) -> Command:
    """Fan-in router: sends to diff_synthesizer for compare mode, aggregate for RAG."""
    if state.get("query_type") == "compare":
        return Command(goto="diff_synthesizer")
    return Command(goto="aggregate_answers")


# ── Checkpointer factory ───────────────────────────────────────────────────────

async def create_checkpointer(db_path: str = SQLITE_CHECKPOINT_PATH):
    """Async checkpointer — must be called from an async context (on_chat_app_start)."""
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = await aiosqlite.connect(db_path)
        cp = AsyncSqliteSaver(conn)
        logger.info(f"Graph checkpointer: AsyncSqliteSaver ({db_path})")
        return cp
    except Exception as e:
        logger.warning(f"AsyncSqliteSaver unavailable ({e}) — using InMemorySaver")
        return InMemorySaver()


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph(
    llm,
    tool_factory: ToolFactory,
    sql_engine: Text2SQLEngine,
    judge: HallucinationJudge,
    checkpointer=None,
):
    if checkpointer is None:
        checkpointer = InMemorySaver()

    # Build two tool-bound LLMs: base (search_chunks only) and full (+ web_search).
    # The orchestrator selects at runtime based on web_search_enabled graph state —
    # hard enforcement so the toggle actually blocks the tool, not just the prompt.
    from .web_search import web_search as _web_search_tool
    rag_tools_base = tool_factory.create_rag_tools()
    rag_tools_full = rag_tools_base + [_web_search_tool]
    llm_no_web     = llm.bind_tools(rag_tools_base)
    llm_with_web   = llm.bind_tools(rag_tools_full)
    tool_node      = ToolNode(rag_tools_full)

    # CRAG grader — uses a separate fast model
    from .llm import get_grader_llm
    grader_llm = get_grader_llm()
    grader     = RetrievalGrader(grader_llm)

    # ── Agent subgraph ─────────────────────────────────────────────────────
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("orchestrator",            partial(orchestrator, llm_no_web=llm_no_web, llm_with_web=llm_with_web))
    agent_builder.add_node("tools",                   tool_node)
    agent_builder.add_node("retrieval_grader",        partial(retrieval_grader_node, grader=grader))
    agent_builder.add_node("query_rewriter_loop",     partial(query_rewriter_loop, llm=llm))
    agent_builder.add_node("should_compress_context", should_compress_context)
    agent_builder.add_node("compress_context",        partial(compress_context, llm=llm))
    agent_builder.add_node("collect_answer",          collect_answer)
    agent_builder.add_node("fallback_response",       partial(fallback_response, llm=llm))

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"},
    )
    # CRAG: tools → retrieval_grader → (should_compress_context | query_rewriter_loop → orchestrator)
    agent_builder.add_edge("tools",                   "retrieval_grader")
    agent_builder.add_conditional_edges(
        "retrieval_grader", route_after_grader,
        {"should_compress_context": "should_compress_context", "query_rewriter_loop": "query_rewriter_loop"},
    )
    agent_builder.add_edge("query_rewriter_loop",     "orchestrator")
    agent_builder.add_edge("compress_context",        "orchestrator")
    agent_builder.add_edge("fallback_response",       "collect_answer")
    agent_builder.add_edge("collect_answer",          END)

    agent_subgraph = agent_builder.compile()

    # ── Main graph ─────────────────────────────────────────────────────────
    main_builder = StateGraph(State)

    main_builder.add_node("summarize_history",       partial(summarize_history, llm=llm))
    main_builder.add_node("rewrite_query",           partial(rewrite_query, llm=llm))
    main_builder.add_node("request_clarification",   request_clarification)
    main_builder.add_node("route_query",             partial(route_query_node, llm=llm))
    main_builder.add_node("agent",                   agent_subgraph)
    main_builder.add_node("after_agents",            _after_agents_router)
    main_builder.add_node("aggregate_answers",       partial(aggregate_answers, llm=llm))
    main_builder.add_node("diff_synthesizer",        partial(diff_synthesizer_node, llm=llm))
    main_builder.add_node("text2sql_node",           partial(text2sql_node, llm=llm, sql_engine=sql_engine))
    # Multi-hop nodes
    main_builder.add_node("reasoning_planner",       partial(reasoning_planner, llm=llm))
    main_builder.add_node("execute_reasoning_step",  partial(execute_reasoning_step, tool_factory=tool_factory))
    main_builder.add_node("reasoning_synthesizer",   partial(reasoning_synthesizer, llm=llm))
    main_builder.add_node("hallucination_judge",     partial(hallucination_judge_node, llm=llm, judge=judge))

    main_builder.add_edge(START,                     "summarize_history")
    main_builder.add_edge("summarize_history",       "rewrite_query")
    main_builder.add_conditional_edges("rewrite_query",  route_after_rewrite)
    main_builder.add_edge("request_clarification",   "rewrite_query")
    main_builder.add_conditional_edges("route_query",    route_after_route_query)

    # Fan-in from parallel agents → router → aggregate or diff
    main_builder.add_edge(["agent"],                 "after_agents")
    main_builder.add_edge("aggregate_answers",       "hallucination_judge")
    main_builder.add_edge("diff_synthesizer",        "hallucination_judge")
    main_builder.add_edge("text2sql_node",           "hallucination_judge")

    # Multi-hop: reasoning_planner → execute_reasoning_step (self-loop) → reasoning_synthesizer
    main_builder.add_edge("reasoning_planner",       "execute_reasoning_step")
    main_builder.add_conditional_edges(
        "execute_reasoning_step", route_reasoning_steps,
        {"execute_reasoning_step": "execute_reasoning_step",
         "reasoning_synthesizer":  "reasoning_synthesizer"},
    )
    main_builder.add_edge("reasoning_synthesizer",   "hallucination_judge")

    main_builder.add_edge("hallucination_judge",     END)

    return main_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )
