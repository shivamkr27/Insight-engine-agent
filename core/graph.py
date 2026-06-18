"""
LangGraph orchestration for the India Policy Intelligence Agent.

Interview concept — full pipeline:

  User question
    │
    ├─ summarize_history     Keep conversation context compact
    ├─ rewrite_query         Clarify + split into sub-questions (structured output)
    │   └─[unclear?]─► request_clarification  (HITL interrupt)
    ├─ route_query           Classify: policy/text → RAG, budget/numbers → SQL
    │
    ├─[RAG]──► agent subgraph × N  (parallel, one per rewritten question)
    │           orchestrator → search_chunks → compress_context → collect_answer
    │           └─► aggregate_answers   Synthesise parallel answers into one
    │
    ├─[SQL]──► text2sql_node   NL → SQL → SQLite → result string
    │
    ├─ hallucination_judge   Score answer 1-5; add badge to state
    └─ END

State flows through both paths and converges at hallucination_judge.
"""

import operator
from functools import partial
from typing import List, Set, Annotated, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, RemoveMessage, ToolMessage,
)
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Command, Send
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from .config import (
    MAX_TOOL_CALLS, MAX_ITERATIONS, GRAPH_RECURSION_LIMIT,
    BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR,
)
from .prompts import (
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
    get_query_router_prompt,
    get_orchestrator_prompt,
    get_fallback_prompt,
    get_compress_prompt,
    get_aggregation_prompt,
)
from .judge import HallucinationJudge
from .tools import ToolFactory
from .text2sql import Text2SQLEngine


# ── Pydantic schemas for structured LLM outputs ───────────────────────────────

class QueryAnalysis(BaseModel):
    """Structured output from the rewrite_query node."""
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
    """Structured output from the route_query node."""
    route: Literal["rag", "sql"] = Field(
        description="rag for policy/text questions, sql for budget number questions."
    )


# ── State definitions ──────────────────────────────────────────────────────────

def _accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """
    Custom reducer for agent_answers.
    Sending [{"__reset__": True}] clears the list (new conversation turn).
    Otherwise, new answers are appended (parallel agents writing concurrently).
    """
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new

def _set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


class State(MessagesState):
    """State for the main graph — shared across all turns of a conversation."""
    question_is_clear: bool = False
    conversation_summary: str = ""
    original_query: str = ""
    rewritten_questions: List[str] = []
    agent_answers: Annotated[List[dict], _accumulate_or_reset] = []
    # Routing
    query_type: str = "rag"          # "rag" | "sql"
    sql_result: str = ""
    # Hallucination judge
    judge_score: int = 0
    judge_reason: str = ""
    judge_is_safe: bool = True
    judge_badge: str = ""


class AgentState(MessagesState):
    """State for one parallel RAG agent subgraph instance."""
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], _set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0


# ── Helper ─────────────────────────────────────────────────────────────────────

def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        total += len(str(content)) // 4
    return total


# ── Main graph nodes ───────────────────────────────────────────────────────────

def summarize_history(state: State, llm) -> dict:
    """
    Condense conversation history into a short summary.
    Only runs when there are enough messages (avoids redundant calls on first turn).
    """
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

    response = llm.invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=history_text),
    ])
    # Reset agent_answers for the new turn
    return {
        "conversation_summary": response.content,
        "agent_answers": [{"__reset__": True}],
    }


def rewrite_query(state: State, llm) -> dict:
    """
    Rewrite the user's question into clear, self-contained search queries.
    Uses structured output to get: questions list + is_clear flag.
    """
    last_message = state["messages"][-1]
    summary = state.get("conversation_summary", "")

    context_block = ""
    if summary.strip():
        context_block = f"Conversation Context:\n{summary}\n\n"
    context_block += f"User Query:\n{last_message.content}"

    structured_llm = llm.with_structured_output(QueryAnalysis)
    result: QueryAnalysis = structured_llm.invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=context_block),
    ])

    if result.is_clear:
        # Clear all messages — agents start fresh (summary is kept in state, not messages)
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


def request_clarification(state: State) -> dict:
    """
    Pause point for HITL (Human-in-the-Loop).
    The graph compiles with interrupt_before=["request_clarification"],
    so execution pauses here and the UI shows the clarification message.
    User's response resumes the graph from rewrite_query.
    """
    return {}


def route_query_node(state: State, llm) -> dict:
    """
    Classify rewritten questions as 'rag' (policy docs) or 'sql' (budget numbers).
    Uses structured output for reliable routing.
    """
    combined_q = " | ".join(state["rewritten_questions"])
    structured_llm = llm.with_structured_output(QueryRoute)
    result: QueryRoute = structured_llm.invoke([
        SystemMessage(content=get_query_router_prompt()),
        HumanMessage(content=combined_q),
    ])
    return {"query_type": result.route}


def text2sql_node(state: State, llm, sql_engine: Text2SQLEngine) -> dict:
    """
    Handle budget/number queries via Text2SQL.
    Takes the first rewritten question, converts to SQL, executes, formats result.
    """
    question = state["rewritten_questions"][0]
    sql_result = sql_engine.query(question, llm)

    formatted = f"**Budget Data Result:**\n\n```\n{sql_result}\n```"
    return {
        "sql_result": sql_result,
        "messages": [AIMessage(content=formatted)],
        # Store in agent_answers so hallucination_judge can access context
        "agent_answers": [{"index": 0, "question": question, "answer": sql_result}],
    }


def aggregate_answers(state: State, llm) -> dict:
    """
    Synthesise answers from all parallel RAG agents into one coherent response.
    Called only on the RAG path after all Send("agent", ...) subgraphs complete.
    """
    answers = [
        a for a in state.get("agent_answers", [])
        if not a.get("__reset__")
    ]
    if not answers:
        return {"messages": [AIMessage(content="No answers were generated from the documents.")]}

    sorted_answers = sorted(answers, key=lambda x: x.get("index", 0))
    combined = "\n\n".join(
        f"Answer {i+1}:\n{a['answer']}"
        for i, a in enumerate(sorted_answers)
    )

    user_msg = HumanMessage(
        content=f"Original question: {state['original_query']}\n\nRetrieved answers:\n{combined}"
    )
    response = llm.invoke([
        SystemMessage(content=get_aggregation_prompt()),
        user_msg,
    ])
    return {"messages": [AIMessage(content=response.content)]}


def hallucination_judge_node(state: State, llm, judge: HallucinationJudge) -> dict:
    """
    Score the final answer for hallucination BEFORE returning to the user.
    Reads the last AIMessage as the answer, and uses agent_answers as context.
    """
    question = state.get("original_query", "")

    # Get final answer from messages
    final_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    # Get retrieved context
    if state.get("query_type") == "sql":
        context = state.get("sql_result", "")
    else:
        answers = [
            a for a in state.get("agent_answers", [])
            if not a.get("__reset__")
        ]
        context = "\n\n---\n\n".join(a.get("answer", "") for a in answers)

    result = judge.score(question, context, final_answer, llm)
    return {
        "judge_score":   result["score"],
        "judge_reason":  result["reason"],
        "judge_is_safe": result["is_safe"],
        "judge_badge":   result["badge"],
    }


# ── Agent subgraph nodes ───────────────────────────────────────────────────────

def orchestrator(state: AgentState, llm_with_tools) -> dict:
    """
    ReAct loop: decide whether to call a tool or produce the final answer.
    On first iteration, forces a tool call so the agent always searches first.
    """
    sys_msg = SystemMessage(content=get_orchestrator_prompt())

    # Inject compressed context if it exists
    compressed = state.get("context_summary", "").strip()
    compressed_msgs = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{compressed}")]
        if compressed else []
    )

    if not state.get("messages"):
        # First call — force a search
        human_msg = HumanMessage(content=state["question"])
        force_msg  = HumanMessage(
            content="You MUST call search_chunks as your first action to find relevant documents."
        )
        response = llm_with_tools.invoke(
            [sys_msg] + compressed_msgs + [human_msg, force_msg]
        )
        return {
            "messages":        [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + compressed_msgs + state["messages"])
    return {
        "messages":        [response],
        "tool_call_count": len(response.tool_calls or []),
        "iteration_count": 1,
    }


def should_compress_context(
    state: AgentState,
) -> Command[Literal["compress_context", "orchestrator"]]:
    """
    Check token usage after each tool call.
    If context is growing too large, compress it before the next orchestrator call.
    This uses LangGraph's Command to both update state and route in one step.
    """
    messages = state["messages"]

    # Track retrieval keys to avoid duplicate searches
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
    """
    Summarise the current message history into a compact context block.
    Then delete the messages so the next orchestrator call starts lean.
    """
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

    response = llm.invoke([
        SystemMessage(content=get_compress_prompt()),
        HumanMessage(content=conv_text),
    ])
    new_summary = response.content

    # Append "already done" block so orchestrator doesn't repeat searches
    if done_keys:
        searches = sorted(k.replace("search::", "") for k in done_keys if k.startswith("search::"))
        if searches:
            new_summary += (
                "\n\n---\n**Already searched (do NOT repeat):**\n"
                + "\n".join(f"- {q}" for q in searches)
            )

    # Delete all messages after the first (keeps the original question msg)
    return {
        "context_summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }


def fallback_response(state: AgentState, llm) -> dict:
    """
    Max iterations reached — generate best possible answer from whatever was retrieved.
    """
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
            + "\n\n".join(
                f"--- Source {i} ---\n{c}"
                for i, c in enumerate(unique_tool_outputs, 1)
            )
        )

    context_text = "\n\n".join(parts) or "No data was retrieved."
    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        "INSTRUCTION: Provide the best possible answer using only the data above."
    )

    response = llm.invoke([
        SystemMessage(content=get_fallback_prompt()),
        HumanMessage(content=prompt_content),
    ])
    return {"messages": [response]}


def collect_answer(state: AgentState) -> dict:
    """Extract the final answer from the agent and store it for aggregation."""
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
    """After rewrite_query: go to clarification or proceed to routing."""
    if not state.get("question_is_clear", False):
        return "request_clarification"
    return "route_query"


def route_after_route_query(state: State):
    """
    After route_query_node classifies the query:
    - SQL  → text2sql_node (single edge)
    - RAG  → parallel Send() for each rewritten question
    """
    if state.get("query_type") == "sql":
        return "text2sql_node"
    # Fan-out: one agent subgraph per rewritten question
    return [
        Send("agent", {
            "question":       q,
            "question_index": i,
            "messages":       [],
            "context_summary": "",
            "retrieval_keys":  set(),
            "tool_call_count": 0,
            "iteration_count": 0,
        })
        for i, q in enumerate(state["rewritten_questions"])
    ]


def route_after_orchestrator(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """
    After each orchestrator call, decide next step:
    - Still has tool calls → run tools
    - Hit iteration/tool limit → fallback
    - No tool calls → collect answer (done)
    """
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "fallback_response"
    if state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "fallback_response"

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "collect_answer"
    return "tools"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph(
    llm,
    tool_factory: ToolFactory,
    sql_engine: Text2SQLEngine,
    judge: HallucinationJudge,
):
    """
    Compile the full LangGraph computation graph.

    Args:
        llm:          Initialised ChatGroq instance.
        tool_factory: Provides the hybrid-search RAG tool.
        sql_engine:   Text2SQL engine (SQLite + NL→SQL).
        judge:        Hallucination scoring judge.

    Returns:
        Compiled LangGraph StateGraph with InMemorySaver checkpointer.
    """
    rag_tools       = tool_factory.create_rag_tools()
    llm_with_tools  = llm.bind_tools(rag_tools)
    tool_node       = ToolNode(rag_tools)
    checkpointer    = InMemorySaver()

    # ── Agent subgraph ─────────────────────────────────────────────────────
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("orchestrator",            partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools",                   tool_node)
    agent_builder.add_node("should_compress_context", should_compress_context)
    agent_builder.add_node("compress_context",        partial(compress_context, llm=llm))
    agent_builder.add_node("collect_answer",          collect_answer)
    agent_builder.add_node("fallback_response",       partial(fallback_response, llm=llm))

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"},
    )
    agent_builder.add_edge("tools",              "should_compress_context")
    # should_compress_context uses Command — no add_conditional_edges needed
    agent_builder.add_edge("compress_context",   "orchestrator")
    agent_builder.add_edge("fallback_response",  "collect_answer")
    agent_builder.add_edge("collect_answer",     END)

    agent_subgraph = agent_builder.compile()

    # ── Main graph ─────────────────────────────────────────────────────────
    main_builder = StateGraph(State)

    main_builder.add_node("summarize_history",     partial(summarize_history, llm=llm))
    main_builder.add_node("rewrite_query",         partial(rewrite_query, llm=llm))
    main_builder.add_node("request_clarification", request_clarification)
    main_builder.add_node("route_query",           partial(route_query_node, llm=llm))
    main_builder.add_node("agent",                 agent_subgraph)
    main_builder.add_node("text2sql_node",         partial(text2sql_node, llm=llm, sql_engine=sql_engine))
    main_builder.add_node("aggregate_answers",     partial(aggregate_answers, llm=llm))
    main_builder.add_node("hallucination_judge",   partial(hallucination_judge_node, llm=llm, judge=judge))

    main_builder.add_edge(START,                    "summarize_history")
    main_builder.add_edge("summarize_history",      "rewrite_query")
    main_builder.add_conditional_edges("rewrite_query",     route_after_rewrite)
    main_builder.add_edge("request_clarification",  "rewrite_query")
    main_builder.add_conditional_edges("route_query",       route_after_route_query)
    main_builder.add_edge(["agent"],                "aggregate_answers")     # fan-in from parallel agents
    main_builder.add_edge("aggregate_answers",      "hallucination_judge")
    main_builder.add_edge("text2sql_node",          "hallucination_judge")
    main_builder.add_edge("hallucination_judge",    END)

    return main_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],  # HITL pause
    )
