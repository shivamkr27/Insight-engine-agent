"""
Study Mode LangGraph — quiz-based learning from uploaded documents.

Flow:
  START
    └► extract_topics       (LLM reads doc overview → list of topics)
          └► generate_question  (LLM generates Q from topic chunks)
                └── [INTERRUPT before evaluate_answer]
                          ↓  (user types answer → update_state → resume)
                    evaluate_answer   (LLM grades using retrieved context)
                          └► next_or_done     (advances indices)
                               ├── generate_question  (more questions remain)
                               └── show_final_score   (all questions done → END)

Checkpointer is shared with the main graph (same SqliteSaver, different thread_id prefix).
"""

import operator
from functools import partial
from typing import List, Annotated

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph, MessagesState

from .prompts import (
    get_topic_extractor_prompt,
    get_question_generator_prompt,
    get_answer_evaluator_prompt,
)
from .logging_config import get_logger

logger = get_logger(__name__)

QUESTIONS_PER_TOPIC = 3
MAX_TOPICS = 8


# ── State ──────────────────────────────────────────────────────────────────────

class StudyState(MessagesState):
    doc_name:           str   = ""
    topics:             List[str] = []
    topic_idx:          int   = 0
    q_idx:              int   = 0
    questions_per_topic: int  = QUESTIONS_PER_TOPIC
    current_question:   str   = ""
    current_context:    str   = ""
    score:              Annotated[float, operator.add] = 0.0
    total:              Annotated[int,   operator.add] = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _search_in_doc(query: str, doc_name: str, tool_factory) -> str:
    """Run hybrid search restricted to a single document."""
    from .tools import _format_search_results
    results = tool_factory._hybrid_search(query, source_filter=doc_name)
    if not results:
        return ""
    return _format_search_results(results, tool_factory._ingestion)


# ── Graph nodes ────────────────────────────────────────────────────────────────

def extract_topics_node(state: StudyState, llm, tool_factory) -> dict:
    doc_name = state["doc_name"]

    raw = _search_in_doc("overview introduction contents main sections", doc_name, tool_factory)
    if not raw:
        raw = _search_in_doc(doc_name.replace(".pdf", "").replace("_", " "), doc_name, tool_factory)

    if not raw:
        topics = [f"Main content of {doc_name}"]
        status_msg = (
            f"📚 **Study Session Started** — *{doc_name}*\n\n"
            f"Could not extract detailed topics. Starting with general questions.\n\n"
            f"*{QUESTIONS_PER_TOPIC} questions · Starting now...*"
        )
        return {
            "topics": topics,
            "messages": [AIMessage(content=status_msg)],
        }

    try:
        response = llm.invoke([
            SystemMessage(content=get_topic_extractor_prompt()),
            HumanMessage(content=f"Document: {doc_name}\n\nContent:\n{raw[:4000]}"),
        ])
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        topics = []
        for line in lines:
            clean = line.lstrip("0123456789. )").strip()
            if clean and len(clean) > 3:
                topics.append(clean)
        topics = topics[:MAX_TOPICS] if topics else [f"Main content of {doc_name}"]
    except Exception as e:
        logger.error(f"extract_topics_node failed: {e}", exc_info=True)
        topics = [f"Main content of {doc_name}"]

    total_qs = len(topics) * QUESTIONS_PER_TOPIC
    status_msg = (
        f"📚 **Study Session Started** — *{doc_name}*\n\n"
        f"Found **{len(topics)} topics**:\n"
        + "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
        + f"\n\n*{total_qs} questions total · Starting now...*"
    )
    return {
        "topics": topics,
        "messages": [AIMessage(content=status_msg)],
    }


def generate_question_node(state: StudyState, llm, tool_factory) -> dict:
    topics       = state["topics"]
    topic_idx    = state.get("topic_idx", 0)
    q_idx        = state.get("q_idx", 0)
    doc_name     = state["doc_name"]
    score        = state.get("score", 0.0)
    total        = state.get("total", 0)
    qs_per_topic = state.get("questions_per_topic", QUESTIONS_PER_TOPIC)

    if topic_idx >= len(topics):
        return {"current_question": "", "current_context": ""}

    topic = topics[topic_idx]
    raw   = _search_in_doc(topic, doc_name, tool_factory)
    if not raw:
        raw = _search_in_doc(f"{topic} policy implementation", doc_name, tool_factory)

    try:
        response = llm.invoke([
            SystemMessage(content=get_question_generator_prompt()),
            HumanMessage(content=(
                f"Topic: {topic}\n"
                f"This is question {q_idx + 1} of {qs_per_topic} on this topic.\n\n"
                f"Retrieved content:\n{raw[:3000]}"
            )),
        ])
        question, hint = "", ""
        for line in response.content.strip().split("\n"):
            if line.startswith("QUESTION:"):
                question = line[len("QUESTION:"):].strip()
            elif line.startswith("HINT:"):
                hint = line[len("HINT:"):].strip()
        if not question:
            question = response.content.strip()
    except Exception as e:
        logger.error(f"generate_question_node failed: {e}", exc_info=True)
        question = f"Explain the key aspects of '{topic}' as described in the document."
        hint     = ""

    overall_q    = topic_idx * qs_per_topic + q_idx + 1
    total_q_sess = len(topics) * qs_per_topic
    score_str    = f"{score:.1f}" if score != int(score) else str(int(score))

    question_msg = (
        f"---\n"
        f"**Topic {topic_idx + 1}/{len(topics)}: {topic}**"
        f"  ·  Question {overall_q}/{total_q_sess}"
        f"  ·  Score: {score_str}/{total}\n\n"
        f"**Q{q_idx + 1}:** {question}"
        + (f"\n\n> 💡 *Hint: {hint}*" if hint else "")
    )
    return {
        "current_question": question,
        "current_context":  raw,
        "messages":         [AIMessage(content=question_msg)],
    }


def evaluate_answer_node(state: StudyState, llm, tool_factory) -> dict:
    # Last human message is the student's answer
    user_answer = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_answer = msg.content
            break

    if not user_answer:
        return {"messages": [AIMessage(content="I didn't receive your answer. Please type your answer below.")]}

    question = state.get("current_question", "")
    context  = state.get("current_context", "")

    try:
        response = llm.invoke([
            SystemMessage(content=get_answer_evaluator_prompt()),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Student's Answer: {user_answer}\n\n"
                f"Document Context (ground truth):\n{context[:3000]}"
            )),
        ])
        score_delta, verdict, explanation = 0.0, "", ""
        for line in response.content.strip().split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score_delta = float(line[len("SCORE:"):].strip())
                except ValueError:
                    score_delta = 0.0
            elif line.startswith("VERDICT:"):
                verdict = line[len("VERDICT:"):].strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line[len("EXPLANATION:"):].strip()
    except Exception as e:
        logger.error(f"evaluate_answer_node failed: {e}", exc_info=True)
        score_delta, verdict, explanation = 0.0, "Unable to evaluate", ""

    feedback = f"**{verdict}**"
    if explanation:
        feedback += f"\n\n{explanation}"

    return {
        "score":    score_delta,   # accumulated via operator.add
        "total":    1,             # accumulated via operator.add
        "messages": [AIMessage(content=feedback)],
    }


def next_or_done_node(state: StudyState) -> dict:
    """Advance topic/question indices after evaluation."""
    qs_per_topic = state.get("questions_per_topic", QUESTIONS_PER_TOPIC)
    q_idx        = state.get("q_idx", 0)
    topic_idx    = state.get("topic_idx", 0)

    next_q = q_idx + 1
    next_t = topic_idx
    if next_q >= qs_per_topic:
        next_q = 0
        next_t = topic_idx + 1

    return {"topic_idx": next_t, "q_idx": next_q}


def show_final_score_node(state: StudyState) -> dict:
    score    = state.get("score", 0.0)
    total    = state.get("total", 0)
    topics   = state.get("topics", [])
    doc_name = state.get("doc_name", "document")

    pct     = (score / total * 100) if total > 0 else 0
    emoji   = "🏆" if pct >= 80 else "📈" if pct >= 60 else "📖"
    verdict = (
        "Great work! You have a strong grasp of this document."
        if pct >= 80 else
        "Good effort. Review the topics you missed and try again."
        if pct >= 60 else
        "Keep studying. Re-read the document and restart the quiz."
    )
    score_str = f"{score:.1f}" if score != int(score) else str(int(score))
    topics_preview = ", ".join(topics[:4]) + ("..." if len(topics) > 4 else "")

    summary = (
        f"## {emoji} Study Session Complete!\n\n"
        f"**Document:** {doc_name}  \n"
        f"**Final Score:** {score_str} / {total}  \n"
        f"**Percentage:** {pct:.0f}%\n\n"
        f"*Topics covered: {topics_preview}*\n\n"
        f"{verdict}"
    )
    return {"messages": [AIMessage(content=summary)]}


# ── Edge routing ───────────────────────────────────────────────────────────────

def _route_after_next_or_done(state: StudyState) -> str:
    if state.get("topic_idx", 0) >= len(state.get("topics", [])):
        return "show_final_score"
    return "generate_question"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_study_graph(llm, tool_factory, checkpointer):
    """
    Compile the study-mode graph.
    Shares the same checkpointer as the main graph — thread_id separation is
    achieved via the 'study_' prefix used in ui/app.py.
    """
    builder = StateGraph(StudyState)

    builder.add_node("extract_topics",    partial(extract_topics_node,    llm=llm, tool_factory=tool_factory))
    builder.add_node("generate_question", partial(generate_question_node, llm=llm, tool_factory=tool_factory))
    builder.add_node("evaluate_answer",   partial(evaluate_answer_node,   llm=llm, tool_factory=tool_factory))
    builder.add_node("next_or_done",      next_or_done_node)
    builder.add_node("show_final_score",  show_final_score_node)

    builder.add_edge(START,               "extract_topics")
    builder.add_edge("extract_topics",    "generate_question")
    builder.add_edge("generate_question", "evaluate_answer")   # INTERRUPT before this
    builder.add_edge("evaluate_answer",   "next_or_done")
    builder.add_conditional_edges(
        "next_or_done",
        _route_after_next_or_done,
        {"generate_question": "generate_question", "show_final_score": "show_final_score"},
    )
    builder.add_edge("show_final_score", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["evaluate_answer"],
    )
