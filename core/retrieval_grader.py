"""
CRAG retrieval grader — evaluates whether retrieved chunks answer the question.

Uses a separate fast/cheap model (llama-3.1-8b-instant) so grading adds minimal
latency and does not consume quota from the main reasoning model.
"""

from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from .logging_config import get_logger

logger = get_logger(__name__)


class GradeResult(BaseModel):
    grade: Literal["relevant", "partial", "irrelevant"]
    reason: str


_GRADER_PROMPT = """You assess whether retrieved document chunks answer a user's question.

Grade using EXACTLY one of these three values:
  relevant    — the chunks directly and sufficiently answer the question
  partial     — the chunks are related but missing key details for a complete answer
  irrelevant  — the chunks are off-topic and do not address the question at all

Base your grade only on the provided content. Return a brief reason (1 sentence)."""


class RetrievalGrader:
    """
    Grades retrieved chunks for relevance to the user's question.
    Created once at startup in build_graph(); shared across all agent invocations.
    """

    def __init__(self, llm):
        self._grader = llm.with_structured_output(GradeResult)

    def grade(self, question: str, retrieved_content: str) -> GradeResult:
        """Grade whether retrieved_content answers question. Never raises — falls back to 'partial'."""
        try:
            result: GradeResult = self._grader.invoke([
                SystemMessage(content=_GRADER_PROMPT),
                HumanMessage(content=(
                    f"Question: {question}\n\n"
                    f"Retrieved content:\n{retrieved_content[:2000]}"
                )),
            ])
            logger.debug(f"CRAG grade={result.grade} | {result.reason[:80]}")
            return result
        except Exception as exc:
            logger.warning(f"RetrievalGrader.grade failed ({exc}) — defaulting to 'partial'")
            return GradeResult(grade="partial", reason="Grader unavailable.")
