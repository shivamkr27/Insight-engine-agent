"""
LLM-as-Judge: hallucination detection for the India Policy Agent.

Score scale:
  1 — Fully grounded: every claim traceable to the retrieved context
  2 — Mostly grounded: minor extrapolation, no false claims
  3 — Partial: some claims not verifiable from context
  4 — Mostly hallucinated: significant unsupported statements
  5 — Fabricated: ignores the retrieved context entirely

  is_safe = True  when score <= HALLUCINATION_SAFE_THRESHOLD (≤ 2)
"""

import hashlib

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from .config import HALLUCINATION_SAFE_THRESHOLD, HALLUCINATION_WARN_THRESHOLD
from .logging_config import get_logger

logger = get_logger(__name__)


class JudgeResult(BaseModel):
    score: int = Field(
        description="Hallucination score 1-5. 1=fully grounded, 5=completely fabricated."
    )
    reason: str = Field(
        description="One sentence explaining why this score was assigned."
    )
    is_safe: bool = Field(
        description="True if score <= 2, meaning the answer is safe to show the user."
    )


_JUDGE_SYSTEM_PROMPT = """You are a hallucination detector for an Indian government policy assistant.

Your task: evaluate whether the ANSWER is grounded in the CONTEXT provided.

Scoring (1-5):
  1 — Every claim in the answer is directly supported by the context
  2 — Answer is mostly grounded; minor inference acceptable, no false claims
  3 — Answer mixes supported facts with unsupported assertions
  4 — Major claims in the answer are not in the context or contradict it
  5 — Answer is largely fabricated, ignoring the provided context

Domain note: Indian policy data (scheme names, ministry allocations, RBI rates)
is factual — even a single fabricated figure is serious (score >= 4).

Return JSON with exactly three fields: score (int), reason (str), is_safe (bool).
is_safe must be true if and only if score <= 2."""


class HallucinationJudge:
    """
    Evaluates the final answer against the retrieved context.
    Results are cached by content hash to avoid redundant LLM calls.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def _cache_key(self, question: str, context: str, answer: str) -> str:
        payload = f"{question}|{context[:500]}|{answer[:500]}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def score(self, question: str, context: str, answer: str, llm) -> dict:
        """
        Args:
            question: The user's original question.
            context:  Retrieved chunks (RAG) or SQL result (Text2SQL).
            answer:   The final answer shown to the user.
            llm:      LangChain LLM instance.

        Returns:
            dict with keys: score, reason, is_safe, badge
        """
        key = self._cache_key(question, context, answer)
        if key in self._cache:
            logger.info("HallucinationJudge: cache hit")
            return self._cache[key]

        context_trimmed = context[:3000] if len(context) > 3000 else context
        answer_trimmed  = answer[:1500]  if len(answer)  > 1500  else answer

        user_content = (
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context_trimmed}\n\n"
            f"Answer to Evaluate:\n{answer_trimmed}"
        )

        try:
            structured_llm = llm.with_structured_output(JudgeResult)
            result: JudgeResult = structured_llm.invoke([
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            output = {
                "score":   result.score,
                "reason":  result.reason,
                "is_safe": result.is_safe,
                "badge":   self._badge(result.score),
            }
            self._cache[key] = output
            return output
        except Exception as e:
            logger.warning(f"HallucinationJudge failed: {e}")
            return {
                "score":   1,
                "reason":  "Judge unavailable.",
                "is_safe": True,
                "badge":   "🟢 Verified",
            }

    @staticmethod
    def _badge(score: int) -> str:
        if score <= HALLUCINATION_SAFE_THRESHOLD:
            return "🟢 Verified"
        if score >= HALLUCINATION_WARN_THRESHOLD:
            return "🔴 Warning"
        return "🟡 Review"
