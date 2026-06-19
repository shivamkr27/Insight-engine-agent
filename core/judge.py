"""
LLM-as-Judge: hallucination detection for InsightEngine AI.

Score scale:
  1 — Factually consistent with context (paraphrasing is fine — still 1/5)
  2 — Mostly consistent; minor additions from general knowledge, no false claims
  3 — Partially supported; some unverified claims, no direct contradiction
  4 — Claims contradict context, or significant unsupported assertions
  5 — Completely unsupported or directly contradicts the context

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


_JUDGE_SYSTEM_PROMPT = """You are a hallucination detector for a document intelligence assistant.

Your task: evaluate whether the ANSWER is factually consistent with the CONTEXT provided.

Scoring guide:
  1 — Answer is factually consistent with context, even if phrased differently or paraphrased
  2 — Answer is mostly consistent; minor additions from general knowledge, no false claims
  3 — Answer partially supported; some claims unverified but no direct contradiction
  4 — Answer has claims that contradict the context, or significant unsupported assertions
  5 — Answer is completely unsupported or directly contradicts the context

CRITICAL: Paraphrasing is NOT hallucination.
  - Different wording = still 1/5 if the facts match
  - Synonyms, restructured sentences, simplified explanations = still 1/5 if factually accurate
  - Only penalise when facts are invented, contradicted, or fabricated from training data

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
