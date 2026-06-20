"""
LLM abstraction with multi-provider fallback.

Primary provider is controlled by LLM_PROVIDER env var (groq | openai | anthropic).
Falls back to other configured providers if the primary fails or rate-limits.
"""

import os
from .logging_config import get_logger

logger = get_logger(__name__)


def get_llm():
    """
    Return the primary LLM with fallbacks registered.
    Lazy-imports provider packages so missing optional deps don't crash startup.
    """
    from .config import GROQ_MODEL, LLM_TEMPERATURE, LLM_PROVIDER, LLM_TIMEOUT

    primary = os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
    logger.info(f"Initializing LLM — primary provider: {primary}")

    def _make_groq():
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)

    def _make_openai():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)

    def _make_anthropic():
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)

    providers = {
        "groq":      (_make_groq,      "GROQ_API_KEY"),
        "openai":    (_make_openai,    "OPENAI_API_KEY"),
        "anthropic": (_make_anthropic, "ANTHROPIC_API_KEY"),
    }

    factory, _ = providers.get(primary, providers["groq"])
    try:
        llm = factory()
    except Exception as e:
        logger.error(f"Primary LLM provider '{primary}' failed to initialize: {e}")
        for name, (fac, key_name) in providers.items():
            if name == primary:
                continue
            try:
                logger.warning(f"Falling back to provider: {name}")
                llm = fac()
                primary = name
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                f"All LLM providers failed to initialize. Primary error: {e}"
            ) from e

    fallbacks = []
    for name, (fac, key_name) in providers.items():
        if name == primary:
            continue
        if os.environ.get(key_name):
            try:
                fallbacks.append(fac())
                logger.info(f"Registered fallback: {name}")
            except Exception as exc:
                logger.warning(f"Could not init fallback provider {name}: {exc}")

    if fallbacks:
        return llm.with_fallbacks(fallbacks)
    return llm


def get_grader_llm():
    """Fast, cheap LLM for CRAG retrieval grading (llama-3.1-8b-instant via Groq)."""
    from .config import GRADER_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GRADER_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)
    except Exception as exc:
        logger.warning(f"Could not init grader LLM ({exc}), falling back to main LLM")
        return get_llm()
