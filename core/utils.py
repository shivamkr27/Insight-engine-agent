import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .logging_config import get_logger

logger = get_logger(__name__)

# httpx is always present (required by langchain-groq → groq SDK → httpx).
# Catching httpx network errors covers Groq transient failures correctly.
try:
    import httpx
    _RETRY_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        httpx.TimeoutException,
        httpx.ConnectError,
    )
except ImportError:
    _RETRY_EXCEPTIONS = (TimeoutError, ConnectionError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def invoke_with_retry(llm, messages):
    """Invoke an LLM with automatic retry on transient network / timeout errors."""
    return llm.invoke(messages)
