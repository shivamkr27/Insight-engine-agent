"""
Web search tool for InsightEngine AI.

Used only when the user has toggled web search ON and the documents
don't contain enough information to answer the question.
"""

from langchain_core.tools import tool
from .logging_config import get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web for current information not available in uploaded documents.
    Use ONLY when: (1) web search is enabled, (2) search_chunks returned NO_RESULTS
    or insufficient information, and (3) the question is about general/current topics.
    NEVER use for personal documents (resume, notes, lecture slides) — those are always
    in the uploaded files.

    Args:
        query: Focused English search query with key terms.

    Returns:
        Web search results as a formatted string.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        logger.info(f"Web search: '{query[:60]}' → {len(results)} chars")
        return f"Web search results for '{query}':\n\n{results}"
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Web search failed: {str(e)}. Please try again or rephrase your question."
