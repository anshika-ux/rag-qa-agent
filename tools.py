"""
Tools the agent can choose to call. Each tool has:
  - a JSON schema (so the LLM knows how/when to call it)
  - a Python function that actually executes it

Add new tools by following the same pattern.
"""
from ddgs import DDGS
from query import retrieve


def search_documents(query: str) -> str:
    """RAG tool: search the user's uploaded PDFs."""
    results = retrieve(query, n_results=4)
    if not results:
        return "No relevant documents found."
    formatted = []
    for chunk, meta in results:
        formatted.append(f"[{meta['source']}] {chunk}")
    return "\n\n".join(formatted)


def web_search(query: str) -> str:
    """Search the live web for current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        formatted = []
        for r in results:
            formatted.append(f"{r['title']}: {r['body']} ({r['href']})")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


def calculator(expression: str) -> str:
    """Evaluate a basic math expression safely."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's uploaded PDF documents for relevant information. Use this for any question about the content of their documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the documents",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current information, news, or anything not likely to be in the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using +, -, *, /, parentheses",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "web_search": web_search,
    "calculator": calculator,
}