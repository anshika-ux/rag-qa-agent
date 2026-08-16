import json
import sys
import requests
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"   # must be a tool-calling capable model

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.
- Use search_documents for questions about the user's uploaded PDFs.
- Use web_search for current events or anything not in their documents.
- Use calculator for math.
Only call a tool when it's actually needed. If you already know the answer
confidently, just answer directly. Always explain your final answer clearly."""


def call_ollama(messages, tools=None):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def run_agent(user_query, max_steps=5, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    trace = []

    for step in range(max_steps):
        result = call_ollama(messages, tools=TOOL_SCHEMAS)
        message = result["message"]

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            final_answer = message.get("content", "").strip()
            trace.append({"type": "final_answer", "content": final_answer})
            if verbose:
                print(f"\n[Final answer after {step + 1} step(s)]")
            return final_answer, trace

        messages.append(message)

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            if verbose:
                print(f"[Step {step + 1}] Calling tool: {fn_name}({fn_args})")

            tool_fn = TOOL_FUNCTIONS.get(fn_name)
            if tool_fn is None:
                tool_result = f"Error: unknown tool '{fn_name}'"
            else:
                tool_result = tool_fn(**fn_args)

            trace.append(
                {
                    "type": "tool_call",
                    "tool": fn_name,
                    "args": fn_args,
                    "result": tool_result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                }
            )

    final_answer = "I wasn't able to fully resolve this within the step limit."
    trace.append({"type": "final_answer", "content": final_answer})
    return final_answer, trace


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask the agent: ")
    answer, trace = run_agent(query)
    print(f"\nAnswer: {answer}")