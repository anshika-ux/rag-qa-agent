import json
import sys
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from llm import chat_completion

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.
- Use search_documents for questions about the user's uploaded PDFs.
- Use web_search for current events or anything not in their documents.
- Use calculator for math.
Only call a tool when it's actually needed. If you already know the answer
confidently, just answer directly. Always explain your final answer clearly."""


def call_ollama(messages, tools=None):
    """Kept for backward compatibility; delegates to the shared llm module
    which auto-switches between local Ollama and hosted Groq."""
    return chat_completion(messages, tools=tools)


def run_agent(user_query, max_steps=5, verbose=True):
    """
    Runs the agent loop:
      1. Ask the model what to do (answer directly, or call a tool)
      2. If it calls a tool, execute it and feed the result back
      3. Repeat until the model gives a final answer or max_steps is hit
    Returns (final_answer, trace) where trace is a list of steps taken.
    """
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
            # model gave a final answer, no more tools needed
            final_answer = message.get("content", "").strip()
            trace.append({"type": "final_answer", "content": final_answer})
            if verbose:
                print(f"\n[Final answer after {step + 1} step(s)]")
            return final_answer, trace

        # model wants to call one or more tools
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

    # ran out of steps
    final_answer = "I wasn't able to fully resolve this within the step limit."
    trace.append({"type": "final_answer", "content": final_answer})
    return final_answer, trace


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask the agent: ")
    answer, trace = run_agent(query)
    print(f"\nAnswer: {answer}")