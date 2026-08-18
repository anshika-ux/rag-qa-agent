"""
Single place that decides which LLM backend to use:
  - If GROQ_API_KEY is set (env var or Streamlit secret) -> use Groq (free,
    hosted, works on Streamlit Cloud where Ollama can't run)
  - Otherwise -> use local Ollama (free, no key needed, for local dev)

Both query.py and agent.py import from here so there's one switching point.
"""
import os
import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def using_groq():
    return bool(_get_groq_key())


def simple_completion(prompt: str) -> str:
    groq_key = _get_groq_key()
    if groq_key:
        response = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {groq_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    else:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


def chat_completion(messages: list, tools: list = None) -> dict:
    groq_key = _get_groq_key()
    if groq_key:
        payload = {"model": GROQ_MODEL, "messages": messages}
        if tools:
            payload["tools"] = tools
        response = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {groq_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        choice_message = data["choices"][0]["message"]
        message = {
            "role": choice_message.get("role", "assistant"),
            "content": choice_message.get("content") or "",
        }
        if choice_message.get("tool_calls"):
            message["tool_calls"] = choice_message["tool_calls"]
        return {"message": message}
    else:
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()