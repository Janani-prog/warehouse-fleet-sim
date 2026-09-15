"""Thin client for the local Ollama chat API - the only way the LLM agent is
served, per Locked Design Decision #3 (prompt-based via Ollama by default).
Uses Ollama's `format: "json"` option to force syntactically valid JSON
output (still not schema/enum-constrained - agent/executor.py does that
validation), and temperature 0 for reproducible action selection given the
same prompt, which matters for the agent accuracy eval and the causal
evaluation harness (M9) both needing repeatable behavior.
"""

from __future__ import annotations

import httpx

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaUnavailableError(RuntimeError):
    pass


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
) -> str:
    """Returns the raw string content of the model's reply (expected to be a
    JSON object per the system prompt's instructions - not parsed here,
    agent/executor.py owns validation)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = httpx.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaUnavailableError(
            f"could not reach Ollama at {OLLAMA_CHAT_URL} (is `ollama serve` running, "
            f"and is `{model}` pulled?): {e}"
        ) from e
    return resp.json()["message"]["content"]
