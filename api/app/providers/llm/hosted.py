"""Hosted free-tier provider over an OpenAI-compatible API (Groq, OpenRouter, Google AI
Studio's OpenAI endpoint).

The key arrives as LLM_API_KEY from the rahbar-secret Secret, created on the command
line. It never appears in a file you commit, and never in a log line.
Enforcement: `response_format` JSON mode; the reply is still validated in base.py.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from .base import ChatJSONProvider, ChatReply


class HostedProvider(ChatJSONProvider):
    name = "hosted"

    def __init__(self, settings: Settings) -> None:
        if not settings.hosted_base_url:
            raise ValueError("LLM_PROVIDER=hosted needs HOSTED_BASE_URL in the ConfigMap")
        self.settings = settings   # hosted_base_url, llm_model, llm_api_key, temperature
        self.url = settings.hosted_base_url.rstrip("/")

    def _chat(self, messages: list[dict[str, str]]) -> ChatReply:
        response = httpx.post(
            f"{self.url}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
            json={
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=httpx.Timeout(self.settings.llm_timeout, connect=5.0),
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return ChatReply(
            text=body["choices"][0]["message"]["content"] or "",
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )
