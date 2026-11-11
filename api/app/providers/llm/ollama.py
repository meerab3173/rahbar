"""Ollama provider: ollama-host or ollama-incluster (section 3.5).

Enforcement: Ollama's `format` parameter takes the Answer JSON schema and constrains
decoding to it. The reply is still validated with Pydantic in base.py.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from ...schemas import Answer
from .base import ChatJSONProvider, ChatReply


class OllamaProvider(ChatJSONProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings   # ollama_url, llm_model, temperature, llm_timeout
        self.url = settings.ollama_url.rstrip("/")
        self.schema = Answer.model_json_schema()

    def _chat(self, messages: list[dict[str, str]]) -> ChatReply:
        response = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.settings.llm_model,
                "messages": messages,
                "format": self.schema,
                "stream": False,
                "options": {"temperature": self.settings.temperature, "num_ctx": 8192},
            },
            timeout=httpx.Timeout(self.settings.llm_timeout, connect=5.0),
        )
        response.raise_for_status()
        body = response.json()
        return ChatReply(
            text=(body.get("message") or {}).get("content", ""),
            tokens_in=int(body.get("prompt_eval_count") or 0),
            tokens_out=int(body.get("eval_count") or 0),
        )
