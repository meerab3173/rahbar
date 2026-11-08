"""The LLM provider interface (section 3.5). Swapping providers is one file.

A provider turns (question, passages) into a validated `Answer`, or raises
`GenerationError`. `ChatJSONProvider` holds everything the real providers share
(prompt, enforce-then-validate, one retry); a concrete provider only implements
`_chat`, the single HTTP call to its API.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from ...schemas import Answer, Passage
from ...services.prompt import RETRY_MESSAGE, build_messages
from ...telemetry import RETRIES

log = logging.getLogger("rahbar.llm")

# Tokens used by the current request (both attempts), read by services/answer.py.
usage_var: ContextVar[dict[str, int] | None] = ContextVar("usage", default=None)


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Answer the question from the passages, as an `Answer`."""
        ...


class GenerationError(RuntimeError):
    """The model could not produce a valid Answer, even after the retry."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class ChatReply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


def parse_answer(text: str) -> Answer:
    """Validate the model's text against the contract. Structure can be guaranteed by the
    provider's JSON mode; this still checks it, because the code must not trust it."""
    text = text.strip()
    if text.startswith("```"):                      # some models wrap JSON in a fence
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the reply")
    return Answer.model_validate(json.loads(text[start:end + 1]))


def _record(reply: ChatReply) -> None:
    usage = usage_var.get()
    if usage is None:
        usage = {"tokens_in": 0, "tokens_out": 0}
        usage_var.set(usage)
    usage["tokens_in"] += reply.tokens_in
    usage["tokens_out"] += reply.tokens_out


class ChatJSONProvider:
    name = "chat"

    def _chat(self, messages: list[dict[str, str]]) -> ChatReply:  # pragma: no cover
        raise TypeError(f"{type(self).__name__} must implement _chat")

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        messages = build_messages(question, passages)
        last_error = ""
        for attempt in (1, 2):                      # enforce, validate, retry once
            try:
                reply = self._chat(messages)
            except Exception as exc:                # timeout, connection refused, 5xx
                raise GenerationError("provider_error", f"{type(exc).__name__}: {exc}") from exc
            _record(reply)
            try:
                return parse_answer(reply.text)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc).splitlines()[0][:200]
                log.warning("model output failed validation",
                            extra={"attempt": attempt, "error": last_error})
                if attempt == 1:
                    RETRIES.inc()
                    messages = messages + [
                        {"role": "assistant", "content": reply.text[:2000]},
                        {"role": "user", "content": RETRY_MESSAGE.format(error=last_error)},
                    ]
        raise GenerationError("validation_failed", last_error)
