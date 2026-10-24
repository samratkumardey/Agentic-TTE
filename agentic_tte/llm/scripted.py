"""A deterministic stand-in LLM for tests and offline demos."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentic_tte.llm.base import LLMClient, LLMResponse, Message

Responder = Callable[[list[Message], dict[str, Any] | None], str]


class ScriptedClient(LLMClient):
    """Answers with ``responder(messages, schema)`` or, if given a list, the next string."""

    def __init__(self, responses: Responder | list[str], model: str = "scripted", **kwargs: Any):
        super().__init__(model, **kwargs)
        self._responses = responses
        self.requests: list[tuple[list[Message], dict[str, Any] | None]] = []

    def _complete(self, messages: list[Message], json_schema: dict[str, Any] | None) -> LLMResponse:
        self.requests.append((messages, json_schema))
        if callable(self._responses):
            text = self._responses(messages, json_schema)
        else:
            text = self._responses.pop(0)
        return LLMResponse(text=text, model=self.model)
