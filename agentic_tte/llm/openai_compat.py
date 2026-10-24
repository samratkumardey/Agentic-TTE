"""OpenAI-compatible backend (``/v1/chat/completions``), for vLLM on Hellbender in a later phase.

vLLM supports ``response_format: json_schema`` with guided decoding, so agents need no change
when the backend switches from Ollama.
"""

from __future__ import annotations

from typing import Any

import httpx

from agentic_tte.llm.base import LLMClient, LLMResponse, Message


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 600.0,
        http: httpx.Client | None = None,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http or httpx.Client(timeout=timeout_s, headers=headers)

    def _complete(self, messages: list[Message], json_schema: dict[str, Any] | None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.temperature,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        resp = self._http.post(f"{self.base_url}/chat/completions", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM server {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"].get("content") or "",
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
