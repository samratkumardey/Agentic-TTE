"""Ollama backend (``/api/chat``) with JSON-schema-constrained output.

Replaces EmulatRx ``llm_zoo_rlhf.OllamaModel``, which sent one user message with temperature
0.7 and no output constraint.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from agentic_tte.llm.base import LLMClient, LLMResponse, Message


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        num_ctx: int | None = 16384,
        num_predict: int | None = 8192,
        think: bool | Literal["low", "medium", "high"] | None = None,
        timeout_s: float = 600.0,
        http: httpx.Client | None = None,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        # Output cap: a runaway generation ends as invalid JSON (retried, then a per-item
        # failure) instead of hitting the HTTP timeout (gpt-oss:20b, exit run 2026-10-24).
        self.num_predict = num_predict
        # None leaves the model default; Ollama rejects ``think`` for models without reasoning.
        # gpt-oss takes a level ("low", "medium", "high") instead of true/false.
        self.think = think
        self._http = http or httpx.Client(timeout=timeout_s)
        self._digest: str | None = None

    def _complete(self, messages: list[Message], json_schema: dict[str, Any] | None) -> LLMResponse:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.num_predict:
            options["num_predict"] = self.num_predict
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = json_schema
        if self.think is not None:
            body["think"] = self.think
        resp = self._http.post(f"{self.host}/api/chat", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            model_digest=self.model_digest(),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def model_digest(self) -> str | None:
        """Digest of the loaded weights, from ``/api/tags``; recorded in every run manifest."""
        if self._digest is None:
            try:
                tags = self._http.get(f"{self.host}/api/tags").json().get("models", [])
            except (httpx.HTTPError, ValueError):
                return None
            names = {self.model, self.model if ":" in self.model else f"{self.model}:latest"}
            for m in tags:
                if m.get("name") in names or m.get("model") in names:
                    self._digest = m.get("digest")
                    break
        return self._digest

    def list_models(self) -> list[dict[str, Any]]:
        return self._http.get(f"{self.host}/api/tags").json().get("models", [])

    def server_version(self) -> str | None:
        try:
            return self._http.get(f"{self.host}/api/version").json().get("version")
        except (httpx.HTTPError, ValueError):
            return None
