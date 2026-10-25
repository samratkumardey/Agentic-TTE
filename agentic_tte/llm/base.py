"""Backend-neutral LLM interface used by every agent.

Ported from and extending EmulatRx ``llm_zoo_rlhf.LLMZoo`` (MIT, TrialLab): a registry of
backends with a response cache. Extensions: schema-constrained JSON output validated with
Pydantic and retried with the validation error, model digests recorded for reproducibility,
and token accounting for the run timeline.
"""

from __future__ import annotations

import copy
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from agentic_tte.llm.cache import ResponseCache

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    model_digest: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached: bool = False


class LLMOutputError(RuntimeError):
    def __init__(self, message: str, last_text: str | None = None):
        super().__init__(message)
        self.last_text = last_text


class Usage(BaseModel):
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0  # requests that timed out or lost the connection
    prompt_tokens: int = 0
    completion_tokens: int = 0
    log: list[dict[str, Any]] = Field(default_factory=list)


class LLMClient(ABC):
    """Subclasses implement ``_complete``; caching, think-stripping and validation live here."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        seed: int | None = 42,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.cache = cache
        self.max_retries = max_retries
        self.usage = Usage()

    @abstractmethod
    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> LLMResponse: ...

    def model_digest(self) -> str | None:
        return None

    def complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        key = None
        if self.cache is not None:
            key = self.cache.key(
                model=self.model,
                digest=self.model_digest(),
                messages=[m.model_dump() for m in messages],
                schema=json_schema,
                options={"temperature": self.temperature, "seed": self.seed},
            )
            hit = self.cache.get(key)
            if hit is not None:
                self.usage.cache_hits += 1
                return LLMResponse.model_validate({**hit, "cached": True})

        resp = self._complete(messages, json_schema)
        resp.text = strip_think(resp.text)
        self.usage.calls += 1
        self.usage.prompt_tokens += resp.prompt_tokens or 0
        self.usage.completion_tokens += resp.completion_tokens or 0
        if key is not None:
            self.cache.put(key, resp.model_dump(exclude={"cached"}))
        return resp

    def structured(self, messages: list[Message], schema: type[T]) -> T:
        """Ask for JSON matching ``schema``; on invalid output, show the model its error and
        retry up to ``max_retries`` times."""
        json_schema = inline_refs(schema.model_json_schema())
        convo = list(messages)
        last_text = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self.complete(convo, json_schema)
            except httpx.TransportError as exc:
                # Timeouts and dropped connections fail this item only, never the whole draft.
                # Not retried: a runaway generation would just run away again.
                self.usage.failures += 1
                raise LLMOutputError(
                    f"{self.model} request failed: {type(exc).__name__}: {exc}", last_text
                ) from exc
            last_text = resp.text
            try:
                return schema.model_validate_json(extract_json(resp.text))
            except (ValidationError, ValueError) as exc:
                # The retry conversation differs, so a cached bad answer is never replayed.
                convo = [
                    *convo,
                    Message(role="assistant", content=resp.text),
                    Message(
                        role="user",
                        content=(
                            "Your previous reply did not match the required JSON schema:\n"
                            f"{_short(exc)}\nReply again with only the corrected JSON object."
                        ),
                    ),
                ]
        raise LLMOutputError(
            f"{self.model} did not return valid {schema.__name__} after "
            f"{self.max_retries + 1} attempts",
            last_text,
        )


def strip_think(text: str) -> str:
    """Remove reasoning blocks (DeepSeek-R1, Qwen3); EmulatRx did this for DeepSeek only."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag lost
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> str:
    """The JSON object in ``text``, tolerating markdown fences or prose around it."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    if t.startswith("{"):
        return t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return t[start : end + 1]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref``s so grammar-constrained decoders get a self-contained schema."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                merged = {**copy.deepcopy(defs[name]), **{k: v for k, v in node.items()
                                                         if k != "$ref"}}  # fmt: skip
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


def _short(exc: Exception, limit: int = 1500) -> str:
    msg = str(exc)
    if isinstance(exc, ValidationError):
        msg = json.dumps(
            [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()[:10]], default=str
        )
    return msg[:limit]
