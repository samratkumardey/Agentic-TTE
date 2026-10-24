from agentic_tte.config import Settings
from agentic_tte.llm.base import LLMClient, LLMOutputError, LLMResponse, Message
from agentic_tte.llm.cache import ResponseCache
from agentic_tte.llm.ollama import OllamaClient
from agentic_tte.llm.openai_compat import OpenAICompatClient
from agentic_tte.llm.scripted import ScriptedClient


def make_client(settings: Settings, *, use_cache: bool = True) -> LLMClient:
    cache = ResponseCache(settings.cache_dir / "llm_cache.sqlite") if use_cache else None
    common = dict(
        temperature=settings.temperature,
        seed=settings.seed,
        cache=cache,
        max_retries=settings.llm_max_retries,
    )
    if settings.llm_backend == "ollama":
        return OllamaClient(
            settings.model,
            host=settings.ollama_host,
            num_ctx=settings.num_ctx,
            think=settings.think,
            timeout_s=settings.llm_timeout_s,
            **common,
        )
    return OpenAICompatClient(
        settings.model,
        base_url=settings.openai_compat_base_url,
        timeout_s=settings.llm_timeout_s,
        **common,
    )


__all__ = [
    "LLMClient",
    "LLMOutputError",
    "LLMResponse",
    "Message",
    "OllamaClient",
    "OpenAICompatClient",
    "ResponseCache",
    "ScriptedClient",
    "make_client",
]
