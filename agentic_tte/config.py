"""Runtime settings, read from environment variables prefixed ``ATTE_`` (or a local ``.env``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATTE_", env_file=".env", extra="ignore", env_parse_none_str="none"
    )

    # LLM serving. On Hellbender, point ollama_host at the GPU service job (see hpc/README.md).
    llm_backend: Literal["ollama", "openai_compat"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    openai_compat_base_url: str = "http://localhost:8000/v1"  # vLLM, later
    model: str = "qwen3:32b"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 16384
    num_predict: int | None = 8192  # output token cap per call
    # Reasoning: true/false for qwen3; "low"/"medium"/"high" for gpt-oss; none = model default.
    think: bool | Literal["low", "medium", "high"] | None = None
    llm_timeout_s: float = 600.0
    llm_max_retries: int = 2

    # ClinicalTrials.gov
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_timeout_s: float = 30.0

    cache_dir: Path = Path(".cache/agentic_tte")


def get_settings() -> Settings:
    return Settings()
