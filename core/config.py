from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    # API / dashboard
    api_base_url: str = "http://api:8000"

    # LLM defaults
    llm_provider: str = "lmstudio"
    llm_model: str = "qwen/qwen3.5-9b"
    llm_api_key: str = "lm-studio"
    llm_timeout: int = 120
    # Source: Qwen3.5 docs (reasoning models can exhaust a 512-token cap)
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.85
    llm_ollama_base_url: str = "http://localhost:11434"
    llm_lmstudio_base_url: str = "http://localhost:1234/v1"
    llm_base_url: str = "http://host.docker.internal:1234/v1"
    llm_timeout_seconds: float = 15.0

    # Simulation defaults
    simulation_duration_hours: float = 8.0
    simulation_dt_minutes: float = 0.5
    simulation_stress_level: float = 0.3
    simulation_sleep_start_hour: float = 23.0
    simulation_request_timeout_seconds: float = 3600.0


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        api_base_url=os.getenv("API_BASE_URL", "http://api:8000"),
        llm_provider=os.getenv("LLM_PROVIDER", "lmstudio"),
        llm_model=os.getenv("LLM_MODEL", "qwen/qwen3.5-9b"),
        llm_api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        llm_timeout=_env_int("LLM_TIMEOUT", 120),
        # Source: Qwen3.5 docs (reasoning models can exhaust a 512-token cap)
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.85),
        llm_ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_lmstudio_base_url=os.getenv(
            "LMSTUDIO_BASE_URL", "http://localhost:1234/v1"
        ),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1"),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 15.0),
        simulation_duration_hours=_env_float("SIM_DURATION_HOURS", 8.0),
        simulation_dt_minutes=_env_float("SIM_DT_MINUTES", 0.5),
        simulation_stress_level=_env_float("SIM_STRESS_LEVEL", 0.3),
        simulation_sleep_start_hour=_env_float("SIM_SLEEP_START_HOUR", 23.0),
        simulation_request_timeout_seconds=_env_float(
            "SIM_REQUEST_TIMEOUT_SECONDS", 3600.0
        ),
    )
