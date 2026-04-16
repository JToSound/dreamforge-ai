from __future__ import annotations

from typing import Any


PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "narrative_generation": {
        "version": "narrative-v1.0",
        "owner": "dreamforge-core",
        "notes": "REM/NREM policy-gated narrative generation with sanitization guardrails.",
    }
}

MODEL_CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    "lmstudio": {
        "json_mode": "provider-dependent",
        "no_think_prefix": "provider-dependent",
        "health_check": "/models",
    },
    "ollama": {
        "json_mode": False,
        "no_think_prefix": True,
        "health_check": "/api/tags or /v1/models (adapter)",
    },
    "openai": {
        "json_mode": True,
        "no_think_prefix": False,
        "health_check": "/models",
    },
    "anthropic": {
        "json_mode": "adapter-dependent",
        "no_think_prefix": False,
        "health_check": "provider sdk",
    },
}


def get_llm_registry_snapshot(
    *,
    active_provider: str,
    active_model: str,
    prompt_profile_version: str,
) -> dict[str, Any]:
    provider_key = str(active_provider or "").lower()
    capabilities = MODEL_CAPABILITY_MATRIX.get(
        provider_key,
        {
            "json_mode": "unknown",
            "no_think_prefix": "unknown",
            "health_check": "unknown",
        },
    )
    return {
        "active": {
            "provider": active_provider,
            "model": active_model,
            "prompt_profile_version": prompt_profile_version,
        },
        "prompt_registry": PROMPT_REGISTRY,
        "model_capability_matrix": MODEL_CAPABILITY_MATRIX,
        "active_provider_capabilities": capabilities,
    }
