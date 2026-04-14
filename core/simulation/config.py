from __future__ import annotations

from core.config import load_runtime_config

_RUNTIME_CONFIG = load_runtime_config()

# Source: Qwen3.5 docs (reasoning-token budget requires >=2048 output tokens)
LLM_MAX_TOKENS: int = int(_RUNTIME_CONFIG.llm_max_tokens)
