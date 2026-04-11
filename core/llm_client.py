"""
OpenAI-compatible client for LM Studio / Ollama / OpenAI.
Reads config from environment variables set in docker-compose.
"""
from __future__ import annotations
import logging, os
from dataclasses import dataclass
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    provider:   str   = "lmstudio"
    base_url:   str   = "http://host.docker.internal:1234/v1"
    model:      str   = "qwen/qwen3.5-9b"
    api_key:    str   = "lm-studio"
    timeout:    int   = 120
    max_tokens: int   = 512
    temperature:float = 0.85

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider    = os.getenv("LLM_PROVIDER",   "lmstudio"),
            base_url    = os.getenv("LLM_BASE_URL",    "http://host.docker.internal:1234/v1"),
            model       = os.getenv("LLM_MODEL",       "qwen/qwen3.5-9b"),
            api_key     = os.getenv("LLM_API_KEY",     "lm-studio"),
            timeout     = int(os.getenv("LLM_TIMEOUT", "120")),
            max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "512")),
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.85")),
        )

class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}",
                     "Content-Type": "application/json"},
            timeout=self.config.timeout,
        )

    async def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"[LLM unavailable: {exc}]"

    async def check_health(self) -> dict:
        try:
            resp = await self._client.get("/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            return {"ok": True, "models": [m.get("id","") for m in models]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def aclose(self):
        await self._client.aclose()

_default_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client