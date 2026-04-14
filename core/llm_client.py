"""Robust async LLM client with retries and exponential backoff.

Provides a thin wrapper around an HTTP-based LLM backend and exposes a
simple `chat(system, user)` coroutine. The client retries transient
errors and logs failures; it returns a clear sentinel string on
permanent failure so callers can fallback to template generators.
"""

from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional
import httpx

from core.config import load_runtime_config

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "lmstudio"
    base_url: str = "http://host.docker.internal:1234/v1"
    model: str = "qwen/qwen3.5-9b"
    api_key: str = "lm-studio"
    timeout: int = 120
    # Source: Qwen3.5 docs (reasoning-token budget requires >=2048 output tokens)
    max_tokens: int = 2048
    temperature: float = 0.85
    # Resilience knobs
    retries: int = 3
    backoff_base: float = 0.5

    @classmethod
    def from_env(cls) -> "LLMConfig":
        runtime = load_runtime_config()
        return cls(
            provider=os.getenv("LLM_PROVIDER", runtime.llm_provider),
            base_url=os.getenv("LLM_BASE_URL", runtime.llm_base_url),
            model=os.getenv("LLM_MODEL", runtime.llm_model),
            api_key=os.getenv("LLM_API_KEY", runtime.llm_api_key),
            timeout=int(os.getenv("LLM_TIMEOUT", str(runtime.llm_timeout))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(runtime.llm_max_tokens))),
            temperature=float(
                os.getenv("LLM_TEMPERATURE", str(runtime.llm_temperature))
            ),
            retries=int(os.getenv("LLM_RETRIES", "3")),
            backoff_base=float(os.getenv("LLM_BACKOFF_BASE", "0.5")),
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        # Keep a long-lived AsyncClient for connection reuse
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

    async def chat(self, system: str, user: str) -> str:
        """Call the LLM chat/completions endpoint with retries/backoff.

        Returns: the text content on success, or a sentinel error string
        beginning with "[LLM unavailable:" on failure.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"/no_think\n\n{user}",
                },
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        last_exc = None
        for attempt in range(1, max(1, int(self.config.retries)) + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.post("/chat/completions", json=payload),
                    timeout=float(self.config.timeout),
                )
                resp.raise_for_status()
                data = resp.json()
                # support multiple response shapes; be defensive
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, TypeError, ValueError):
                    # fallback: try common alternative
                    return str(data.get("choices", [data])[0])
            except (
                asyncio.TimeoutError,
                httpx.HTTPError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt,
                    self.config.retries,
                    exc,
                )
                if attempt < self.config.retries:
                    backoff = float(self.config.backoff_base) * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

        logger.error(
            "LLM call failed after %d attempts: %s", int(self.config.retries), last_exc
        )
        return f"[LLM unavailable: {last_exc}]"

    async def check_health(self) -> dict:
        try:
            resp = await self._client.get("/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            return {"ok": True, "models": [m.get("id", "") for m in models]}
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
