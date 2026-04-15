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
from typing import Any, Optional
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
    no_think: bool = True
    json_mode: bool = True
    timeout_seconds: float = 15.0
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
            no_think=os.getenv("LLM_NO_THINK", "true").lower() == "true",
            json_mode=os.getenv("LLM_JSON_MODE", "true").lower() == "true",
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0")),
            retries=int(os.getenv("LLM_RETRIES", "3")),
            backoff_base=float(os.getenv("LLM_BACKOFF_BASE", "0.5")),
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.last_response_meta: dict[str, Any] = {}
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
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"/no_think\n\n{system}" if self.config.no_think else system
                    ),
                },
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_exc = None
        for attempt in range(1, max(1, int(self.config.retries)) + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.post("/chat/completions", json=payload),
                    timeout=float(self.config.timeout_seconds),
                )
                resp.raise_for_status()
                data_obj = resp.json()
                if not isinstance(data_obj, dict):
                    self.last_response_meta = {}
                    return str(data_obj)
                self.last_response_meta = {}
                usage = data_obj.get("usage")
                if isinstance(usage, dict):
                    self.last_response_meta["usage"] = usage
                # support multiple response shapes; be defensive
                try:
                    choices = data_obj.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        return ""
                    first = choices[0]
                    if not isinstance(first, dict):
                        return str(first)
                    self.last_response_meta["finish_reason"] = first.get(
                        "finish_reason"
                    )
                    message = first.get("message", {})
                    if not isinstance(message, dict):
                        return str(message)
                    content = message.get("content", "")
                    return str(content).strip()
                except (KeyError, IndexError, TypeError, ValueError):
                    # fallback: try common alternative
                    choices = data_obj.get("choices", [data_obj])
                    if isinstance(choices, list) and choices:
                        return str(choices[0])
                    return str(data_obj)
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

    async def check_health(self) -> dict[str, object]:
        try:
            resp = await self._client.get("/models", timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                return {"ok": False, "error": "Invalid /models response payload"}
            models = payload.get("data", [])
            if not isinstance(models, list):
                return {"ok": False, "error": "Invalid /models data field"}
            model_ids = [
                str(m.get("id", ""))
                for m in models
                if isinstance(m, dict) and "id" in m
            ]
            return {"ok": True, "models": model_ids}
        except (httpx.HTTPError, ValueError, TypeError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    async def aclose(self) -> None:
        await self._client.aclose()


_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
