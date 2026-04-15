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
            timeout_seconds=float(
                os.getenv("LLM_TIMEOUT_SECONDS", str(runtime.llm_timeout_seconds))
            ),
            retries=int(os.getenv("LLM_RETRIES", "3")),
            backoff_base=float(os.getenv("LLM_BACKOFF_BASE", "0.5")),
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.last_response_meta: dict[str, Any] = {}
        self._preferred_no_think = self.config.no_think
        self._preferred_json_mode = self.config.json_mode
        self._compat_lock = asyncio.Lock()
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
        payload_variants = self._build_payload_variants(system=system, user=user)
        total_attempts = max(1, int(self.config.retries), len(payload_variants))

        last_exc = None
        saw_payload_rejection = False
        for attempt in range(1, total_attempts + 1):
            variant_idx = min(attempt - 1, len(payload_variants) - 1)
            variant = payload_variants[variant_idx]
            payload = variant["payload"]
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
                if saw_payload_rejection and variant_idx > 0:
                    await self._remember_compatible_variant(
                        include_no_think=bool(variant["include_no_think"]),
                        include_json_mode=bool(variant["include_json_mode"]),
                    )
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
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else 0
                )
                # Compatibility path: some providers reject response_format/no_think
                # combinations with 400. Progressively relax payload controls.
                if status_code == 400 and variant_idx < len(payload_variants) - 1:
                    saw_payload_rejection = True
                    logger.warning(
                        "LLM payload rejected (400) on variant %d/%d; retrying with relaxed payload",
                        variant_idx + 1,
                        len(payload_variants),
                    )
                    continue
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt,
                    total_attempts,
                    exc,
                )
                if attempt < total_attempts:
                    backoff = float(self.config.backoff_base) * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
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
                    total_attempts,
                    exc,
                )
                if attempt < total_attempts:
                    backoff = float(self.config.backoff_base) * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

        logger.error(
            "LLM call failed after %d attempts: %s", int(total_attempts), last_exc
        )
        return f"[LLM unavailable: {last_exc}]"

    def _build_payload(
        self,
        *,
        system: str,
        user: str,
        include_no_think: bool,
        include_json_mode: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"/no_think\n\n{system}"
                        if include_no_think and self.config.no_think
                        else system
                    ),
                },
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if include_json_mode and self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _build_payload_variants(
        self, *, system: str, user: str
    ) -> list[dict[str, object]]:
        variants: list[tuple[bool, bool]] = [
            (self._preferred_no_think, self._preferred_json_mode),  # preferred payload
            (self._preferred_no_think, False),  # relax response_format
            (False, self._preferred_json_mode),  # relax /no_think
            (False, False),  # relax both
        ]
        payloads: list[dict[str, object]] = []
        seen: set[tuple[bool, bool]] = set()
        for include_no_think, include_json_mode in variants:
            signature = (include_no_think, include_json_mode)
            if signature in seen:
                continue
            seen.add(signature)
            payload = self._build_payload(
                system=system,
                user=user,
                include_no_think=include_no_think,
                include_json_mode=include_json_mode,
            )
            payloads.append(
                {
                    "include_no_think": include_no_think,
                    "include_json_mode": include_json_mode,
                    "payload": payload,
                }
            )
        return payloads

    async def _remember_compatible_variant(
        self, *, include_no_think: bool, include_json_mode: bool
    ) -> None:
        async with self._compat_lock:
            if (
                self._preferred_no_think == include_no_think
                and self._preferred_json_mode == include_json_mode
            ):
                return
            self._preferred_no_think = include_no_think
            self._preferred_json_mode = include_json_mode
            logger.info(
                "LLM compatibility learned: no_think=%s, json_mode=%s",
                include_no_think,
                include_json_mode,
            )

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
