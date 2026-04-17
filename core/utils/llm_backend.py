"""Unified LLM backend with provider auto-detection and demo-mode DreamScript.

Provides a lightweight abstraction for generating text from OpenAI/Anthropic/
Ollama or the offline DreamScript engine when no provider is available.
"""

from __future__ import annotations

import os
import time
import re
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import httpx
from pydantic import BaseModel, Field

from core.config import load_runtime_config
from core.simulation.dreamscript import DreamScriptEngine
from core.utils.llm_adapters import create_llm_callable

_RUNTIME_CONFIG = load_runtime_config()


class Providers(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    DREAMSCRIPT = "dreamscript"


class LLMConfig(BaseModel):
    provider: Providers = Providers.DREAMSCRIPT
    model_name: str = Field(default_factory=lambda: _RUNTIME_CONFIG.llm_model)
    api_key: Optional[str] = Field(default_factory=lambda: _RUNTIME_CONFIG.llm_api_key)
    ollama_base_url: str = Field(
        default_factory=lambda: _RUNTIME_CONFIG.llm_ollama_base_url
    )
    lmstudio_base_url: str = Field(
        default_factory=lambda: _RUNTIME_CONFIG.llm_lmstudio_base_url
    )
    temperature: float = Field(default_factory=lambda: _RUNTIME_CONFIG.llm_temperature)
    max_tokens: int = Field(default_factory=lambda: _RUNTIME_CONFIG.llm_max_tokens)
    demo_mode: bool = Field(
        default_factory=lambda: os.getenv("DEMO_MODE", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def strip_thinking_tags(response: str) -> str:
    if not response:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


class LLMBackend:
    """Auto-detecting LLM backend with DreamScript offline fallback.

    Detection order:
      1. OPENAI_API_KEY environment variable
      2. ANTHROPIC_API_KEY environment variable
      3. Ollama local HTTP service at http://localhost:11434/api/tags
      4. Fallback to DREAMSCRIPT (offline)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._callable: Optional[Callable[[str], str]] = None
        self._dreamscript = DreamScriptEngine()
        self._detect_provider()

    def _detect_provider(self) -> None:
        if self.config.demo_mode and not (
            os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        ):
            self.config.provider = Providers.DREAMSCRIPT
            self._callable = None
            return

        # 1) Environment keys
        if os.getenv("OPENAI_API_KEY"):
            self.config.provider = Providers.OPENAI
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.config.provider = Providers.ANTHROPIC
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # 2) Check Ollama HTTP endpoint
            try:
                with httpx.Client(timeout=2.0) as client:
                    r = client.get(f"{self.config.ollama_base_url}/api/tags")
                    if r.status_code == 200:
                        self.config.provider = Providers.OLLAMA
                    else:
                        m = client.get(f"{self.config.lmstudio_base_url}/models")
                        if m.status_code == 200:
                            self.config.provider = Providers.LMSTUDIO
            except Exception:
                try:
                    with httpx.Client(timeout=2.0) as client:
                        m = client.get(f"{self.config.lmstudio_base_url}/models")
                        if m.status_code == 200:
                            self.config.provider = Providers.LMSTUDIO
                        else:
                            self.config.provider = Providers.DREAMSCRIPT
                except Exception:
                    # default to DREAMSCRIPT
                    self.config.provider = Providers.DREAMSCRIPT

        # Instantiate callable if not DreamsScript
        if self.config.provider != Providers.DREAMSCRIPT:
            # try to create an adapter callable (may raise if libs missing)
            try:
                self._callable = create_llm_callable(
                    self.config.provider, self.config.model_name
                )
            except Exception:
                self._callable = None
                self.config.provider = Providers.DREAMSCRIPT

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Synchronous generate with retry/backoff. Returns full string.

        For DREAMSCRIPT provider the prompt is not used; instead callers should
        call the offline engine directly via `generate_offline`.
        """
        if self.config.provider == Providers.DREAMSCRIPT or self._callable is None:
            return self.generate_offline(prompt)

        backoff = 0.5
        for attempt in range(max_retries):
            try:
                # callable is expected to be synchronous
                return strip_thinking_tags(self._callable(prompt))
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2.0
        return ""

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield string chunks for streaming consumers.

        When a provider does not support streaming, yield the full result once.
        """
        if self.config.provider == Providers.DREAMSCRIPT or self._callable is None:
            yield self.generate_offline(prompt)
            return

        # Not all adapters support streaming; fall back to single-shot
        out = self.generate(prompt)
        yield out

    def generate_offline(self, prompt: str) -> str:
        """Simple bridge to DreamScriptEngine for demo narratives.

        The prompt may contain a lightweight encoded context; if the string
        'DREAMSCRIPT:' is present we attempt to parse stage/neuro/arousal info.
        """
        # best-effort parse of prompt for a few fields
        stage = None
        neuro = type("_N", (), {"ach": 0.5, "ne": 0.5, "cortisol": 0.5})()
        active_memories: list[Any] = []
        biz = type("_B", (), {"total_score": 0.5})()

        # If the prompt encodes a simple CSV-like context, try to extract
        # stage=REM;ach=0.9;ne=0.05;arousal=0.7;cycle=2
        try:
            parts = [p.strip() for p in prompt.split(";") if "=" in p]
            for part in parts:
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "stage":
                    try:
                        stage = getattr(
                            __import__(
                                "core.models.sleep_cycle", fromlist=["SleepStage"]
                            ),
                            "SleepStage",
                        )(v)
                    except Exception:
                        stage = None
                elif k == "ach":
                    neuro.ach = float(v)
                elif k == "ne":
                    neuro.ne = float(v)
                elif k == "cortisol":
                    neuro.cortisol = float(v)
                elif k == "arousal":
                    biz.total_score = float(v)
        except Exception:
            pass

        if stage is None:
            # default to REM-like or inferred from biz
            stage = (
                getattr(
                    __import__("core.models.sleep_cycle", fromlist=["SleepStage"]),
                    "SleepStage",
                ).REM
                if getattr(biz, "total_score", 0.5) > 0.6
                else getattr(
                    __import__("core.models.sleep_cycle", fromlist=["SleepStage"]),
                    "SleepStage",
                ).N2
            )

        return self._dreamscript.generate_narrative(
            stage=stage,
            neurochemistry=neuro,
            active_memories=active_memories,
            bizarreness=biz,
            prev_segment_text=None,
        )
