from __future__ import annotations

import os
from typing import Callable, Optional

from core.models.memory_graph import EmotionLabel, MemoryGraph

LLMCallable = Callable[[str], str]


def _load_openai_client(
    base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY"
):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "openai package is not installed; cannot use OpenAI/LM Studio backend"
        ) from exc

    api_key = os.getenv(api_key_env, "not-set")
    client = (
        OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else OpenAI(api_key=api_key)
    )
    return client


def _load_ollama_client():
    try:
        import ollama  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ollama package is not installed; cannot use Ollama backend"
        ) from exc

    return ollama


def create_llm_callable(
    provider: Optional[str], model: Optional[str]
) -> Optional[LLMCallable]:
    """Create an LLMCallable based on provider/model configuration.

    Supported providers:
      - "openai": Hosted OpenAI-compatible APIs (gpt-4o, gpt-4.1-mini, etc.).
      - "lmstudio": Local LM Studio, using its OpenAI-compatible HTTP API.
      - "ollama": Local Ollama HTTP API.
    """

    if provider is None or not model:
        return None

    provider = provider.lower()

    if provider == "openai":
        client = _load_openai_client()

        def _call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        return _call

    if provider == "lmstudio":
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        client = _load_openai_client(base_url=base_url, api_key_env="LMSTUDIO_API_KEY")

        def _call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        return _call

    if provider == "ollama":
        ollama = _load_ollama_client()

        def _call(prompt: str) -> str:
            resp = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            return resp.get("message", {}).get("content", "")

        return _call

    return None


def populate_memory_from_journal(graph: MemoryGraph) -> None:
    """Populate MemoryGraph from on-disk journal entries if available.

    This uses a simple JSONL store written by the API or dashboard when the
    user encodes daytime experiences. Each entry is mapped to an episodic
    fragment with appropriate emotional tagging.
    """

    from .journal_store import load_journal_entries  # local import to avoid cycles

    entries = load_journal_entries()
    for entry in entries:
        try:
            emotion = EmotionLabel(entry["emotion"])
        except Exception:
            emotion = EmotionLabel.NEUTRAL
        tags = entry.get("tags", [])
        graph.encode_from_user_input(
            text=entry["text"], emotion=emotion, tags=tags, is_episodic=True
        )
