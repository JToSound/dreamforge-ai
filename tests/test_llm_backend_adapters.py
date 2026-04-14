from __future__ import annotations

import types

from core.models.memory_graph import MemoryGraph
from core.utils import llm_adapters
from core.utils.llm_backend import LLMBackend, LLMConfig, Providers, strip_thinking_tags


def test_create_llm_callable_variants(monkeypatch):
    class _Resp:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

    class _OpenAIClient:
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: _Resp())
        )

    class _OllamaClient:
        @staticmethod
        def chat(**kwargs):
            return {"message": {"content": "ollama-ok"}}

    monkeypatch.setattr(
        llm_adapters, "_load_openai_client", lambda **kwargs: _OpenAIClient()
    )
    monkeypatch.setattr(llm_adapters, "_load_ollama_client", lambda: _OllamaClient())

    assert llm_adapters.create_llm_callable(None, "m") is None
    assert llm_adapters.create_llm_callable("openai", "") is None

    openai_call = llm_adapters.create_llm_callable("openai", "gpt")
    assert openai_call and openai_call("hello") == "ok"

    lmstudio_call = llm_adapters.create_llm_callable("lmstudio", "local-model")
    assert lmstudio_call and lmstudio_call("hello") == "ok"

    ollama_call = llm_adapters.create_llm_callable("ollama", "llama")
    assert ollama_call and ollama_call("hello") == "ollama-ok"

    assert llm_adapters.create_llm_callable("unknown", "m") is None


def test_populate_memory_from_journal(monkeypatch):
    monkeypatch.setattr(
        "core.utils.journal_store.load_journal_entries",
        lambda: [
            {"text": "meeting with team", "emotion": "joy", "tags": ["work"]},
            {"text": "late train", "emotion": "not-a-real-emotion", "tags": []},
        ],
    )
    graph = MemoryGraph()
    llm_adapters.populate_memory_from_journal(graph)
    assert graph.to_networkx().number_of_nodes() == 2


def test_strip_thinking_tags():
    assert strip_thinking_tags("<think>hidden</think>visible") == "visible"
    assert strip_thinking_tags("") == ""


def test_llm_backend_provider_detection_and_generate(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _Client:
        def __init__(self, timeout=2.0):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            if url.endswith("/api/tags"):
                return _Resp(200)
            return _Resp(404)

    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("core.utils.llm_backend.httpx.Client", _Client)
    monkeypatch.setattr(
        "core.utils.llm_backend.create_llm_callable",
        lambda provider, model: (lambda prompt: "<think>x</think>live"),
    )

    backend = LLMBackend(LLMConfig(demo_mode=False, model_name="m"))
    assert backend.config.provider == Providers.OLLAMA
    assert backend.generate("prompt") == "live"

    # stream falls back to single full chunk when adapter is not streaming-aware
    chunks = list(backend.generate_stream("prompt"))
    assert chunks == ["live"]


def test_llm_backend_lmstudio_and_callable_failure(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _Client:
        def __init__(self, timeout=2.0):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            if url.endswith("/api/tags"):
                return _Resp(404)
            if url.endswith("/models"):
                return _Resp(200)
            return _Resp(404)

    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("core.utils.llm_backend.httpx.Client", _Client)
    monkeypatch.setattr(
        "core.utils.llm_backend.create_llm_callable",
        lambda provider, model: (_ for _ in ()).throw(RuntimeError("adapter failed")),
    )

    backend = LLMBackend(LLMConfig(demo_mode=False, model_name="m"))
    assert backend.config.provider == Providers.DREAMSCRIPT


def test_llm_backend_generate_retry_and_offline_parse(monkeypatch):
    attempts = {"count": 0}

    def _callable(prompt: str) -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("temporary")
        return "<think>debug</think>final"

    backend = LLMBackend(LLMConfig(provider=Providers.OPENAI, demo_mode=False))
    backend.config.provider = Providers.OPENAI
    backend._callable = _callable
    monkeypatch.setattr("core.utils.llm_backend.time.sleep", lambda *_: None)
    assert backend.generate("x", max_retries=3) == "final"
    assert attempts["count"] == 2

    class _FakeDream:
        def __init__(self):
            self.called_with = None

        def generate_narrative(self, **kwargs):
            self.called_with = kwargs
            return "offline-ok"

    fake = _FakeDream()
    backend.config.provider = Providers.DREAMSCRIPT
    backend._dreamscript = fake
    out = backend.generate_offline("stage=REM;ach=0.9;ne=0.1;cortisol=0.7;arousal=0.8")
    assert out == "offline-ok"
    assert str(fake.called_with["stage"]) == "SleepStage.REM"
