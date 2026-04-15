import os

import pytest

import core.config as runtime_config
import core.llm_client as llm_client


def test_env_helpers_and_runtime_config(monkeypatch):
    assert runtime_config._env_float("DOES_NOT_EXIST", 1.5) == 1.5
    assert runtime_config._env_int("DOES_NOT_EXIST_INT", 10) == 10

    monkeypatch.setenv("BAD_FLOAT", "abc")
    monkeypatch.setenv("BAD_INT", "def")
    assert runtime_config._env_float("BAD_FLOAT", 3.2) == 3.2
    assert runtime_config._env_int("BAD_INT", 7) == 7

    monkeypatch.setenv("API_BASE_URL", "http://localhost:9999")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TIMEOUT", "42")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "27.5")
    monkeypatch.setenv("SIM_DURATION_HOURS", "6.5")
    monkeypatch.setenv("SIM_REQUEST_TIMEOUT_SECONDS", "900")
    cfg = runtime_config.load_runtime_config()
    assert cfg.api_base_url == "http://localhost:9999"
    assert cfg.llm_model == "custom-model"
    assert cfg.llm_timeout == 42
    assert cfg.llm_timeout_seconds == 27.5
    assert cfg.simulation_duration_hours == 6.5
    assert cfg.simulation_request_timeout_seconds == 900


def test_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.local")
    monkeypatch.setenv("LLM_MODEL", "m-1")
    monkeypatch.setenv("LLM_API_KEY", "k-1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "321")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.6")
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.1")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "33.0")

    cfg = llm_client.LLMConfig.from_env()
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://llm.local"
    assert cfg.model == "m-1"
    assert cfg.api_key == "k-1"
    assert cfg.max_tokens == 321
    assert cfg.temperature == 0.6
    assert cfg.timeout_seconds == 33.0
    assert cfg.retries == 2
    assert cfg.backoff_base == 0.1


class _FakeResponse:
    def __init__(self, payload=None, error=None):
        self._payload = payload or {}
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_llm_client_chat_and_health_success(monkeypatch):
    class SuccessAsyncClient:
        def __init__(self, *args, **kwargs):
            self.closed = False

        async def post(self, path, json):
            assert path == "/chat/completions"
            assert "messages" in json
            return _FakeResponse(
                payload={"choices": [{"message": {"content": "hello from model"}}]}
            )

        async def get(self, path, timeout=5):
            assert path == "/models"
            return _FakeResponse(payload={"data": [{"id": "model-a"}]})

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", SuccessAsyncClient)
    client = llm_client.LLMClient(
        llm_client.LLMConfig(
            provider="openai",
            base_url="http://fake",
            model="x",
            api_key="k",
            retries=1,
        )
    )

    out = await client.chat(system="s", user="u")
    assert out == "hello from model"

    health = await client.check_health()
    assert health["ok"] is True
    assert health["models"] == ["model-a"]

    await client.aclose()


@pytest.mark.asyncio
async def test_llm_client_chat_failure_and_health_failure(monkeypatch):
    class FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def post(self, path, json):
            raise RuntimeError("network down")

        async def get(self, path, timeout=5):
            raise RuntimeError("health down")

        async def aclose(self):
            return None

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", FailingAsyncClient)
    monkeypatch.setattr(llm_client.asyncio, "sleep", _no_sleep)

    client = llm_client.LLMClient(
        llm_client.LLMConfig(
            provider="openai",
            base_url="http://fake",
            model="x",
            api_key="k",
            retries=2,
            backoff_base=0.0,
        )
    )
    out = await client.chat(system="s", user="u")
    assert out.startswith("[LLM unavailable:")

    health = await client.check_health()
    assert health["ok"] is False
    assert "health down" in health["error"]


@pytest.mark.asyncio
async def test_llm_client_retries_with_relaxed_payload_on_400(monkeypatch):
    calls: list[dict[str, object]] = []

    class VariantAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def post(self, path, json):
            calls.append(json)
            if len(calls) == 1:
                req = llm_client.httpx.Request("POST", "http://fake/chat/completions")
                err = llm_client.httpx.HTTPStatusError(
                    "400 Bad Request",
                    request=req,
                    response=llm_client.httpx.Response(400, request=req),
                )
                return _FakeResponse(error=err)
            return _FakeResponse(
                payload={"choices": [{"message": {"content": "ok after relax"}}]}
            )

        async def get(self, path, timeout=5):
            return _FakeResponse(payload={"data": [{"id": "model-a"}]})

        async def aclose(self):
            return None

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", VariantAsyncClient)
    client = llm_client.LLMClient(
        llm_client.LLMConfig(
            provider="openai",
            base_url="http://fake",
            model="x",
            api_key="k",
            retries=3,
            backoff_base=0.0,
            json_mode=True,
            no_think=True,
        )
    )
    out = await client.chat(system="sys", user="usr")
    assert out == "ok after relax"
    assert len(calls) >= 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_get_llm_client_singleton_reset(monkeypatch):
    llm_client._default_client = None

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def post(self, path, json):
            return _FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})

        async def get(self, path, timeout=5):
            return _FakeResponse(payload={"data": []})

        async def aclose(self):
            return None

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://singleton")
    monkeypatch.setenv("LLM_MODEL", "singleton-model")
    monkeypatch.setenv("LLM_API_KEY", "singleton-key")

    c1 = llm_client.get_llm_client()
    c2 = llm_client.get_llm_client()
    assert c1 is c2
    assert c1.config.base_url == os.getenv("LLM_BASE_URL")
