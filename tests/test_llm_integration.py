from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.agents.dream_constructor_agent import DreamConstructorAgent, GenerationMode
from core.llm_client import LLMClient, LLMConfig
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepStage, SleepState
from tests.mock_llm_server import MockLLMServer

pytestmark = pytest.mark.integration


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class _FakeCompletions:
    def __init__(self, server: MockLLMServer) -> None:
        self._server = server

    def create(self, **kwargs: Any) -> Any:
        extra_headers = kwargs.pop("extra_headers", {}) or {}
        headers = {str(k).lower(): str(v) for k, v in dict(extra_headers).items()}
        payload = dict(kwargs)
        response = self._server.build_response(payload=payload, headers=headers)
        return _to_namespace(response)


class _FakeChat:
    def __init__(self, server: MockLLMServer) -> None:
        self.completions = _FakeCompletions(server)


class _FakeOpenAIClient:
    def __init__(self, server: MockLLMServer) -> None:
        self.chat = _FakeChat(server)


@pytest.fixture
def mock_server() -> MockLLMServer:
    return MockLLMServer()


def test_no_think_header_enforced(mock_server: MockLLMServer) -> None:
    agent = DreamConstructorAgent({"provider": "openai", "model": "mock"})
    agent._client = _FakeOpenAIClient(mock_server)
    payload, error, used, _ = agent._call_llm("system prompt", "user prompt")
    assert used is True
    assert error is None
    assert payload.get("narrative")

    request = mock_server.received_requests[-1]["json"]
    messages = request["messages"]
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith("/no_think")
    assert agent.last_llm_metadata.get("finish_reason") == "stop"


@pytest.mark.asyncio
async def test_json_mode_request_format(
    monkeypatch: pytest.MonkeyPatch, mock_server: MockLLMServer
) -> None:
    class _LocalAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.ASGITransport(app=mock_server.app)
            super().__init__(*args, **kwargs)

    import core.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", _LocalAsyncClient)
    client = LLMClient(
        LLMConfig(
            provider="openai",
            base_url="http://mock/v1",
            model="mock-model",
            api_key="mock",
            json_mode=True,
            retries=1,
        )
    )
    content = await client.chat(system="system", user="user")
    request = mock_server.received_requests[-1]["json"]
    assert request.get("response_format") == {"type": "json_object"}
    parsed = json.loads(content)
    assert "narrative" in parsed


def test_retry_on_malformed_json(mock_server: MockLLMServer) -> None:
    agent = DreamConstructorAgent(
        {"provider": "openai", "model": "mock-model", "mock_mode": "truncate"}
    )
    agent._client = _FakeOpenAIClient(mock_server)

    segment = agent.generate_segment(
        segment_index=0,
        sleep_state=SleepState(
            time_hours=0.5,
            process_s=0.8,
            process_c=0.0,
            stage=SleepStage.REM,
            cycle_index=0,
        ),
        neuro_state=NeurochemistryState(
            time_hours=0.5,
            ach=0.9,
            serotonin=0.1,
            ne=0.1,
            cortisol=0.3,
        ),
        replay=None,
        stress_level=0.2,
        prior_events=[],
        prev_segments=[],
    )

    assert segment.generation_mode == GenerationMode.LLM_FALLBACK
    assert len(segment.narrative.strip()) > 20
    assert len(mock_server.received_requests) == 2


@pytest.mark.asyncio
async def test_reasoning_tokens_zero(
    monkeypatch: pytest.MonkeyPatch, mock_server: MockLLMServer
) -> None:
    class _LocalAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.ASGITransport(app=mock_server.app)
            super().__init__(*args, **kwargs)

    import core.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", _LocalAsyncClient)
    client = LLMClient(
        LLMConfig(
            provider="openai",
            base_url="http://mock/v1",
            model="mock-model",
            api_key="mock",
            json_mode=True,
            retries=1,
        )
    )
    await client.chat(system="system", user="user")
    usage = client.last_response_meta.get("usage", {})
    details = usage.get("completion_tokens_details", {})
    assert details.get("reasoning_tokens") == 0
