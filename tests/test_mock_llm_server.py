from __future__ import annotations

from fastapi.testclient import TestClient

from tests.mock_llm_server import MockLLMServer


def test_mock_server_responds_correctly() -> None:
    server = MockLLMServer()
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_object"},
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    assert len(server.received_requests) == 1
