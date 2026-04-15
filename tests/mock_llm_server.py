from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class MockLLMServer:
    """OpenAI-compatible in-memory mock server for integration tests."""

    received_requests: list[dict[str, Any]] = field(default_factory=list)

    def build_response(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        self.received_requests.append({"headers": headers, "json": payload})
        mock_mode = headers.get("x-mock-mode", "").strip().lower()

        if mock_mode == "truncate":
            content = '{"narrative":"truncated output'
            finish_reason = "length"
            reasoning_tokens = 511
        else:
            response_format = payload.get("response_format", {})
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            ):
                content = json.dumps(
                    {
                        "narrative": "I walk through an endless corridor while clocks melt into mirrors and the air hums with quiet dread.",
                        "scene": "endless corridor",
                    }
                )
            else:
                content = "non-json content"
            finish_reason = "stop"
            reasoning_tokens = 0

        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 64,
                "completion_tokens": 128,
                "total_tokens": 192,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }

    def __post_init__(self) -> None:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        @app.post("/chat/completions")
        async def chat_completions(request: Request) -> JSONResponse:
            payload = await request.json()
            headers = {k.lower(): v for k, v in request.headers.items()}
            response = self.build_response(payload=payload, headers=headers)
            return JSONResponse(response)

        @app.get("/models")
        @app.get("/v1/models")
        async def models() -> JSONResponse:
            return JSONResponse({"data": [{"id": "mock-model"}]})

        self.app = app
