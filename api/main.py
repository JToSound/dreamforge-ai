"""
api/main.py
───────────
DreamForge AI — FastAPI application entry point.

Endpoints
─────────
GET  /                        Health ping
GET  /api/health/llm          LLM connectivity check
GET  /api/llm/config          Read current LLM config
POST /api/llm/config          Update LLM config at runtime
POST /api/simulation/night    Run a full-night dream simulation
POST /api/simulation/multi-night  Run a multi-night continuity batch
GET  /api/simulation/{id}     Retrieve a stored simulation
POST /api/simulation/counterfactual  Run a counterfactual variant
GET  /api/dreams              List all stored simulation summaries

Docs
────
Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from core.data.template_loader import SchemaValidationError, TemplateBank
from core.models.sleep_cycle import SleepCycleModel, SleepStage
from core.models.neurochemistry import cortisol_profile
from core.generation.narrative_generator import (
    NarrativeGenerator,
    NarrativeGeneratorConfig,
)
from core.llm_registry import get_llm_registry_snapshot
from core.quality.narrative_quality import summarize_narrative_quality
from core.simulation.lucidity_model import LucidityModel, LucidityTickState
from core.simulation.llm_client import parse_narrative_response
from core.simulation.exporters import (
    export_memory_activations_csv,
    export_neurochemistry_csv,
)
from core.simulation.narrative_cache import NarrativeCache

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dreamforge.api")

API_CONTRACT_VERSION = "v1"
PROMPT_PROFILE_VERSION = "narrative-v1.0"
SLO_TARGETS: Dict[str, float] = {
    "api_success_rate_min": 0.995,
    "simulation_completion_rate_min": 0.99,
    "simulation_p95_latency_seconds_max": 30.0,
    "export_success_rate_min": 0.995,
}
ERROR_TAXONOMY: Dict[str, str] = {
    "validation_error": "Request payload or parameters are invalid.",
    "not_found": "Requested resource does not exist.",
    "unauthorized": "Authentication failed or missing credentials.",
    "rate_limited": "Too many requests in the configured time window.",
    "provider_error": "Upstream model/provider rejected or failed the request.",
    "timeout": "Request or model call exceeded timeout threshold.",
    "internal_error": "Unhandled server-side failure.",
}
_API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()
_REDIS_URL = os.getenv("REDIS_URL", "").strip()
_METRICS_PUBLIC = os.getenv("METRICS_PUBLIC", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_OUTPUT_RETENTION_DAYS = max(0, int(os.getenv("OUTPUT_RETENTION_DAYS", "14")))
_OUTPUT_RETENTION_MAX_RUNS = max(1, int(os.getenv("OUTPUT_RETENTION_MAX_RUNS", "200")))
_ASYNC_MAX_PENDING_JOBS = max(1, int(os.getenv("ASYNC_MAX_PENDING_JOBS", "128")))
_ASYNC_MAX_RUNNING_JOBS = max(1, int(os.getenv("ASYNC_MAX_RUNNING_JOBS", "8")))
_STATE_EVENT_LOG_FILE = os.getenv(
    "DREAMFORGE_STATE_EVENT_LOG", "state-events.jsonl"
).strip()
if not _STATE_EVENT_LOG_FILE:
    _STATE_EVENT_LOG_FILE = "state-events.jsonl"
_ARTIFACT_MANIFEST_PATH = os.getenv(
    "DREAMFORGE_ARTIFACT_MANIFEST", "artifacts/manifest.json"
).strip()
if not _ARTIFACT_MANIFEST_PATH:
    _ARTIFACT_MANIFEST_PATH = "artifacts/manifest.json"


def _load_token_role_map() -> Dict[str, Dict[str, Any]]:
    raw = os.getenv("API_TOKEN_ROLE_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid API_TOKEN_ROLE_MAP JSON: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("API_TOKEN_ROLE_MAP must be a JSON object mapping token->policy")
        return {}

    role_map: Dict[str, Dict[str, Any]] = {}
    for token, policy in parsed.items():
        token_str = str(token).strip()
        if not token_str:
            continue
        if not isinstance(policy, dict):
            continue
        role = str(policy.get("role", "viewer")).strip().lower() or "viewer"
        scopes_raw = policy.get("scopes", [])
        if isinstance(scopes_raw, list):
            scopes = [str(scope).strip() for scope in scopes_raw if str(scope).strip()]
        elif isinstance(scopes_raw, str) and scopes_raw.strip():
            scopes = [scopes_raw.strip()]
        else:
            scopes = []
        role_map[token_str] = {"role": role, "scopes": scopes}
    return role_map


_API_TOKEN_ROLE_MAP = _load_token_role_map()
try:
    _API_RATE_LIMIT_PER_MINUTE = max(
        10, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "240"))
    )
except ValueError:
    _API_RATE_LIMIT_PER_MINUTE = 240
_RATE_LIMIT_WINDOW_SECONDS = 60.0

# ── Import local modules (with graceful fallback if running standalone) ────────
try:
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from core.llm_client import LLMClient, LLMConfig, get_llm_client
except ImportError:
    # Minimal inline fallback so the server still starts for /docs inspection
    logger.warning("core.llm_client not found — using inline stub")

    from dataclasses import dataclass

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

        @classmethod
        def from_env(cls):
            return cls(
                provider=os.getenv("LLM_PROVIDER", "lmstudio"),
                base_url=os.getenv(
                    "LLM_BASE_URL", "http://host.docker.internal:1234/v1"
                ),
                model=os.getenv("LLM_MODEL", "qwen/qwen3.5-9b"),
                api_key=os.getenv("LLM_API_KEY", "lm-studio"),
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
                # Source: Qwen3.5 docs (reasoning-token budget requires >=2048 output tokens)
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.85")),
            )

    class LLMClient:
        def __init__(self, config=None):
            self.config = config or LLMConfig.from_env()

        async def chat(self, system: str, user: str) -> str:
            import httpx

            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            try:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.config.timeout,
                ) as client:
                    resp = await client.post("/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                return f"[LLM unavailable: {exc}]"

        async def check_health(self) -> dict:
            import httpx

            try:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    timeout=5,
                ) as client:
                    resp = await client.get("/models")
                    resp.raise_for_status()
                    models = resp.json().get("data", [])
                    return {"ok": True, "models": [m.get("id", "") for m in models]}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        async def aclose(self):
            pass

    _default_client: Optional[LLMClient] = None

    def get_llm_client() -> LLMClient:
        global _default_client
        if _default_client is None:
            _default_client = LLMClient()
        return _default_client


# ── In-memory store ───────────────────────────────────────────────────────────
_simulations: Dict[str, dict] = {}
_template_bank: TemplateBank | None = None
_template_bank_initialized = False
_metrics_lock = threading.Lock()
_runtime_metrics: Dict[str, float] = {
    "simulation_requests_total": 0.0,
    "simulation_completed_total": 0.0,
    "simulation_failed_total": 0.0,
    "simulation_duration_seconds_total": 0.0,
    "llm_invocations_total": 0.0,
    "llm_fallback_segments_total": 0.0,
    "api_requests_total": 0.0,
    "api_errors_total": 0.0,
    "api_rate_limited_total": 0.0,
    "api_unauthorized_total": 0.0,
    "export_failures_total": 0.0,
}
_simulation_duration_seconds_history: deque[float] = deque(maxlen=2048)
_simulation_duration_seconds_llm_history: deque[float] = deque(maxlen=1024)
_simulation_duration_seconds_no_llm_history: deque[float] = deque(maxlen=1024)
_llm_invocation_latency_seconds_history: deque[float] = deque(maxlen=4096)
_llm_invocation_density_history: deque[float] = deque(maxlen=1024)
_api_error_codes: Dict[str, float] = defaultdict(float)
_request_windows: Dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()
_jobs_lock = threading.Lock()
_simulation_jobs: Dict[str, Dict[str, Any]] = {}
_simulation_job_tasks: Dict[str, asyncio.Task[Any]] = {}
_workspaces_lock = threading.Lock()
_workspaces: Dict[str, Dict[str, Any]] = {}
_state_log_lock = threading.Lock()
_artifact_health_lock = threading.Lock()
_artifact_health_cache: Dict[str, Any] = {}
_artifact_health_cached_at = 0.0
_simulation_progress_callback_var: contextvars.ContextVar[
    Optional[Callable[[Dict[str, Any]], None]]
] = contextvars.ContextVar("simulation_progress_callback", default=None)
_audit_events: deque[Dict[str, Any]] = deque(maxlen=2000)
_redis_lock = threading.Lock()
_redis_client: Any = None
_redis_disabled = False

_REDIS_KEY_SIMULATION_PREFIX = "dreamforge:simulation:"
_REDIS_KEY_SIMULATION_INDEX = "dreamforge:simulations"
_REDIS_KEY_JOB_PREFIX = "dreamforge:job:"
_REDIS_KEY_AUDIT_EVENTS = "dreamforge:audit:events"
_REDIS_KEY_WORKSPACE_PREFIX = "dreamforge:workspace:"
_REDIS_KEY_WORKSPACE_INDEX = "dreamforge:workspaces"


def _get_redis_client():
    global _redis_client, _redis_disabled
    if _redis_disabled or not _REDIS_URL:
        return None
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis

            client = redis.Redis.from_url(
                _REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            client.ping()
            _redis_client = client
            return _redis_client
        except (ImportError, RuntimeError, ValueError) as exc:
            logger.warning("Redis disabled: %s", exc)
            _redis_disabled = True
            return None
        except Exception as exc:
            logger.warning("Redis unavailable at %s: %s", _REDIS_URL, exc)
            _redis_disabled = True
            return None


def _persist_simulation(sim_id: str, payload: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is not None:
        try:
            raw = json.dumps(payload, ensure_ascii=False)
            client.set(f"{_REDIS_KEY_SIMULATION_PREFIX}{sim_id}", raw)
            client.sadd(_REDIS_KEY_SIMULATION_INDEX, sim_id)
        except Exception as exc:
            logger.warning("Failed to persist simulation %s to Redis: %s", sim_id, exc)
    _append_state_snapshot("simulation", sim_id, payload)


def _load_persisted_simulation(sim_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(f"{_REDIS_KEY_SIMULATION_PREFIX}{sim_id}")
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Invalid persisted simulation payload for %s: %s", sim_id, exc
            )
        except Exception as exc:
            logger.warning("Failed loading simulation %s from Redis: %s", sim_id, exc)
    payload = _load_state_snapshot("simulation", sim_id)
    return payload if isinstance(payload, dict) else None


def _persist_job(job_id: str, payload: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is not None:
        try:
            client.set(
                f"{_REDIS_KEY_JOB_PREFIX}{job_id}",
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("Failed to persist job %s to Redis: %s", job_id, exc)
    _append_state_snapshot("job", job_id, payload)


def _persist_workspace(workspace_id: str, payload: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is not None:
        try:
            client.set(
                f"{_REDIS_KEY_WORKSPACE_PREFIX}{workspace_id}",
                json.dumps(payload, ensure_ascii=False),
            )
            client.sadd(_REDIS_KEY_WORKSPACE_INDEX, workspace_id)
        except Exception as exc:
            logger.warning(
                "Failed to persist workspace %s to Redis: %s", workspace_id, exc
            )
    _append_state_snapshot("workspace", workspace_id, payload)


def _load_persisted_job(job_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(f"{_REDIS_KEY_JOB_PREFIX}{job_id}")
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Invalid persisted job payload for %s: %s", job_id, exc)
        except Exception as exc:
            logger.warning("Failed loading job %s from Redis: %s", job_id, exc)
    payload = _load_state_snapshot("job", job_id)
    return payload if isinstance(payload, dict) else None


def _load_persisted_workspace(workspace_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(f"{_REDIS_KEY_WORKSPACE_PREFIX}{workspace_id}")
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Invalid persisted workspace payload for %s: %s", workspace_id, exc
            )
        except Exception as exc:
            logger.warning(
                "Failed loading workspace %s from Redis: %s", workspace_id, exc
            )
    payload = _load_state_snapshot("workspace", workspace_id)
    return payload if isinstance(payload, dict) else None


def _list_persisted_sim_ids() -> List[str]:
    client = _get_redis_client()
    if client is not None:
        try:
            values = client.smembers(_REDIS_KEY_SIMULATION_INDEX) or []
            return [str(item) for item in values if str(item)]
        except Exception as exc:
            logger.warning("Failed listing persisted simulation IDs: %s", exc)
    return _list_state_entity_ids("simulation")


def _list_persisted_workspace_ids() -> List[str]:
    client = _get_redis_client()
    if client is not None:
        try:
            values = client.smembers(_REDIS_KEY_WORKSPACE_INDEX) or []
            return [str(item) for item in values if str(item)]
        except Exception as exc:
            logger.warning("Failed listing persisted workspace IDs: %s", exc)
    return _list_state_entity_ids("workspace")


def _persist_audit_event(entry: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is not None:
        try:
            client.lpush(_REDIS_KEY_AUDIT_EVENTS, json.dumps(entry, ensure_ascii=False))
            client.ltrim(_REDIS_KEY_AUDIT_EVENTS, 0, 1999)
        except Exception as exc:
            logger.warning("Failed to persist audit event: %s", exc)
    event_id = str(uuid.uuid4())
    _append_state_snapshot("audit", event_id, entry)


def _state_event_log_file() -> Path:
    return _resolve_output_root_dir() / _STATE_EVENT_LOG_FILE


def _append_state_snapshot(
    entity_type: str, entity_id: str, payload: Dict[str, Any]
) -> None:
    event = {
        "entity_type": str(entity_type),
        "entity_id": str(entity_id),
        "timestamp": time.time(),
        "payload": payload,
    }
    path = _state_event_log_file()
    try:
        with _state_log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False))
                handle.write("\n")
    except OSError as exc:
        logger.warning("Failed appending state snapshot to %s: %s", path, exc)


def _iter_state_events() -> List[Dict[str, Any]]:
    path = _state_event_log_file()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with _state_log_lock:
            lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Failed reading state event log %s: %s", path, exc)
        return []
    for line in lines:
        try:
            parsed = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _load_state_snapshot(entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
    entity_type_norm = str(entity_type)
    entity_id_norm = str(entity_id)
    latest_payload: Optional[Dict[str, Any]] = None
    latest_ts = -1.0
    for item in _iter_state_events():
        if str(item.get("entity_type", "")) != entity_type_norm:
            continue
        if str(item.get("entity_id", "")) != entity_id_norm:
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        ts = float(item.get("timestamp", 0.0) or 0.0)
        if ts >= latest_ts:
            latest_payload = payload
            latest_ts = ts
    return latest_payload


def _list_state_entity_ids(entity_type: str) -> List[str]:
    entity_type_norm = str(entity_type)
    latest_by_id: Dict[str, float] = {}
    for item in _iter_state_events():
        if str(item.get("entity_type", "")) != entity_type_norm:
            continue
        entity_id = str(item.get("entity_id", "")).strip()
        if not entity_id:
            continue
        ts = float(item.get("timestamp", 0.0) or 0.0)
        prev = latest_by_id.get(entity_id)
        if prev is None or ts > prev:
            latest_by_id[entity_id] = ts
    return sorted(
        latest_by_id.keys(),
        key=lambda item: latest_by_id.get(item, 0.0),
        reverse=True,
    )


def _load_persisted_audit_events(limit: int = 200) -> List[Dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        return []
    try:
        rows = client.lrange(_REDIS_KEY_AUDIT_EVENTS, 0, max(0, int(limit) - 1))
    except Exception as exc:
        logger.warning("Failed loading audit events from Redis: %s", exc)
        return []
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        try:
            item = json.loads(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def _resolve_simulation(sim_id: str) -> Optional[Dict[str, Any]]:
    cached = _simulations.get(sim_id)
    if isinstance(cached, dict):
        return cached
    persisted = _load_persisted_simulation(sim_id)
    if isinstance(persisted, dict):
        _simulations[sim_id] = persisted
    return persisted


def _resolve_workspace(workspace_id: str) -> Optional[Dict[str, Any]]:
    cached = _workspaces.get(workspace_id)
    if isinstance(cached, dict):
        return cached
    persisted = _load_persisted_workspace(workspace_id)
    if isinstance(persisted, dict):
        _workspaces[workspace_id] = persisted
        return persisted
    return None


def _audit(event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={fields[k]}" for k in sorted(fields))
    logger.info("AUDIT %s %s", event, payload)
    entry = {
        "event": str(event),
        "timestamp": time.time(),
        "fields": {str(k): fields[k] for k in fields},
    }
    _audit_events.append(entry)
    _persist_audit_event(entry)


def _record_simulation_request() -> None:
    with _metrics_lock:
        _runtime_metrics["simulation_requests_total"] += 1.0


def _record_simulation_completion(
    duration_seconds: float,
    llm_invocations: int,
    llm_fallback_segments: int,
    *,
    total_segments: int = 0,
    llm_latency_seconds_samples: Optional[List[float]] = None,
) -> None:
    with _metrics_lock:
        duration_value = max(0.0, float(duration_seconds))
        _runtime_metrics["simulation_completed_total"] += 1.0
        _runtime_metrics["simulation_duration_seconds_total"] += duration_value
        _runtime_metrics["llm_invocations_total"] += float(max(0, llm_invocations))
        _runtime_metrics["llm_fallback_segments_total"] += float(
            max(0, llm_fallback_segments)
        )
        _simulation_duration_seconds_history.append(duration_value)
        if int(llm_invocations) > 0:
            _simulation_duration_seconds_llm_history.append(duration_value)
            total_segment_count = max(1, int(total_segments))
            density = float(max(0, int(llm_invocations))) / float(total_segment_count)
            _llm_invocation_density_history.append(max(0.0, min(1.0, density)))
            if llm_latency_seconds_samples:
                for sample in llm_latency_seconds_samples:
                    sample_val = max(0.0, float(sample))
                    if sample_val > 0.0:
                        _llm_invocation_latency_seconds_history.append(sample_val)
        else:
            _simulation_duration_seconds_no_llm_history.append(duration_value)


def _record_simulation_failure(error_code: str = "internal_error") -> None:
    with _metrics_lock:
        _runtime_metrics["simulation_failed_total"] += 1.0
    _record_error_code(error_code)


def _record_api_request(status_code: int) -> None:
    with _metrics_lock:
        _runtime_metrics["api_requests_total"] += 1.0
        if int(status_code) >= 400:
            _runtime_metrics["api_errors_total"] += 1.0


def _record_error_code(error_code: str) -> None:
    with _metrics_lock:
        _api_error_codes[str(error_code)] += 1.0


def _record_export_failure() -> None:
    with _metrics_lock:
        _runtime_metrics["export_failures_total"] += 1.0


def _rate_limited(client_key: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        window = _request_windows[client_key]
        while window and (now - window[0]) > _RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= _API_RATE_LIMIT_PER_MINUTE:
            return True
        window.append(now)
        return False


def _extract_auth_token(request: Request) -> str:
    header_token = request.headers.get("x-api-key", "").strip()
    if header_token:
        return header_token
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return ""


def _resolve_token_policy(token: str) -> tuple[str, set[str]]:
    token_value = str(token or "").strip()
    if not token_value:
        return ("anonymous", set())
    policy = _API_TOKEN_ROLE_MAP.get(token_value)
    if isinstance(policy, dict):
        role = str(policy.get("role", "viewer")).strip().lower() or "viewer"
        scopes_raw = policy.get("scopes", [])
        scopes = {
            str(scope).strip()
            for scope in (scopes_raw if isinstance(scopes_raw, list) else [])
            if str(scope).strip()
        }
        return role, scopes
    if _API_ACCESS_TOKEN and token_value == _API_ACCESS_TOKEN:
        return ("admin", {"*"})
    return ("anonymous", set())


def _scope_allowed(request: Request, scope: str) -> bool:
    if not (_API_ACCESS_TOKEN or _API_TOKEN_ROLE_MAP):
        return True
    required = str(scope).strip()
    if not required:
        return True
    scopes = getattr(request.state, "scopes", set())
    if not isinstance(scopes, set):
        scopes = set()
    return "*" in scopes or required in scopes


def _is_auth_exempt_path(path: str) -> bool:
    public_metrics_paths = (
        {"/metrics", "/metrics/prometheus"} if _METRICS_PUBLIC else set()
    )
    if path in {"/", "/health", *public_metrics_paths}:
        return True
    return path.startswith("/docs") or path.startswith("/openapi")


def _api_client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client is None:
        return "unknown"
    return request.client.host or "unknown"


def _read_runtime_metrics() -> Dict[str, float]:
    with _metrics_lock:
        snapshot = dict(_runtime_metrics)
        error_codes = dict(_api_error_codes)
        duration_history = list(_simulation_duration_seconds_history)
    completed = snapshot.get("simulation_completed_total", 0.0)
    requests_total = snapshot.get("simulation_requests_total", 0.0)
    failed_total = snapshot.get("simulation_failed_total", 0.0)
    duration_total = snapshot.get("simulation_duration_seconds_total", 0.0)
    fallback_total = snapshot.get("llm_fallback_segments_total", 0.0)
    llm_total = snapshot.get("llm_invocations_total", 0.0)
    export_failures_total = snapshot.get("export_failures_total", 0.0)
    api_requests = snapshot.get("api_requests_total", 0.0)
    api_errors = snapshot.get("api_errors_total", 0.0)
    with _jobs_lock:
        jobs = list(_simulation_jobs.values())
    jobs_running = sum(1 for j in jobs if j.get("status") == "running")
    jobs_pending = sum(1 for j in jobs if j.get("status") == "pending")
    jobs_failed = sum(1 for j in jobs if j.get("status") == "failed")
    snapshot["simulation_duration_seconds_avg"] = round(
        (duration_total / completed) if completed > 0 else 0.0, 6
    )
    if duration_history:
        snapshot["simulation_duration_seconds_p95"] = round(
            float(np.percentile(duration_history, 95)), 6
        )
        snapshot["simulation_duration_seconds_p99"] = round(
            float(np.percentile(duration_history, 99)), 6
        )
    else:
        snapshot["simulation_duration_seconds_p95"] = 0.0
        snapshot["simulation_duration_seconds_p99"] = 0.0
    snapshot["llm_fallback_rate"] = round(
        (fallback_total / llm_total) if llm_total > 0 else 0.0, 6
    )
    snapshot["simulation_completion_rate"] = round(
        (completed / requests_total) if requests_total > 0 else 0.0, 6
    )
    snapshot["simulation_failure_rate"] = round(
        (failed_total / requests_total) if requests_total > 0 else 0.0, 6
    )
    snapshot["api_success_rate"] = round(
        ((api_requests - api_errors) / api_requests) if api_requests > 0 else 0.0, 6
    )
    snapshot["export_success_rate"] = round(
        (1.0 - (export_failures_total / completed)) if completed > 0 else 1.0, 6
    )
    snapshot["job_queue_pending"] = float(jobs_pending)
    snapshot["job_queue_running"] = float(jobs_running)
    snapshot["job_queue_failed"] = float(jobs_failed)
    snapshot["error_codes"] = error_codes
    return snapshot


def _build_release_gate_status(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    api_success_rate = float(metrics_snapshot.get("api_success_rate", 0.0))
    simulation_completion_rate = float(
        metrics_snapshot.get("simulation_completion_rate", 0.0)
    )
    latency_p95 = float(metrics_snapshot.get("simulation_duration_seconds_p95", 0.0))
    export_success_rate = float(metrics_snapshot.get("export_success_rate", 0.0))
    quality_snapshot = _recent_simulation_quality_snapshot()

    checks = {
        "api_success_rate_pass": api_success_rate
        >= float(SLO_TARGETS["api_success_rate_min"]),
        "simulation_completion_rate_pass": simulation_completion_rate
        >= float(SLO_TARGETS["simulation_completion_rate_min"]),
        "simulation_latency_p95_pass": latency_p95
        <= float(SLO_TARGETS["simulation_p95_latency_seconds_max"]),
        "export_success_rate_pass": export_success_rate
        >= float(SLO_TARGETS["export_success_rate_min"]),
        "narrative_quality_pass": float(
            quality_snapshot.get("narrative_quality_mean", 0.0)
        )
        >= 0.55,
        "llm_fallback_sla_pass": float(
            quality_snapshot.get("llm_fallback_rate_mean", 1.0)
        )
        <= 0.35,
        "memory_grounding_pass": float(
            quality_snapshot.get("memory_grounding_mean", 0.0)
        )
        >= 0.10,
    }
    breaches = [name for name, ok in checks.items() if not ok]
    return {
        "pass": len(breaches) == 0,
        "checks": checks,
        "breaches": breaches,
        "targets": SLO_TARGETS,
        "current": metrics_snapshot,
        "quality_window": quality_snapshot,
        "notes": [
            "simulation_latency_p95_pass uses in-process p95 over recent simulation durations.",
            "narrative_quality_pass, llm_fallback_sla_pass, and memory_grounding_pass use recent simulation summaries.",
            "Release gate should fail when any check is false.",
        ],
    }


def _recent_simulation_quality_snapshot(max_runs: int = 20) -> Dict[str, Any]:
    simulation_ids = sorted(_simulations.keys())
    if not simulation_ids:
        simulation_ids = sorted(_list_persisted_sim_ids())
    recent_ids = simulation_ids[-max_runs:]
    quality_vals: List[float] = []
    grounding_vals: List[float] = []
    fallback_rates: List[float] = []
    for sim_id in recent_ids:
        payload = _resolve_simulation(str(sim_id))
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary") or {}
        if not isinstance(summary, dict):
            continue
        quality_vals.append(float(summary.get("narrative_quality_mean", 0.0)))
        grounding_vals.append(
            float(summary.get("narrative_memory_grounding_mean", 0.0))
        )
        fallback_segments = float(summary.get("llm_fallback_segments", 0.0))
        llm_total = float(
            summary.get("llm_total_invocations", summary.get("llm_calls_total", 0.0))
        )
        fallback_rates.append((fallback_segments / llm_total) if llm_total > 0 else 0.0)
    count = len(quality_vals)
    if count == 0:
        return {
            "window_size": 0,
            "narrative_quality_mean": 0.0,
            "memory_grounding_mean": 0.0,
            "llm_fallback_rate_mean": 1.0,
        }
    return {
        "window_size": count,
        "narrative_quality_mean": round(sum(quality_vals) / float(count), 4),
        "memory_grounding_mean": round(sum(grounding_vals) / float(count), 4),
        "llm_fallback_rate_mean": round(sum(fallback_rates) / float(count), 4),
    }


# ── Pydantic schemas (inline — no dependency on api/schemas.py) ───────────────


class LLMConfigRequest(BaseModel):
    provider: Optional[str] = Field(None, examples=["lmstudio", "ollama", "openai"])
    base_url: Optional[str] = Field(None, examples=["http://localhost:1234/v1"])
    model: Optional[str] = Field(None, examples=["qwen/qwen3.5-9b", "llama3", "gpt-4o"])
    api_key: Optional[str] = Field(None, examples=["lm-studio", "ollama", "sk-..."])
    max_tokens: Optional[int] = Field(None, ge=64, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    timeout: Optional[int] = Field(None, ge=10, le=600)


class LLMConfigResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_set: bool
    max_tokens: int
    temperature: float
    timeout: int


class LLMHealthResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    base_url: str
    available_models: List[str] = []
    error: Optional[str] = None


class SimulationConfig(BaseModel):
    duration_hours: float = Field(8.0, ge=1.0, le=12.0)
    dt_minutes: float = Field(0.5, ge=0.1, le=5.0)
    ssri_strength: float = Field(1.0, ge=0.0, le=3.0)
    stress_level: float = Field(0.2, ge=0.0, le=1.0)
    sleep_start_hour: float = Field(23.0, ge=0.0, le=26.0)
    melatonin: bool = Field(False)
    cannabis: bool = Field(False)
    prior_day_events: List[str] = Field(default_factory=list)
    emotional_state: str = Field("neutral")
    style_preset: str = Field("scientific")
    prompt_profile: str = Field("A")
    use_llm: bool = Field(True)
    llm_segments_only: bool = Field(False)


class DreamSegmentResponse(BaseModel):
    id: str
    start_time_hours: float
    end_time_hours: float
    stage: str
    narrative: str
    scene_description: str
    dominant_emotion: str
    bizarreness_score: float
    lucidity_probability: float
    is_lucid: bool = False
    neurochemistry: Optional[Dict[str, float]] = None
    active_memory_ids: List[str] = []
    generation_mode: str = "TEMPLATE"
    llm_trigger_type: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    llm_fallback_reason: Optional[str] = None
    template_bank: Optional[str] = None
    narrative_quality: Optional[Dict[str, float]] = None


class SimulationResponse(BaseModel):
    id: str
    config: SimulationConfig
    segments: List[DreamSegmentResponse]
    summary: Dict[str, Any] = {}
    neurochemistry_ticks: List[Dict[str, Any]] = Field(default_factory=list)
    neurochemistry_series: List[Dict[str, Any]] = Field(default_factory=list)
    memory_activations: List[Dict[str, Any]] = Field(default_factory=list)
    memory_activation_series: List[Dict[str, Any]] = Field(default_factory=list)
    memory_graph: Dict[str, Any] = Field(default_factory=dict)
    lucid_events: List[Dict[str, Any]] = Field(default_factory=list)
    llm_used: bool = False
    llm_model: Optional[str] = None


class MultiNightSimulationRequest(BaseModel):
    nights: int = Field(default=3, ge=2, le=30)
    config: SimulationConfig
    carryover_memory: bool = Field(default=True)
    carryover_top_k: int = Field(default=5, ge=1, le=20)
    max_prior_events: int = Field(default=12, ge=1, le=40)


class MultiNightSimulationResponse(BaseModel):
    series_id: str
    nights: List[SimulationResponse]
    summary: Dict[str, Any] = {}
    continuity: Dict[str, Any] = {}


class CounterfactualRequest(BaseModel):
    base_simulation_id: str
    perturbations: Dict[str, Any] = Field(...)
    use_llm: bool = True


class CompareRequest(BaseModel):
    baseline_simulation_id: str
    candidate_simulation_id: str


class AsyncSimulationSubmitResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


class AsyncSimulationJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    progress_percent: float = 0.0
    schema_version: str = "v2"
    phase: Optional[str] = None
    progress_source: Optional[str] = None
    eta_seconds: Optional[int] = None
    eta_source: Optional[str] = None
    eta_margin_seconds: Optional[int] = None
    estimated_duration_seconds: Optional[float] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    simulation_id: Optional[str] = None


class AsyncSimulationCancelResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PSGChannelQARequest(BaseModel):
    channels: List[str] = Field(default_factory=list)
    expected_roles: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "eeg": ["C3", "C4", "F3", "F4", "EEG"],
            "eog": ["EOG-L", "EOG-R", "EOG", "LOC", "ROC"],
            "emg": ["EMG", "CHIN"],
        }
    )


class PSGChannelQAResponse(BaseModel):
    detected_channels: List[str] = Field(default_factory=list)
    matched_roles: Dict[str, str] = Field(default_factory=dict)
    missing_roles: List[str] = Field(default_factory=list)
    extras: List[str] = Field(default_factory=list)
    pass_check: bool = False


class EvaluatorRunRequest(BaseModel):
    evaluator: str = Field(default="quality-v1")
    summary: Dict[str, Any] = Field(default_factory=dict)


class ChartExportRequest(BaseModel):
    figure: Dict[str, Any]
    format: str = Field(default="png")
    scale: float = Field(default=2.0, ge=0.5, le=4.0)


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list, max_length=20)


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: float
    updated_at: float
    run_ids: List[str] = Field(default_factory=list)


class WorkspaceAttachRunRequest(BaseModel):
    simulation_id: str = Field(..., min_length=1)
    label: Optional[str] = Field(None, max_length=120)


class WorkspaceAttachRunResponse(BaseModel):
    workspace_id: str
    simulation_id: str
    attached: bool
    run_count: int


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="DreamForge AI",
    description=(
        "## DreamForge AI REST API\n\n"
        "Multi-agent computational dream simulation with LLM-powered narrative generation.\n\n"
        "### Quick start\n"
        "1. `POST /api/llm/config` — point at your LLM (LM Studio / Ollama / OpenAI)\n"
        "2. `GET  /api/health/llm` — verify connectivity\n"
        "3. `POST /api/simulation/night` — run a full dream simulation\n"
        "4. `GET  /api/simulation/{id}` — retrieve results\n"
    ),
    version="0.2.0",
    contact={"name": "DreamForge", "url": "https://github.com/your-org/dreamforge-ai"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _security_and_telemetry_middleware(request: Request, call_next):
    path = request.url.path
    request_id = uuid.uuid4().hex
    started_at = time.perf_counter()
    request.state.request_id = request_id
    request.state.role = "anonymous"
    request.state.scopes = set()

    if path.startswith("/api") and not _is_auth_exempt_path(path):
        token = _extract_auth_token(request)
        auth_enabled = bool(_API_ACCESS_TOKEN or _API_TOKEN_ROLE_MAP)
        if auth_enabled:
            role, scopes = _resolve_token_policy(token)
            if role == "anonymous":
                with _metrics_lock:
                    _runtime_metrics["api_unauthorized_total"] += 1.0
                _record_error_code("unauthorized")
                _record_api_request(401)
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Missing or invalid API key",
                        "request_id": request_id,
                    },
                )
            request.state.role = role
            request.state.scopes = scopes
        elif token:
            request.state.role = "admin"
            request.state.scopes = {"*"}

        if _rate_limited(_api_client_key(request)):
            with _metrics_lock:
                _runtime_metrics["api_rate_limited_total"] += 1.0
            _record_error_code("rate_limited")
            _record_api_request(429)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": "API rate limit exceeded",
                    "request_id": request_id,
                },
            )

    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    logger.info(
        "TRACE request path=%s method=%s status=%s duration_ms=%s request_id=%s role=%s",
        path,
        request.method,
        response.status_code,
        duration_ms,
        request_id,
        getattr(request.state, "role", "anonymous"),
    )
    _record_api_request(response.status_code)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    status_code = int(exc.status_code)
    if status_code == 404:
        code = "not_found"
    elif status_code in {401, 403}:
        code = "unauthorized"
    elif status_code == 422:
        code = "validation_error"
    elif status_code == 429:
        code = "rate_limited"
    else:
        code = "internal_error" if status_code >= 500 else "provider_error"
    _record_error_code(code)
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": exc.detail,
            "error": code,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    _record_error_code("internal_error")
    logger.exception("Unhandled API exception: %s", exc)
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": "internal_error",
            "request_id": request_id,
        },
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    client = get_llm_client()
    health = await client.check_health()
    if health.get("ok"):
        logger.info(
            "LLM backend reachable at %s — models: %s",
            client.config.base_url,
            health.get("models"),
        )
    else:
        logger.warning(
            "LLM backend NOT reachable at %s: %s",
            client.config.base_url,
            health.get("error"),
        )
    artifact_health = _artifact_health_snapshot(force_refresh=True)
    if artifact_health.get("pass"):
        logger.info(
            "Artifact manifest healthy (%s items): %s",
            artifact_health.get("total_artifacts", 0),
            artifact_health.get("manifest_path", ""),
        )
    else:
        logger.warning(
            "Artifact manifest check failed: missing=%s mismatched_hash=%s",
            artifact_health.get("required_missing", []),
            artifact_health.get("mismatched_hash", []),
        )


# ── Helper: biophysical simulation ───────────────────────────────────────────


def _rem_episode_features(
    sleep_states: List[Any], n_steps: int
) -> tuple[list[float], list[int]]:
    rem_fraction = [0.0 for _ in range(n_steps)]
    rem_episode = [0 for _ in range(n_steps)]
    rem_groups: list[list[int]] = []
    active_group: list[int] = []
    for i in range(n_steps):
        state_idx = min(i + 1, len(sleep_states) - 1)
        stage = sleep_states[state_idx].stage.value
        if stage == "REM":
            active_group.append(i)
        elif active_group:
            rem_groups.append(active_group[:])
            active_group = []
    if active_group:
        rem_groups.append(active_group[:])

    for ep_idx, group in enumerate(rem_groups, start=1):
        group_len = max(1, len(group))
        for pos, tick_idx in enumerate(group):
            frac = 1.0 if group_len == 1 else float(pos) / float(group_len - 1)
            rem_fraction[tick_idx] = frac
            rem_episode[tick_idx] = ep_idx
    return rem_fraction, rem_episode


def _annotate_lucid_events(
    segments: List[dict], lucidity_threshold: float
) -> List[Dict[str, Any]]:
    lucid_events: List[Dict[str, Any]] = []
    streak_start = -1
    streak_peak = 0.0
    for idx, seg in enumerate(segments):
        score = float(seg.get("lucidity_probability") or 0.0)
        if score >= lucidity_threshold:
            if streak_start < 0:
                streak_start = idx
                streak_peak = score
            else:
                streak_peak = max(streak_peak, score)
        else:
            if streak_start >= 0:
                run_len = idx - streak_start
                if run_len >= 3:
                    for mark_idx in range(streak_start, idx):
                        segments[mark_idx]["is_lucid"] = True
                    lucid_events.append(
                        {
                            "time_hours": float(
                                segments[streak_start].get("start_time_hours", 0.0)
                            ),
                            "duration_ticks": run_len,
                            "peak_lucidity": round(streak_peak, 4),
                        }
                    )
                streak_start = -1
                streak_peak = 0.0
    if streak_start >= 0:
        run_len = len(segments) - streak_start
        if run_len >= 3:
            for mark_idx in range(streak_start, len(segments)):
                segments[mark_idx]["is_lucid"] = True
            lucid_events.append(
                {
                    "time_hours": float(
                        segments[streak_start].get("start_time_hours", 0.0)
                    ),
                    "duration_ticks": run_len,
                    "peak_lucidity": round(streak_peak, 4),
                }
            )
    return lucid_events


def _simulate_night_physics(config: SimulationConfig) -> List[dict]:
    """
    Lightweight biophysical simulation (Process S/C + neurochemistry).
    Returns list of segment dicts with stage, neurochemistry snapshot, etc.
    No LLM involved here.
    """
    dt_h = config.dt_minutes / 60.0
    n_steps = int(config.duration_hours / dt_h)
    sleep_model = SleepCycleModel()
    sleep_states, _ = sleep_model.simulate_night(
        duration_hours=config.duration_hours,
        dt_minutes=config.dt_minutes,
        sleep_start_clock_time=config.sleep_start_hour,
    )
    rem_fraction_by_tick, rem_episode_by_tick = _rem_episode_features(
        sleep_states, n_steps
    )
    lucidity_model = LucidityModel.from_settings()

    segments = []

    # Simple neurochemistry baselines by stage (cortisol is computed from profile)
    neuro_by_stage = {
        "N1": {"ach": 0.50, "serotonin": 0.30, "ne": 0.30},
        "N2": {"ach": 0.45, "serotonin": 0.28, "ne": 0.28},
        "N3": {"ach": 0.30, "serotonin": 0.20, "ne": 0.20},
        "REM": {"ach": 0.90, "serotonin": 0.05, "ne": 0.05},
        "WAKE": {"ach": 0.70, "serotonin": 0.80, "ne": 0.80},
    }

    # Stage-calibrated baseline/ceiling values.
    # Keep REM highly bizarre while allowing occasional high-biz NREM episodes.
    stage_base = {"N1": 0.10, "N2": 0.14, "N3": 0.08, "REM": 0.40, "WAKE": 0.04}
    stage_ceiling = {"N1": 0.58, "N2": 0.62, "N3": 0.35, "REM": 0.98, "WAKE": 0.20}

    emotions = [
        "neutral",
        "curious",
        "anxious",
        "joyful",
        "melancholic",
        "fearful",
        "serene",
    ]
    emotion_weights_by_stress = {
        "low": [0.30, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05],
        "high": [0.10, 0.10, 0.35, 0.10, 0.15, 0.15, 0.05],
    }

    stress_cat = "high" if config.stress_level > 0.5 else "low"
    emo_weights = emotion_weights_by_stress[stress_cat]

    for i in range(n_steps):
        t = i * dt_h
        state_idx = min(i + 1, len(sleep_states) - 1)
        stage = sleep_states[state_idx].stage.value
        cycle_idx = int(t // 1.5) + 1
        rem_fraction = rem_fraction_by_tick[i]
        rem_episode = rem_episode_by_tick[i]

        # Neurochemistry with noise + SSRI modulation
        neuro = dict(neuro_by_stage[stage])
        neuro["serotonin"] = min(1.0, neuro["serotonin"] * config.ssri_strength)
        neuro["cortisol"] = cortisol_profile(t)
        neuro = {k: max(0.0, v + random.gauss(0, 0.02)) for k, v in neuro.items()}

        # Parametric bizarreness with stage ceilings to keep NREM physiologically lower.
        memory_arousal = float(np.clip(config.stress_level, 0.0, 1.0))
        biz_val = (
            stage_base.get(stage, 0.18)
            + 0.25 * neuro["ach"]
            + 0.20 * (1.0 - neuro["ne"])
            + 0.10 * memory_arousal
            + 0.03 * min(cycle_idx, 3)
            + random.gauss(0, 0.03)
        )
        if stage == "N2":
            # Keep high-biz N2 reachable but not dominant.
            biz_val += 0.02 * memory_arousal - 0.045
        elif stage == "N1":
            # N1 should only rarely exceed the high-biz trigger gate.
            biz_val += 0.01 * memory_arousal - 0.05
        bizarreness = round(
            float(np.clip(biz_val, 0.0, stage_ceiling.get(stage, 0.95))),
            3,
        )

        # Lucidity model calibrated for late-night REM peaks.
        if stage != "REM":
            lucidity = 0.0
        else:
            late_rem_gain = min(1.0, max(0.0, (float(rem_episode) - 2.0) / 2.0))
            rem_depth = (
                0.46
                + 0.16 * float(rem_fraction)
                + 0.14 * float(late_rem_gain)
                + 0.08 * float(max(0.0, neuro["ach"] - neuro["ne"]))
                + random.gauss(0.0, 0.015)
            )
            rem_depth = float(np.clip(rem_depth, 0.0, 0.85))
            lucidity = lucidity_model.compute_lucidity(
                LucidityTickState(
                    stage=stage,
                    rem_depth=rem_depth,
                    t_rem_fraction=rem_fraction,
                    cycle_index=cycle_idx,
                )
            )
        lucidity = round(lucidity, 4)

        # Dominant emotion
        dominant_emotion = random.choices(emotions, weights=emo_weights, k=1)[0]
        if config.emotional_state != "neutral" and random.random() < 0.3:
            dominant_emotion = config.emotional_state

        segments.append(
            {
                "id": str(uuid.uuid4()),
                "start_time_hours": round(t, 6),
                "end_time_hours": round(t + dt_h, 6),
                "stage": stage,
                "dominant_emotion": dominant_emotion,
                "bizarreness_score": bizarreness,
                "lucidity_probability": lucidity,
                "is_lucid": False,
                "neurochemistry": {k: round(v, 4) for k, v in neuro.items()},
                "active_memory_ids": [],
                # Placeholders — will be replaced by LLM below
                "narrative": "",
                "scene_description": "",
                "llm_trigger_type": None,
                "llm_latency_ms": None,
                "template_bank": f"TEMPLATE_{stage}",
            }
        )

    return segments


def _build_neurochemistry_ticks(segments: List[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seg in segments:
        neuro = seg.get("neurochemistry") or {}
        rows.append(
            {
                "time_hours": seg.get("start_time_hours"),
                "stage": seg.get("stage"),
                "ach": neuro.get("ach"),
                "serotonin": neuro.get("serotonin"),
                "ne": neuro.get("ne"),
                "cortisol": neuro.get("cortisol"),
            }
        )
    return rows


def _build_memory_outputs(
    segments: List[dict], prior_day_events: List[str]
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    node_map: Dict[str, Dict[str, Any]] = {}
    snapshots: List[Dict[str, Any]] = []

    semantic_seed = [
        ("childhood_home", "episodic", -0.1),
        ("school_corridor", "episodic", -0.2),
        ("familiar_face", "episodic", 0.1),
        ("recurring_journey", "episodic", 0.0),
        ("falling", "conceptual", -0.6),
        ("being_chased", "conceptual", -0.7),
        ("flying", "conceptual", 0.6),
        ("being_late", "conceptual", -0.4),
        ("teeth_loosening", "conceptual", -0.5),
        ("cold_water", "sensory", -0.1),
        ("bright_light", "sensory", 0.2),
        ("loud_crowd", "sensory", -0.2),
        ("grief", "emotional", -0.8),
        ("joy", "emotional", 0.8),
        ("dread", "emotional", -0.9),
        ("wonder", "emotional", 0.5),
    ]
    for label, category, valence in semantic_seed:
        node_id = f"mem::{label}"
        node_map[node_id] = {
            "id": node_id,
            "label": label,
            "category": category,
            "valence": float(valence),
            "activation": round(random.uniform(0.1, 0.4), 4),
            "salience": round(random.uniform(0.3, 0.8), 4),
        }

    for event in prior_day_events:
        label = event.strip()[:80]
        if not label:
            continue
        node_id = f"event::{label.lower().replace(' ', '_')}"
        node_map[node_id] = {
            "id": node_id,
            "label": label,
            "category": "episodic",
            "valence": round(random.uniform(-0.4, 0.4), 4),
            "activation": round(random.uniform(0.2, 0.5), 4),
            "salience": round(random.uniform(0.4, 0.9), 4),
        }

    nodes = list(node_map.values())
    weighted_edges: List[Dict[str, Any]] = []
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            src_vec = np.array(
                [
                    src["valence"],
                    src["salience"],
                    src["activation"],
                    float(len(src["label"])) / 20.0,
                    1.0 if src["category"] == dst["category"] else 0.0,
                ],
                dtype=float,
            )
            dst_vec = np.array(
                [
                    dst["valence"],
                    dst["salience"],
                    dst["activation"],
                    float(len(dst["label"])) / 20.0,
                    1.0 if src["category"] == dst["category"] else 0.0,
                ],
                dtype=float,
            )
            denom = float(np.linalg.norm(src_vec) * np.linalg.norm(dst_vec))
            if denom <= 0:
                continue
            weight = float(np.dot(src_vec, dst_vec) / denom)
            same_category = src["category"] == dst["category"]
            valence_gap = abs(float(src["valence"]) - float(dst["valence"]))
            if weight >= 0.22 and (same_category or valence_gap <= 0.7):
                weighted_edges.append(
                    {
                        "source": src["id"],
                        "target": dst["id"],
                        "weight": round(weight, 4),
                    }
                )
    # Keep graph sparse enough for meaningful communities in dashboard rendering.
    # We cap node degree while preserving strongest semantic links.
    max_degree = 5
    degree_count: Dict[str, int] = defaultdict(int)
    edges: List[Dict[str, Any]] = []
    for edge in sorted(weighted_edges, key=lambda e: float(e["weight"]), reverse=True):
        src_id = str(edge.get("source"))
        dst_id = str(edge.get("target"))
        if degree_count[src_id] >= max_degree or degree_count[dst_id] >= max_degree:
            continue
        edges.append(edge)
        degree_count[src_id] += 1
        degree_count[dst_id] += 1

    prev_time = 0.0
    for idx, seg in enumerate(segments):
        t = float(seg.get("start_time_hours") or 0.0)
        dt_h = max(1 / 120.0, t - prev_time)
        prev_time = t
        stage = str(seg.get("stage") or "N2")
        emotion = str(seg.get("dominant_emotion") or "neutral").lower()
        neuro = seg.get("neurochemistry") or {}
        ach = float(neuro.get("ach") or 0.0)

        decay_rate = 0.3 if stage == "REM" else 0.8
        decay_factor = float(np.exp(-decay_rate * dt_h))
        for node in nodes:
            node["activation"] = round(
                max(0.0, min(1.0, float(node["activation"]) * decay_factor)), 4
            )

        if stage == "REM":
            ranked = sorted(nodes, key=lambda n: float(n["activation"]), reverse=True)[
                :2
            ]
            for node in ranked:
                current_activation = float(node["activation"])
                headroom = max(0.0, 0.95 - current_activation)
                boost = float(node["salience"]) * ach * 0.35 * (headroom**1.2)
                node["activation"] = round(min(0.95, current_activation + boost), 4)

        target_valence = (
            -1.0
            if emotion in {"fearful", "anxious", "melancholic", "dread"}
            else 1.0 if emotion in {"joyful", "serene", "wonder"} else 0.0
        )
        for node in nodes:
            if target_valence != 0.0 and float(node["valence"]) * target_valence > 0:
                current_activation = float(node["activation"])
                headroom = max(0.0, 0.92 - current_activation)
                node["activation"] = round(
                    min(0.92, current_activation + (0.12 * headroom)), 4
                )

        active_ids = [
            n["id"]
            for n in sorted(nodes, key=lambda x: float(x["activation"]), reverse=True)
            if float(n["activation"]) >= 0.50
        ][:5]
        seg["active_memory_ids"] = active_ids

        if idx % 10 == 0:
            snapshots.append(
                {
                    "time_hours": t,
                    "stage": stage,
                    "activations": [
                        {
                            "id": n["id"],
                            "label": n["label"],
                            "activation": float(n["activation"]),
                        }
                        for n in sorted(
                            nodes, key=lambda x: float(x["activation"]), reverse=True
                        )[:12]
                    ],
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
        "activation_snapshots": snapshots,
    }, snapshots


def _get_template_bank() -> TemplateBank | None:
    global _template_bank, _template_bank_initialized
    if _template_bank_initialized:
        return _template_bank
    _template_bank_initialized = True
    try:
        bank = TemplateBank(data_dir=Path("core") / "data")
        bank.load()
        _template_bank = bank
        return _template_bank
    except (FileNotFoundError, OSError, SchemaValidationError, ValueError) as exc:
        logger.warning("TemplateBank unavailable, using inline templates: %s", exc)
        _template_bank = None
        return None


def _template_narrative(
    seg: dict,
    config: SimulationConfig,
    *,
    segment_index: int = 0,
    narrative_cache: NarrativeCache | None = None,
) -> tuple[str, str, str]:
    """Fallback template-based narrative when LLM is disabled."""

    def _force_word_window(text: str, minimum: int, maximum: int, *, pad: str) -> str:
        words = str(text).split()
        if len(words) < minimum:
            pad_words = pad.split()
            missing = minimum - len(words)
            while missing > 0:
                words.extend(pad_words[:missing])
                missing = minimum - len(words)
        if len(words) > maximum:
            words = words[:maximum]
        out = " ".join(words).strip()
        if out and out[-1] not in ".!?":
            out += "."
        return out

    def _window_for_stage(stage_name: str, bizarre_score: float) -> tuple[int, int]:
        if stage_name == "REM":
            return (60, 90) if bizarre_score >= 0.8 else (40, 60)
        if stage_name == "N2":
            return (20, 35)
        if stage_name == "N3":
            return (10, 20)
        if stage_name == "N1":
            return (10, 15)
        return (10, 20)

    def _normalize_template(
        text: str,
        stage_name: str,
        bizarre_score: float,
        marker_hours: float,
        seg_idx: int,
    ) -> str:
        min_words, max_words = _window_for_stage(stage_name, bizarre_score)
        base = " ".join(str(text).split())
        if stage_name in {"N1", "N2", "N3"}:
            variation_bank = {
                "N1": [
                    "a faint edge of wakefulness keeps intruding",
                    "small sensory flashes arrive and vanish quickly",
                    "the scene blurs as attention slips in and out",
                ],
                "N2": [
                    "familiar places merge into fragmented corridors",
                    "voices and objects recombine before they settle",
                    "memory fragments overlap in unstable sequences",
                    "the environment shifts between partial storylines",
                ],
                "N3": [
                    "wordless imagery pulses through heavy darkness",
                    "slow symbolic flashes appear without clear sequence",
                    "near-silent sensations drift and dissolve",
                    "deep textures replace coherent plot structure",
                ],
            }
            options = variation_bank.get(stage_name, ["the image tilts and resets"])
            variation = options[seg_idx % len(options)]
            base = f"At {marker_hours:.3f}h, {variation}. {base}"
        if stage_name == "N3":
            pad = (
                "wordless textures drift through darkness while sensation pulses softly"
            )
        elif stage_name == "N2":
            pad = "fragmented impressions echo as places and voices dissolve before they settle"
        elif stage_name == "N1":
            pad = "liminal flashes pass quickly across awareness at the edge of sleep"
        elif stage_name == "REM":
            pad = "the dream keeps mutating through symbolic transitions and intense sensory detail"
        else:
            pad = "the scene remains faint and unstable as perception reorients"
        return _force_word_window(base, min_words, max_words, pad=pad)

    stage = seg["stage"]
    emotion = seg["dominant_emotion"]
    bizarre = seg["bizarreness_score"]
    time_marker = float(seg.get("start_time_hours") or 0.0)

    if narrative_cache is not None and stage in {s.value for s in SleepStage}:
        narrative = narrative_cache.get_segment_narrative(
            segment_index=segment_index,
            emotion=str(emotion),
            stage=SleepStage(stage),
            nchem=seg.get("neurochemistry", {}),
        )
        if narrative:
            template_id = narrative_cache.last_template_id or f"TEMPLATE_{stage}"
            if narrative_cache.last_template_id:
                logger.debug(
                    "template_source=yaml stage=%s template_id=%s",
                    stage,
                    narrative_cache.last_template_id,
                )
            scene = f"A {emotion} dreamscape at {stage} depth."
            return (
                _normalize_template(
                    narrative,
                    stage,
                    float(bizarre),
                    time_marker,
                    segment_index,
                ),
                " ".join(scene.split()[:25]),
                template_id,
            )

    templates = {
        "REM": f"The dream unfolds vividly — a {emotion} scene fractured by impossible geometry. "
        f"Faces shift and timelines collapse. Bizarreness index: {bizarre:.2f}.",
        "N3": f"Deep dreamless silence, punctuated by fleeting sensations. A faint {emotion} undercurrent.",
        "N2": f"Fragmented images surface briefly — a {emotion} echo of the prior day.",
        "N1": f"The boundary between wake and sleep blurs. A {emotion} hypnagogic drift.",
        "WAKE": "A brief awakening. The room is dark. Fragments dissolve.",
    }
    scene_templates = {
        "REM": f"A surreal landscape shaped by {emotion} — colours oversaturated, gravity optional.",
        "N3": "An infinite dark plain, warm and formless.",
        "N2": f"Flickering impressions: corridors, voices, a {emotion} feeling without source.",
        "N1": "The hypnagogic edge — phosphenes and whispers.",
        "WAKE": "A brief moment of wakefulness in darkness.",
    }
    narrative = templates.get(stage, "")
    return (
        _normalize_template(
            narrative,
            stage,
            float(bizarre),
            time_marker,
            segment_index,
        ),
        " ".join(scene_templates.get(stage, "").split()[:25]),
        f"TEMPLATE_{stage}",
    )


def _resolve_output_dir(session_id: str) -> Path:
    """Resolve per-simulation output directory from env or default pattern."""
    output_env = os.getenv("DREAMFORGE_OUTPUT_DIR", "").strip()
    if output_env:
        if "{session_id}" in output_env:
            target = Path(output_env.format(session_id=session_id))
        else:
            target = Path(output_env) / session_id
    else:
        target = Path("outputs") / session_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_output_root_dir() -> Path:
    """Resolve outputs root directory for index/retention operations."""
    output_env = os.getenv("DREAMFORGE_OUTPUT_DIR", "").strip()
    if output_env and "{session_id}" not in output_env:
        root = Path(output_env)
    else:
        root = Path("outputs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _outputs_index_file() -> Path:
    return _resolve_output_root_dir() / "index.json"


def _artifact_manifest_file() -> Path:
    manifest_path = Path(_ARTIFACT_MANIFEST_PATH)
    if manifest_path.is_absolute():
        return manifest_path
    return Path.cwd() / manifest_path


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.warning("Failed hashing artifact %s: %s", path, exc)
        return None
    return digest.hexdigest()


def _build_artifact_health_snapshot() -> Dict[str, Any]:
    checked_at = time.time()
    manifest_path = _artifact_manifest_file()
    if not manifest_path.exists():
        return {
            "manifest_loaded": False,
            "manifest_path": str(manifest_path),
            "schema_version": None,
            "checked_at": checked_at,
            "pass": False,
            "error": "manifest_not_found",
            "compatibility": {},
            "artifacts": [],
            "total_artifacts": 0,
            "required_missing": ["manifest"],
            "mismatched_hash": [],
        }
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed reading artifact manifest %s: %s", manifest_path, exc)
        return {
            "manifest_loaded": False,
            "manifest_path": str(manifest_path),
            "schema_version": None,
            "checked_at": checked_at,
            "pass": False,
            "error": "manifest_parse_error",
            "compatibility": {},
            "artifacts": [],
            "total_artifacts": 0,
            "required_missing": ["manifest"],
            "mismatched_hash": [],
        }
    if not isinstance(data, dict):
        return {
            "manifest_loaded": False,
            "manifest_path": str(manifest_path),
            "schema_version": None,
            "checked_at": checked_at,
            "pass": False,
            "error": "manifest_invalid_shape",
            "compatibility": {},
            "artifacts": [],
            "total_artifacts": 0,
            "required_missing": ["manifest"],
            "mismatched_hash": [],
        }

    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    mismatched_hash: List[str] = []
    raw_artifacts = data.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raw_artifacts = []
    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip() or "unnamed"
        path_val = str(item.get("path", "")).strip()
        required = bool(item.get("required", False))
        expected_sha = str(item.get("sha256", "")).strip() or None
        artifact_path = Path(path_val) if path_val else Path(name)
        if not artifact_path.is_absolute():
            artifact_path = Path.cwd() / artifact_path
        exists = artifact_path.exists() and artifact_path.is_file()
        actual_sha = _sha256_file(artifact_path) if exists else None
        hash_match = (
            (actual_sha == expected_sha)
            if (exists and expected_sha is not None)
            else None
        )
        if required and not exists:
            missing_required.append(name)
        if expected_sha and hash_match is False:
            mismatched_hash.append(name)
        rows.append(
            {
                "name": name,
                "path": str(artifact_path),
                "required": required,
                "exists": exists,
                "kind": str(item.get("kind", "unknown")),
                "expected_sha256": expected_sha,
                "sha256": actual_sha,
                "hash_match": hash_match,
            }
        )

    return {
        "manifest_loaded": True,
        "manifest_path": str(manifest_path),
        "schema_version": int(data.get("schema_version", 1)),
        "checked_at": checked_at,
        "pass": len(missing_required) == 0 and len(mismatched_hash) == 0,
        "error": None,
        "compatibility": (
            data.get("compatibility")
            if isinstance(data.get("compatibility"), dict)
            else {}
        ),
        "artifacts": rows,
        "total_artifacts": len(rows),
        "required_missing": missing_required,
        "mismatched_hash": mismatched_hash,
    }


def _artifact_health_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    global _artifact_health_cache, _artifact_health_cached_at
    now_ts = time.time()
    with _artifact_health_lock:
        if (
            not force_refresh
            and _artifact_health_cache
            and (now_ts - _artifact_health_cached_at) <= 30.0
        ):
            return dict(_artifact_health_cache)
        snapshot = _build_artifact_health_snapshot()
        _artifact_health_cache = dict(snapshot)
        _artifact_health_cached_at = now_ts
    return snapshot


def _load_outputs_index() -> Dict[str, Any]:
    index_path = _outputs_index_file()
    if not index_path.exists():
        return {"version": 1, "updated_at": time.time(), "items": []}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed reading outputs index %s: %s", index_path, exc)
        return {"version": 1, "updated_at": time.time(), "items": []}
    if not isinstance(data, dict):
        return {"version": 1, "updated_at": time.time(), "items": []}
    items = data.get("items", [])
    if not isinstance(items, list):
        items = []
    return {
        "version": int(data.get("version", 1)),
        "updated_at": float(data.get("updated_at", time.time())),
        "items": [item for item in items if isinstance(item, dict)],
    }


def _save_outputs_index(index_data: Dict[str, Any]) -> None:
    index_path = _outputs_index_file()
    payload = {
        "version": 1,
        "updated_at": time.time(),
        "items": index_data.get("items", []),
    }
    try:
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed writing outputs index %s: %s", index_path, exc)


def _upsert_output_metadata(
    sim_id: str, result_payload: Dict[str, Any], output_dir: Path
) -> None:
    summary = result_payload.get("summary", {})
    entry = {
        "simulation_id": sim_id,
        "created_at": float(result_payload.get("created_at_unix", time.time())),
        "output_dir": str(output_dir.resolve()),
        "llm_used": bool(result_payload.get("llm_used", False)),
        "segments": int(summary.get("total_segments", 0) or 0),
        "rem_fraction": float(summary.get("rem_fraction", 0.0) or 0.0),
        "narrative_quality_mean": float(
            summary.get("narrative_quality_mean", 0.0) or 0.0
        ),
    }
    index_data = _load_outputs_index()
    existing = [
        item
        for item in index_data["items"]
        if str(item.get("simulation_id", "")) != sim_id
    ]
    existing.append(entry)
    existing.sort(key=lambda item: float(item.get("created_at", 0.0)), reverse=True)
    index_data["items"] = existing[:_OUTPUT_RETENTION_MAX_RUNS]
    _save_outputs_index(index_data)


def _enforce_outputs_retention() -> None:
    root = _resolve_output_root_dir()
    now = time.time()
    cutoff = now - (_OUTPUT_RETENTION_DAYS * 86400)

    candidates: List[Path] = []
    try:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    except OSError as exc:
        logger.warning("Failed listing output directories in %s: %s", root, exc)
        return

    candidates_sorted = sorted(
        candidates,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    stale_by_count = set(candidates_sorted[_OUTPUT_RETENTION_MAX_RUNS:])
    stale_by_age = {
        path
        for path in candidates_sorted
        if _OUTPUT_RETENTION_DAYS > 0 and path.stat().st_mtime < cutoff
    }
    stale = stale_by_count | stale_by_age
    for path in stale:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning("Failed deleting stale output directory %s: %s", path, exc)

    index_data = _load_outputs_index()
    kept_paths = {str(path.resolve()) for path in candidates if path not in stale}
    index_data["items"] = [
        item
        for item in index_data["items"]
        if str(item.get("output_dir", "")) in kept_paths
    ][:_OUTPUT_RETENTION_MAX_RUNS]
    _save_outputs_index(index_data)


def _strip_thinking_tags(response: str) -> str:
    if not response:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _normalize_channel_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _psg_channel_qa(payload: PSGChannelQARequest) -> PSGChannelQAResponse:
    channels = [str(item).strip() for item in payload.channels if str(item).strip()]
    normalized_to_raw = {_normalize_channel_name(item): item for item in channels}
    matched_roles: Dict[str, str] = {}
    missing_roles: List[str] = []
    used_channels: set[str] = set()

    for role, aliases in payload.expected_roles.items():
        alias_list = [str(alias).strip() for alias in aliases if str(alias).strip()]
        match: Optional[str] = None
        for alias in alias_list:
            alias_norm = _normalize_channel_name(alias)
            for channel_norm, raw_channel in normalized_to_raw.items():
                if alias_norm and alias_norm in channel_norm:
                    match = raw_channel
                    break
            if match is not None:
                break
        role_key = str(role).strip().lower() or str(role)
        if match is None:
            missing_roles.append(role_key)
        else:
            matched_roles[role_key] = match
            used_channels.add(match)

    extras = sorted([item for item in channels if item not in used_channels])
    return PSGChannelQAResponse(
        detected_channels=channels,
        matched_roles=matched_roles,
        missing_roles=sorted(missing_roles),
        extras=extras,
        pass_check=len(missing_roles) == 0,
    )


def _evaluate_quality_v1(summary: Dict[str, Any]) -> Dict[str, float]:
    quality = float(summary.get("narrative_quality_mean", 0.0) or 0.0)
    grounding = float(summary.get("narrative_memory_grounding_mean", 0.0) or 0.0)
    llm_total = float(summary.get("llm_total_invocations", 0.0) or 0.0)
    llm_fallback = float(summary.get("llm_fallback_segments", 0.0) or 0.0)
    fallback_rate = (llm_fallback / llm_total) if llm_total > 0 else 0.0
    overall = max(
        0.0,
        min(1.0, (quality * 0.55) + (grounding * 0.25) + ((1.0 - fallback_rate) * 0.2)),
    )
    return {
        "overall": round(overall, 6),
        "narrative_quality_mean": round(quality, 6),
        "memory_grounding_mean": round(grounding, 6),
        "llm_fallback_rate": round(fallback_rate, 6),
    }


def _evaluate_stability_v1(summary: Dict[str, Any]) -> Dict[str, float]:
    rem_fraction = float(summary.get("rem_fraction", 0.0) or 0.0)
    biz = float(summary.get("mean_bizarreness", 0.0) or 0.0)
    lucid = float(summary.get("lucid_event_count", 0.0) or 0.0)
    rem_score = max(0.0, 1.0 - min(1.0, abs(rem_fraction - 0.22) / 0.22))
    biz_score = max(0.0, 1.0 - min(1.0, abs(biz - 0.52) / 0.52))
    lucid_score = max(0.0, min(1.0, lucid / 12.0))
    overall = max(
        0.0, min(1.0, (rem_score * 0.4) + (biz_score * 0.4) + (lucid_score * 0.2))
    )
    return {
        "overall": round(overall, 6),
        "rem_score": round(rem_score, 6),
        "bizarreness_score": round(biz_score, 6),
        "lucidity_score": round(lucid_score, 6),
    }


_EVALUATOR_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, float]]] = {
    "quality-v1": _evaluate_quality_v1,
    "stability-v1": _evaluate_stability_v1,
}


async def _generate_llm_narrative(
    seg: dict,
    config: SimulationConfig,
    client: LLMClient,
) -> tuple[str, str]:
    """
    Call the LLM to generate a dream narrative and scene description
    for a single segment.
    """
    events_str = (
        "\n".join(f"- {e}" for e in config.prior_day_events)
        if config.prior_day_events
        else "No specific prior-day events provided."
    )

    system_prompt = (
        "You are the PhenomenologyReporter agent inside DreamForge AI, a computational "
        "dream simulation system. Your role is to translate biophysical sleep data into "
        "vivid, scientifically grounded first-person dream narratives.\n\n"
        "Guidelines:\n"
        "- Write in present tense, first person ('I find myself...')\n"
        "- Incorporate the sleep stage characteristics (REM=vivid/bizarre, N3=deep/fragmentary)\n"
        "- Reflect the emotional tone and bizarreness score in your prose\n"
        "- Weave in prior-day events as transformed, symbolic memory fragments\n"
        "- Keep narrative under 120 words, scene description under 60 words\n"
        '- Output ONLY valid JSON: {"narrative": "...", "scene": "..."}'
    )

    user_prompt = (
        f"/no_think\n\nSleep stage: {seg['stage']}\n"
        f"Dominant emotion: {seg['dominant_emotion']}\n"
        f"Bizarreness score: {seg['bizarreness_score']:.2f} (0=mundane, 1=extremely bizarre)\n"
        f"Lucidity probability: {seg['lucidity_probability']:.2f}\n"
        f"Neurochemistry snapshot:\n"
        f"  ACh={seg['neurochemistry']['ach']:.2f}  "
        f"5-HT={seg['neurochemistry']['serotonin']:.2f}  "
        f"NE={seg['neurochemistry']['ne']:.2f}  "
        f"Cortisol={seg['neurochemistry']['cortisol']:.2f}\n"
        f"Prior-day events:\n{events_str}\n\n"
        f"Generate the dream narrative and scene description as JSON."
    )

    raw = await client.chat(system=system_prompt, user=user_prompt)
    parsed = parse_narrative_response(raw)
    narrative = parsed.get("narrative", "")
    scene = parsed.get("scene_description", "")
    if not narrative:
        narrative = _strip_thinking_tags(raw)
    if not scene:
        scene = f"A {seg['dominant_emotion']} dreamscape at {seg['stage']} depth."

    return narrative, scene


def _stage_minutes(segments: List[dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for seg in segments:
        stage = str(seg.get("stage") or "N2")
        start = float(seg.get("start_time_hours", seg.get("time_hours", 0.0)) or 0.0)
        end = float(seg.get("end_time_hours", start) or start)
        totals[stage] += max(0.0, (end - start) * 60.0)
    return {k: round(v, 3) for k, v in totals.items()}


def _derive_carryover_events_from_simulation(
    simulation_payload: dict[str, Any], top_k: int = 5
) -> List[str]:
    segments = simulation_payload.get("segments", [])
    if not isinstance(segments, list):
        return []
    id_counts: Counter[str] = Counter()
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for mem_id in seg.get("active_memory_ids", []):
            mem_id_str = str(mem_id).strip()
            if mem_id_str:
                id_counts[mem_id_str] += 1
    if not id_counts:
        return []
    label_map: Dict[str, str] = {}
    memory_graph = simulation_payload.get("memory_graph", {})
    if isinstance(memory_graph, dict):
        nodes = memory_graph.get("nodes", [])
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id", "")).strip()
                label = str(node.get("label", node_id)).strip()
                if node_id and label:
                    label_map[node_id] = label
    events: List[str] = []
    for mem_id, _count in id_counts.most_common(max(1, int(top_k))):
        label = label_map.get(mem_id)
        if not label:
            label = mem_id.replace("mem::", "").replace("_", " ").strip()
        if label:
            events.append(label)
    return events


def _build_multi_night_continuity_payload(
    night_payloads: List[dict[str, Any]],
) -> dict[str, Any]:
    usage: Dict[str, set[int]] = defaultdict(set)
    night_count = len(night_payloads)
    for night_idx, payload in enumerate(night_payloads):
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            continue
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            for mem_id in seg.get("active_memory_ids", []):
                mem_id_str = str(mem_id).strip()
                if mem_id_str:
                    usage[mem_id_str].add(night_idx)

    recurring = {
        mem_id: sorted(int(idx) for idx in night_set)
        for mem_id, night_set in usage.items()
        if len(night_set) >= 2
    }
    recurring_sorted = sorted(
        recurring.items(),
        key=lambda item: (len(item[1]), item[0]),
        reverse=True,
    )
    top_recurring = [mem_id for mem_id, _ in recurring_sorted[:10]]

    flows: Dict[tuple[int, int], int] = {}
    for _mem_id, night_seq in recurring_sorted:
        for i in range(len(night_seq) - 1):
            pair = (night_seq[i], night_seq[i + 1])
            flows[pair] = flows.get(pair, 0) + 1
    source: List[int] = []
    target: List[int] = []
    value: List[int] = []
    for (src_idx, dst_idx), cnt in sorted(flows.items()):
        source.append(int(src_idx))
        target.append(int(dst_idx))
        value.append(int(cnt))

    rem_fractions: List[float] = []
    quality_means: List[float] = []
    fallback_rates: List[float] = []
    for payload in night_payloads:
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            continue
        rem_fractions.append(float(summary.get("rem_fraction", 0.0)))
        quality_means.append(float(summary.get("narrative_quality_mean", 0.0)))
        fallback_segments = float(summary.get("llm_fallback_segments", 0.0))
        llm_total = float(
            summary.get("llm_total_invocations", summary.get("llm_calls_total", 0.0))
        )
        fallback_rates.append((fallback_segments / llm_total) if llm_total > 0 else 0.0)

    divisor = float(len(night_payloads)) if night_payloads else 1.0
    return {
        "recurring_memory_count": int(len(recurring)),
        "top_recurring_memory_ids": top_recurring,
        "recurring_memory_night_map": recurring,
        "sankey": {
            "nodes": {"labels": [f"Night {i + 1}" for i in range(night_count)]},
            "links": {"source": source, "target": target, "value": value},
        },
        "night_averages": {
            "rem_fraction_mean": round(sum(rem_fractions) / divisor, 4),
            "narrative_quality_mean": round(sum(quality_means) / divisor, 4),
            "llm_fallback_rate_mean": round(sum(fallback_rates) / divisor, 4),
        },
    }


def _build_comparison_payload(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict:
    b_summary = baseline.get("summary", {})
    c_summary = candidate.get("summary", {})
    b_segments = baseline.get("segments", [])
    c_segments = candidate.get("segments", [])
    delta_mean_bizarreness = round(
        float(c_summary.get("mean_bizarreness", 0.0))
        - float(b_summary.get("mean_bizarreness", 0.0)),
        4,
    )
    delta_rem_fraction = round(
        float(c_summary.get("rem_fraction", 0.0))
        - float(b_summary.get("rem_fraction", 0.0)),
        4,
    )
    delta_lucid_event_count = int(c_summary.get("lucid_event_count", 0)) - int(
        b_summary.get("lucid_event_count", 0)
    )
    delta_narrative_quality = round(
        float(c_summary.get("narrative_quality_mean", 0.0))
        - float(b_summary.get("narrative_quality_mean", 0.0)),
        4,
    )
    delta_memory_grounding = round(
        float(c_summary.get("narrative_memory_grounding_mean", 0.0))
        - float(b_summary.get("narrative_memory_grounding_mean", 0.0)),
        4,
    )
    baseline_llm_fallback_segments = int(b_summary.get("llm_fallback_segments", 0))
    candidate_llm_fallback_segments = int(c_summary.get("llm_fallback_segments", 0))
    baseline_llm_total = int(
        b_summary.get("llm_total_invocations", b_summary.get("llm_calls_total", 0))
    )
    candidate_llm_total = int(
        c_summary.get("llm_total_invocations", c_summary.get("llm_calls_total", 0))
    )
    baseline_fallback_rate = (
        float(baseline_llm_fallback_segments) / float(baseline_llm_total)
        if baseline_llm_total > 0
        else 0.0
    )
    candidate_fallback_rate = (
        float(candidate_llm_fallback_segments) / float(candidate_llm_total)
        if candidate_llm_total > 0
        else 0.0
    )
    delta_llm_fallback_rate = round(candidate_fallback_rate - baseline_fallback_rate, 4)
    segment_count = max(
        1,
        min(
            len(b_segments) if isinstance(b_segments, list) else 0,
            len(c_segments) if isinstance(c_segments, list) else 0,
        ),
    )
    sample_confidence = round(min(1.0, (float(segment_count) / 240.0) ** 0.5), 4)
    anomaly_flags: List[str] = []
    if abs(delta_mean_bizarreness) >= 0.2:
        anomaly_flags.append("bizarreness_shift")
    if abs(delta_rem_fraction) >= 0.08:
        anomaly_flags.append("rem_fraction_shift")
    if abs(delta_narrative_quality) >= 0.15:
        anomaly_flags.append("narrative_quality_shift")
    if abs(delta_lucid_event_count) >= 3:
        anomaly_flags.append("lucid_event_spike")
    if delta_llm_fallback_rate >= 0.20:
        anomaly_flags.append("llm_fallback_spike")
    if delta_memory_grounding <= -0.10:
        anomaly_flags.append("memory_grounding_drop")
    return {
        "baseline_id": baseline.get("id"),
        "candidate_id": candidate.get("id"),
        "delta": {
            "mean_bizarreness": delta_mean_bizarreness,
            "rem_fraction": delta_rem_fraction,
            "lucid_event_count": delta_lucid_event_count,
            "narrative_quality_mean": delta_narrative_quality,
            "narrative_memory_grounding_mean": delta_memory_grounding,
            "llm_fallback_rate": delta_llm_fallback_rate,
        },
        "confidence": {
            "sample_size": segment_count,
            "metric_confidence": {
                "mean_bizarreness": sample_confidence,
                "rem_fraction": sample_confidence,
                "lucid_event_count": sample_confidence,
                "narrative_quality_mean": sample_confidence,
                "narrative_memory_grounding_mean": sample_confidence,
                "llm_fallback_rate": sample_confidence,
            },
        },
        "stage_minutes": {
            "baseline": _stage_minutes(
                b_segments if isinstance(b_segments, list) else []
            ),
            "candidate": _stage_minutes(
                c_segments if isinstance(c_segments, list) else []
            ),
        },
        "event_markers": {
            "baseline_lucid_events": int(b_summary.get("lucid_event_count", 0)),
            "candidate_lucid_events": int(c_summary.get("lucid_event_count", 0)),
            "baseline_llm_fallback_segments": baseline_llm_fallback_segments,
            "candidate_llm_fallback_segments": candidate_llm_fallback_segments,
            "baseline_llm_fallback_rate": round(baseline_fallback_rate, 4),
            "candidate_llm_fallback_rate": round(candidate_fallback_rate, 4),
        },
        "methodology": {
            "delta_formula": "candidate - baseline",
            "metrics": {
                "mean_bizarreness": "summary.mean_bizarreness",
                "rem_fraction": "summary.rem_fraction",
                "lucid_event_count": "summary.lucid_event_count",
                "narrative_quality_mean": "summary.narrative_quality_mean",
                "narrative_memory_grounding_mean": "summary.narrative_memory_grounding_mean",
                "llm_fallback_rate": "llm_fallback_segments / llm_total_invocations",
            },
            "thresholds": {
                "bizarreness_shift_abs": 0.2,
                "rem_fraction_shift_abs": 0.08,
                "narrative_quality_shift_abs": 0.15,
                "lucid_event_spike_abs": 3,
                "llm_fallback_spike_delta_min": 0.20,
                "memory_grounding_drop_delta_max": -0.10,
            },
            "confidence_proxy": "sqrt(sample_size / 240), capped at 1.0",
        },
        "anomaly_flags": anomaly_flags,
        "generated_at_unix": time.time(),
    }


def _estimate_no_llm_duration_seconds(
    config: SimulationConfig, no_llm_history: Optional[List[float]] = None
) -> float:
    history = list(no_llm_history or [])
    if history:
        p70 = float(np.percentile(history, 70))
        return max(3.0, min(15.0, p70))
    dt = max(0.1, float(config.dt_minutes))
    ticks = max(1.0, (float(config.duration_hours) * 60.0) / dt)
    physics_seconds = max(2.0, ticks * 0.004)
    return max(4.0, min(12.0, physics_seconds + 2.0))


def _estimate_llm_invocation_density(
    density_history: Optional[List[float]] = None,
) -> float:
    history = list(density_history or [])
    if history:
        return max(0.05, min(0.9, float(np.percentile(history, 60))))
    return 0.28


def _estimate_llm_invocation_seconds(
    timeout_seconds: float, latency_history: Optional[List[float]] = None
) -> float:
    history = list(latency_history or [])
    if history:
        return max(0.8, float(np.percentile(history, 70)))
    return max(2.0, min(18.0, timeout_seconds * 0.45))


def _estimate_job_duration_seconds(
    config: SimulationConfig,
    *,
    expected_llm_invocations: Optional[int] = None,
    llm_concurrency: Optional[int] = None,
    llm_timeout_seconds: Optional[float] = None,
) -> float:
    with _metrics_lock:
        llm_history = list(_simulation_duration_seconds_llm_history)
        no_llm_history = list(_simulation_duration_seconds_no_llm_history)
        global_history = list(_simulation_duration_seconds_history)
        llm_density_history = list(_llm_invocation_density_history)
        llm_latency_history = list(_llm_invocation_latency_seconds_history)

    baseline_no_llm = _estimate_no_llm_duration_seconds(config, no_llm_history)
    if not bool(config.use_llm):
        return baseline_no_llm

    cfg = NarrativeGeneratorConfig.from_settings()
    concurrency = (
        max(1, int(llm_concurrency))
        if llm_concurrency is not None
        else max(1, int(cfg.concurrency))
    )
    timeout_budget = (
        max(1.0, float(llm_timeout_seconds))
        if llm_timeout_seconds is not None
        else max(5.0, float(cfg.timeout_seconds))
    )
    dt = max(0.1, float(config.dt_minutes))
    ticks = max(1.0, (float(config.duration_hours) * 60.0) / dt)
    invocation_density = _estimate_llm_invocation_density(llm_density_history)
    planned_invocations = (
        max(0, int(expected_llm_invocations))
        if expected_llm_invocations is not None
        else max(1, int(round(ticks * invocation_density)))
    )
    llm_seconds_per_invocation = _estimate_llm_invocation_seconds(
        timeout_budget, llm_latency_history
    )
    llm_wall_seconds = (
        float(planned_invocations) * llm_seconds_per_invocation / float(concurrency)
    )
    model_estimate = baseline_no_llm + llm_wall_seconds + 4.0

    if llm_history:
        llm_reference = float(np.percentile(llm_history, 65))
        return max(20.0, (model_estimate * 0.7) + (llm_reference * 0.3))
    if global_history:
        global_reference = float(np.percentile(global_history, 70))
        return max(20.0, (model_estimate * 0.85) + (global_reference * 0.15))
    return max(20.0, model_estimate)


def _emit_simulation_progress_event(event: Dict[str, Any]) -> None:
    progress_callback = _simulation_progress_callback_var.get()
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception as exc:
        logger.debug("Progress callback error ignored: %s", exc)


def _job_progress_snapshot(
    job: dict[str, Any], status_value: str
) -> tuple[float, Optional[int]]:
    status_norm = str(status_value).lower()
    if status_norm == "completed":
        return 100.0, 0
    if status_norm in {"failed", "cancelled"}:
        return 100.0, None

    now_ts = time.time()
    started_at = float(job.get("started_at") or now_ts)
    elapsed = max(0.0, now_ts - started_at)
    estimated = float(job.get("estimated_duration_seconds") or 0.0)
    explicit_progress = job.get("progress_percent")
    explicit_eta = job.get("eta_seconds")
    phase = str(job.get("phase") or "").lower()
    use_llm = bool(job.get("use_llm", False))

    if explicit_progress is not None:
        progress = max(0.0, min(99.0, float(explicit_progress)))
        if status_norm == "cancelling":
            progress = min(99.0, max(95.0, progress))
        if status_norm in {"running", "cancelling"}:
            if estimated <= 0.0:
                estimated = max(
                    60.0,
                    (
                        elapsed + float(max(0, int(explicit_eta)))
                        if explicit_eta is not None
                        else elapsed
                    ),
                )
            if progress >= 0.5:
                effective_progress = progress
                if phase == "physics":
                    effective_progress = max(effective_progress, 10.0)
                elif phase == "narrative":
                    effective_progress = max(effective_progress, 22.0)
                projected_total = elapsed / max(effective_progress / 100.0, 0.005)
                risk_multiplier = (
                    1.15
                    if use_llm and phase in {"physics", "narrative"}
                    else (1.08 if use_llm else 1.05)
                )
                estimated = max(estimated, projected_total * risk_multiplier)
            if explicit_eta is not None:
                estimated = max(estimated, elapsed + float(max(0, int(explicit_eta))))
            last_event_ts = float(job.get("last_progress_event_at") or started_at)
            stale_for = max(0.0, now_ts - last_event_ts)
            llm_expected = 0
            llm_completed = 0
            llm_remaining = 0
            llm_avg_seconds = 0.0
            llm_concurrency = 1
            if use_llm:
                try:
                    llm_expected = max(0, int(job.get("llm_expected_invocations") or 0))
                except (TypeError, ValueError):
                    llm_expected = 0
                try:
                    llm_completed = max(
                        0, int(job.get("llm_completed_invocations") or 0)
                    )
                except (TypeError, ValueError):
                    llm_completed = 0
                try:
                    llm_remaining = max(
                        0,
                        int(
                            job.get(
                                "llm_remaining_invocations",
                                max(0, llm_expected - llm_completed),
                            )
                        ),
                    )
                except (TypeError, ValueError):
                    llm_remaining = max(0, llm_expected - llm_completed)
                try:
                    llm_avg_seconds = max(
                        0.0, float(job.get("llm_avg_invocation_seconds") or 0.0)
                    )
                except (TypeError, ValueError):
                    llm_avg_seconds = 0.0
                try:
                    llm_concurrency = max(1, int(job.get("llm_concurrency") or 1))
                except (TypeError, ValueError):
                    llm_concurrency = 1

            if (
                use_llm
                and phase == "narrative"
                and llm_expected > 0
                and llm_remaining > 0
                and llm_avg_seconds > 0.0
            ):
                llm_eta_projection = (
                    float(llm_remaining) * llm_avg_seconds / float(llm_concurrency)
                ) + 3.0
                estimated = max(estimated, elapsed + llm_eta_projection)
            if stale_for >= 10.0 and progress < 99.0:
                stale_penalty = stale_for * (0.8 if use_llm else 0.4)
                if use_llm and llm_expected > 0 and llm_remaining > 0:
                    llm_floor = (
                        float(llm_remaining) * max(1.0, llm_avg_seconds)
                    ) / float(max(1, llm_concurrency))
                    stale_penalty = max(stale_penalty, llm_floor + (stale_for * 0.35))
                estimated = max(estimated, elapsed + stale_penalty)

            eta = max(0, int(round(max(0.0, estimated - elapsed))))
            if progress >= 95.0 and eta > 15:
                progress = min(progress, 94.5)
            if estimated > 0.0 and progress > 0.0:
                time_progress = min(99.0, (elapsed / estimated) * 100.0)
                progress = min(progress, max(time_progress, progress - 4.0))
            return round(progress, 2), eta

        eta = int(explicit_eta) if explicit_eta is not None else None
        return round(progress, 2), eta

    if estimated <= 0.0:
        estimated = 60.0
    if status_norm == "pending":
        eta = max(0, int(round(estimated)))
        return 0.0, eta

    progress = min(95.0, (elapsed / estimated) * 100.0)
    eta = max(0, int(round(estimated - elapsed)))
    if status_norm == "cancelling":
        progress = min(99.0, max(progress, 95.0))
    return round(progress, 2), eta


def _job_provenance_snapshot(
    job: dict[str, Any], status_value: str
) -> tuple[str, str, Dict[str, Any]]:
    now_ts = time.time()
    status_norm = str(status_value).lower()
    phase = str(job.get("phase") or status_norm)
    last_progress_event_at = job.get("last_progress_event_at")
    event_age_seconds: Optional[float] = None
    if last_progress_event_at is not None:
        event_age_seconds = max(0.0, now_ts - float(last_progress_event_at))
    progress_source = str(job.get("progress_source", "")).strip()
    if not progress_source:
        progress_source = (
            "event_stream" if last_progress_event_at is not None else "time_projection"
        )
    eta_source = str(job.get("eta_source", "")).strip()
    if not eta_source:
        eta_source = (
            "event_projection"
            if job.get("eta_seconds") is not None
            else "historical_projection"
        )
    provenance = {
        "phase": phase,
        "use_llm": bool(job.get("use_llm", False)),
        "event_age_seconds": (
            round(event_age_seconds, 3) if event_age_seconds is not None else None
        ),
        "has_explicit_eta": job.get("eta_seconds") is not None,
        "estimated_duration_seconds": (
            float(job["estimated_duration_seconds"])
            if job.get("estimated_duration_seconds") is not None
            else None
        ),
        "llm_expected_invocations": (
            int(job["llm_expected_invocations"])
            if job.get("llm_expected_invocations") is not None
            else None
        ),
        "llm_completed_invocations": (
            int(job["llm_completed_invocations"])
            if job.get("llm_completed_invocations") is not None
            else None
        ),
        "llm_remaining_invocations": (
            int(job["llm_remaining_invocations"])
            if job.get("llm_remaining_invocations") is not None
            else None
        ),
        "llm_avg_invocation_seconds": (
            float(job["llm_avg_invocation_seconds"])
            if job.get("llm_avg_invocation_seconds") is not None
            else None
        ),
        "llm_concurrency": (
            int(job["llm_concurrency"])
            if job.get("llm_concurrency") is not None
            else None
        ),
    }
    return progress_source, eta_source, provenance


async def _run_simulation_job(job_id: str, config: SimulationConfig) -> None:
    last_persist_ts = 0.0
    last_phase = ""
    last_progress = -1.0

    def _on_progress(event: Dict[str, Any]) -> None:
        nonlocal last_persist_ts, last_phase, last_progress
        now_ts = time.time()
        progress = max(0.0, min(99.0, float(event.get("progress_percent", 0.0))))
        phase = str(event.get("phase") or "running")
        progress_source_event = str(event.get("progress_source") or "").strip()
        eta_source_event = str(event.get("eta_source") or "").strip()
        eta_seconds_raw = event.get("eta_seconds")
        eta_margin_raw = event.get("eta_margin_seconds")
        should_persist = (
            abs(progress - last_progress) >= 1.0
            or phase != last_phase
            or (now_ts - last_persist_ts) >= 0.5
        )
        with _jobs_lock:
            job = _simulation_jobs.get(job_id)
            if not job:
                return
            job["progress_percent"] = round(progress, 2)
            job["phase"] = phase
            job["progress_source"] = progress_source_event or "event_stream"
            job["last_progress_event_at"] = now_ts
            if eta_seconds_raw is not None:
                eta_value = max(0, int(eta_seconds_raw))
                job["eta_seconds"] = eta_value
                job["eta_source"] = eta_source_event or "event_projection"
                started_at = float(job.get("started_at") or now_ts)
                elapsed = max(0.0, now_ts - started_at)
                projected_total = max(elapsed + 1.0, elapsed + float(eta_value))
                existing_est = float(job.get("estimated_duration_seconds") or 0.0)
                if existing_est > 0.0:
                    projected_total = (existing_est * 0.65) + (projected_total * 0.35)
                job["estimated_duration_seconds"] = round(
                    projected_total,
                    2,
                )
            if eta_margin_raw is not None:
                job["eta_margin_seconds"] = max(0, int(eta_margin_raw))
            if event.get("estimated_duration_seconds") is not None:
                target_est = max(1.0, float(event["estimated_duration_seconds"]))
                existing_est = float(job.get("estimated_duration_seconds") or 0.0)
                if existing_est > 0.0:
                    target_est = (existing_est * 0.7) + (target_est * 0.3)
                job["estimated_duration_seconds"] = round(target_est, 2)
            elif eta_seconds_raw is None and not str(job.get("eta_source", "")).strip():
                job["eta_source"] = "historical_projection"
            if event.get("llm_expected_invocations") is not None:
                job["llm_expected_invocations"] = max(
                    0, int(event.get("llm_expected_invocations"))
                )
            if event.get("llm_completed_invocations") is not None:
                job["llm_completed_invocations"] = max(
                    0, int(event.get("llm_completed_invocations"))
                )
            if event.get("llm_remaining_invocations") is not None:
                job["llm_remaining_invocations"] = max(
                    0, int(event.get("llm_remaining_invocations"))
                )
            if event.get("llm_avg_invocation_seconds") is not None:
                job["llm_avg_invocation_seconds"] = max(
                    0.0, float(event.get("llm_avg_invocation_seconds"))
                )
            if event.get("llm_concurrency") is not None:
                job["llm_concurrency"] = max(1, int(event.get("llm_concurrency")))
            if should_persist:
                _persist_job(job_id, dict(job))
                last_persist_ts = now_ts
                last_phase = phase
                last_progress = progress

    with _jobs_lock:
        job = _simulation_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = time.time()
        job["last_progress_event_at"] = job["started_at"]
        job["phase"] = "physics"
        job["progress_percent"] = 1.0
        job["progress_source"] = "event_stream"
        job["eta_source"] = "historical_projection"
        _persist_job(job_id, dict(job))
    token = _simulation_progress_callback_var.set(_on_progress)
    try:
        result = await simulate_night(config)
        payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        with _jobs_lock:
            job = _simulation_jobs.get(job_id)
            if not job:
                return
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["simulation_id"] = payload.get("id")
            job["result"] = payload
            job["progress_percent"] = 100.0
            job["phase"] = "complete"
            job["progress_source"] = "terminal"
            job["eta_seconds"] = 0
            job["eta_source"] = "terminal"
            job["eta_margin_seconds"] = 0
            _persist_job(job_id, dict(job))
    except asyncio.CancelledError:
        logger.info("Async simulation job cancelled: %s", job_id)
        with _jobs_lock:
            job = _simulation_jobs.get(job_id)
            if job:
                job["status"] = "cancelled"
                job["completed_at"] = time.time()
                job["error_code"] = "cancelled"
                job["error_message"] = "Simulation cancelled by user."
                job["phase"] = "cancelled"
                job["progress_percent"] = float(job.get("progress_percent") or 0.0)
                job["progress_source"] = "terminal"
                job["eta_source"] = "terminal"
                _persist_job(job_id, dict(job))
        _audit("simulation_job_cancelled", job_id=job_id)
        raise
    except Exception as exc:
        logger.exception("Async simulation job failed: %s", exc)
        _record_simulation_failure("internal_error")
        with _jobs_lock:
            job = _simulation_jobs.get(job_id)
            if not job:
                return
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error_code"] = "internal_error"
            job["error_message"] = str(exc)
            job["phase"] = "failed"
            job["progress_percent"] = float(job.get("progress_percent") or 0.0)
            job["progress_source"] = "terminal"
            job["eta_source"] = "terminal"
            _persist_job(job_id, dict(job))
    finally:
        _simulation_progress_callback_var.reset(token)
        with _jobs_lock:
            _simulation_job_tasks.pop(job_id, None)


def _prometheus_metrics_text(metrics: dict[str, Any]) -> str:
    lines = [
        "# HELP dreamforge_simulation_requests_total Total simulation requests",
        "# TYPE dreamforge_simulation_requests_total counter",
        f"dreamforge_simulation_requests_total {metrics.get('simulation_requests_total', 0.0)}",
        "# HELP dreamforge_simulation_completed_total Total completed simulations",
        "# TYPE dreamforge_simulation_completed_total counter",
        f"dreamforge_simulation_completed_total {metrics.get('simulation_completed_total', 0.0)}",
        "# HELP dreamforge_simulation_failed_total Total failed simulations",
        "# TYPE dreamforge_simulation_failed_total counter",
        f"dreamforge_simulation_failed_total {metrics.get('simulation_failed_total', 0.0)}",
        "# HELP dreamforge_simulation_duration_seconds_avg Average simulation duration",
        "# TYPE dreamforge_simulation_duration_seconds_avg gauge",
        f"dreamforge_simulation_duration_seconds_avg {metrics.get('simulation_duration_seconds_avg', 0.0)}",
        "# HELP dreamforge_simulation_duration_seconds_p95 p95 simulation duration",
        "# TYPE dreamforge_simulation_duration_seconds_p95 gauge",
        f"dreamforge_simulation_duration_seconds_p95 {metrics.get('simulation_duration_seconds_p95', 0.0)}",
        "# HELP dreamforge_simulation_duration_seconds_p99 p99 simulation duration",
        "# TYPE dreamforge_simulation_duration_seconds_p99 gauge",
        f"dreamforge_simulation_duration_seconds_p99 {metrics.get('simulation_duration_seconds_p99', 0.0)}",
        "# HELP dreamforge_llm_fallback_rate LLM fallback ratio",
        "# TYPE dreamforge_llm_fallback_rate gauge",
        f"dreamforge_llm_fallback_rate {metrics.get('llm_fallback_rate', 0.0)}",
        "# HELP dreamforge_api_success_rate API success ratio",
        "# TYPE dreamforge_api_success_rate gauge",
        f"dreamforge_api_success_rate {metrics.get('api_success_rate', 0.0)}",
        "# HELP dreamforge_job_queue_pending Pending async jobs",
        "# TYPE dreamforge_job_queue_pending gauge",
        f"dreamforge_job_queue_pending {metrics.get('job_queue_pending', 0.0)}",
        "# HELP dreamforge_job_queue_running Running async jobs",
        "# TYPE dreamforge_job_queue_running gauge",
        f"dreamforge_job_queue_running {metrics.get('job_queue_running', 0.0)}",
    ]
    return "\n".join(lines) + "\n"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", tags=["System"])
async def root():
    """Health ping."""
    return {
        "service": "DreamForge AI",
        "status": "running",
        "version": "0.2.0",
        "api_contract": API_CONTRACT_VERSION,
    }


@app.get("/api/health/llm", response_model=LLMHealthResponse, tags=["LLM"])
@app.get("/api/v1/health/llm", response_model=LLMHealthResponse, tags=["LLM"])
async def llm_health():
    """Check connectivity to the configured LLM backend."""
    client = get_llm_client()
    result = await client.check_health()
    return LLMHealthResponse(
        ok=result.get("ok", False),
        provider=client.config.provider,
        model=client.config.model,
        base_url=client.config.base_url,
        available_models=result.get("models", []),
        error=result.get("error"),
    )


@app.get("/api/llm/health", response_model=LLMHealthResponse, tags=["LLM"])
@app.get("/api/v1/llm/health", response_model=LLMHealthResponse, tags=["LLM"])
async def llm_health_alias():
    return await llm_health()


@app.post("/api/charts/export", tags=["System"])
@app.post("/api/v1/charts/export", tags=["System"])
async def export_chart_image(request: ChartExportRequest):
    image_format = str(request.format).lower()
    if image_format not in {"png", "svg"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="format must be 'png' or 'svg'.",
        )
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Plot export backend unavailable: {exc}",
        ) from exc

    try:
        fig = go.Figure(request.figure)
        export_scale = float(request.scale) if image_format == "png" else 1.0
        image_bytes = pio.to_image(fig, format=image_format, scale=export_scale)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chart export failed: {exc}",
        ) from exc

    media_type = "image/png" if image_format == "png" else "image/svg+xml"
    return Response(content=image_bytes, media_type=media_type)


@app.get("/api/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
@app.get("/api/v1/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
async def get_llm_config():
    """Return the current LLM configuration (API key is masked)."""
    client = get_llm_client()
    cfg = client.config
    return LLMConfigResponse(
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        api_key_set=bool(cfg.api_key and cfg.api_key not in ("", "none")),
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )


@app.get("/api/llm/registry", tags=["LLM"])
@app.get("/api/v1/llm/registry", tags=["LLM"])
async def get_llm_registry():
    client = get_llm_client()
    return get_llm_registry_snapshot(
        active_provider=client.config.provider,
        active_model=client.config.model,
        prompt_profile_version=PROMPT_PROFILE_VERSION,
    )


@app.post("/api/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
@app.post("/api/v1/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
async def update_llm_config(req: LLMConfigRequest):
    """
    Update the LLM configuration at runtime — no restart required.

    Useful for switching between LM Studio, Ollama, and OpenAI without
    changing environment variables.
    """

    # Rebuild config from current + overrides
    client = get_llm_client()
    old = client.config

    new_cfg = LLMConfig(
        provider=req.provider or old.provider,
        base_url=req.base_url or old.base_url,
        model=req.model or old.model,
        api_key=req.api_key or old.api_key,
        max_tokens=req.max_tokens if req.max_tokens is not None else old.max_tokens,
        temperature=req.temperature if req.temperature is not None else old.temperature,
        timeout=req.timeout if req.timeout is not None else old.timeout,
    )

    # Close old client, reinitialise singleton
    await client.aclose()

    # Patch the singleton via the module
    try:
        import core.llm_client as llm_mod

        llm_mod._default_client = LLMClient(config=new_cfg)
        updated_client = llm_mod._default_client
    except Exception:
        # Fallback for stub path
        global _default_client  # type: ignore[name-defined]
        _default_client = LLMClient(config=new_cfg)
        updated_client = _default_client

    logger.info(
        "LLM config updated: provider=%s model=%s base_url=%s",
        new_cfg.provider,
        new_cfg.model,
        new_cfg.base_url,
    )
    _audit(
        "llm_config_updated",
        provider=new_cfg.provider,
        model=new_cfg.model,
        base_url=new_cfg.base_url,
    )

    cfg = updated_client.config
    return LLMConfigResponse(
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        api_key_set=bool(cfg.api_key and cfg.api_key not in ("", "none")),
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )


@app.post(
    "/api/simulation/night",
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Simulation"],
)
@app.post(
    "/api/v1/simulation/night",
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Simulation"],
)
@app.post(
    "/simulate-night",
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Simulation"],
)
async def simulate_night(config: SimulationConfig):
    """
    Run a full-night dream simulation.

    **Pipeline:**
    1. Two-process sleep model (Borbély) determines stage trajectory
    2. ODE-based neurochemistry computed per segment
    3. Bizarreness + lucidity scores derived from neurochemical state
    4. If `use_llm=True`: LLM generates vivid first-person narrative per REM segment
       (or all segments if `llm_segments_only=False`)
    5. Results persisted in memory and returned as structured JSON

    **Tip:** Set `use_llm=false` for a fast (< 1 s) template-only run to verify
    the simulation pipeline, then re-enable for full narrative generation.
    """
    sim_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    _record_simulation_request()
    logger.info(
        "Starting simulation %s (%.1fh, dt=%.1fmin, LLM=%s)",
        sim_id,
        config.duration_hours,
        config.dt_minutes,
        config.use_llm,
    )
    _audit(
        "simulation_started",
        simulation_id=sim_id,
        duration_hours=config.duration_hours,
        dt_minutes=config.dt_minutes,
        llm=config.use_llm,
    )
    initial_estimate_seconds = _estimate_job_duration_seconds(config)
    _emit_simulation_progress_event(
        {
            "phase": "physics",
            "progress_percent": 1.0,
            "eta_seconds": int(round(initial_estimate_seconds)),
            "eta_margin_seconds": int(round(max(3.0, initial_estimate_seconds * 0.2))),
            "estimated_duration_seconds": initial_estimate_seconds,
        }
    )

    # Step 1: biophysical simulation
    raw_segments = _simulate_night_physics(config)
    memory_graph, memory_activations = _build_memory_outputs(
        raw_segments, config.prior_day_events
    )
    lucidity_threshold = LucidityModel.from_settings().threshold
    lucid_events = _annotate_lucid_events(raw_segments, lucidity_threshold)
    _emit_simulation_progress_event(
        {
            "phase": "physics",
            "progress_percent": 20.0,
            "eta_seconds": int(
                round(max(3.0, len(raw_segments) * (0.06 if config.use_llm else 0.002)))
            ),
            "eta_margin_seconds": int(
                round(
                    max(2.0, len(raw_segments) * (0.015 if config.use_llm else 0.001))
                )
            ),
        }
    )

    # Step 2: narrative generation
    client = get_llm_client()
    llm_used = False
    narrative_cache = NarrativeCache(template_bank=_get_template_bank())

    # Determine trigger provenance for all policy-eligible segments.
    llm_jobs: List[tuple[int, str]] = []
    if config.use_llm:
        prev_stage: Optional[str] = None

        def _llm_eligible(seg: dict[str, Any]) -> bool:
            stage_val = str(seg.get("stage") or "N2")
            biz_val = float(seg.get("bizarreness_score") or 0.0)
            return stage_val == "REM" or (
                stage_val in {"N1", "N2", "N3"} and biz_val >= 0.55
            )

        for idx, seg in enumerate(raw_segments):
            stage = str(seg.get("stage") or "N2")
            neuro = seg.get("neurochemistry") or {}
            trigger_type: Optional[str] = None

            if not _llm_eligible(seg):
                prev_stage = stage
                continue

            if prev_stage != "REM" and stage == "REM":
                trigger_type = "REM_ONSET"
            elif (
                stage == "REM" and float(seg.get("lucidity_probability") or 0.0) >= 0.55
            ):
                trigger_type = "LUCIDITY_THRESHOLD"
            elif stage == "REM" and float(seg.get("bizarreness_score") or 0.0) >= 0.80:
                trigger_type = "BIZARRENESS_PEAK"
            elif stage == "REM":
                trigger_type = "REM_CONTINUATION"
            elif prev_stage != "N3" and stage == "N3":
                trigger_type = "N3_ONSET"
            elif float(neuro.get("ach") or 0.0) >= 0.75:
                trigger_type = "MEMORY_SALIENCE"
            else:
                trigger_type = "NREM_BIZARRENESS"

            llm_jobs.append((idx, trigger_type))
            prev_stage = stage

    trigger_by_idx = {idx: trig for idx, trig in llm_jobs}
    if config.use_llm:
        label_map = {
            str(node.get("id")): str(node.get("label", node.get("id")))
            for node in memory_graph.get("nodes", [])
            if isinstance(node, dict)
        }
        generator = NarrativeGenerator(
            llm_client=client,
            memory_labeler=lambda node_id: label_map.get(str(node_id), str(node_id)),
            style_preset=config.style_preset,
            prompt_profile=config.prompt_profile,
        )
        elapsed_before_narrative = max(0.0, time.perf_counter() - started_at)
        eligible_total_est = max(1, len(llm_jobs))
        narrative_concurrency = max(1, int(generator.config.concurrency))
        timeout_budget_seconds = max(5.0, float(generator.config.timeout_seconds))
        with _metrics_lock:
            llm_latency_history = list(_llm_invocation_latency_seconds_history)
        historical_llm_avg_seconds = _estimate_llm_invocation_seconds(
            timeout_budget_seconds, llm_latency_history
        )
        initial_narrative_eta_seconds = max(
            3.0,
            (
                (float(eligible_total_est) / float(narrative_concurrency))
                * historical_llm_avg_seconds
            )
            + 4.0,
        )
        last_narrative_progress = 20.0
        _emit_simulation_progress_event(
            {
                "phase": "narrative",
                "progress_percent": 20.0,
                "progress_source": "llm_invocation_progress",
                "eta_seconds": int(round(initial_narrative_eta_seconds)),
                "eta_source": "llm_invocation_projection",
                "eta_margin_seconds": int(
                    round(max(3.0, initial_narrative_eta_seconds * 0.3))
                ),
                "llm_expected_invocations": eligible_total_est,
                "llm_completed_invocations": 0,
                "llm_remaining_invocations": eligible_total_est,
                "llm_avg_invocation_seconds": round(historical_llm_avg_seconds, 3),
                "llm_concurrency": narrative_concurrency,
                "estimated_duration_seconds": elapsed_before_narrative
                + initial_narrative_eta_seconds,
            }
        )

        def _on_narrative_progress(event: Dict[str, Any]) -> None:
            nonlocal last_narrative_progress
            completed = int(event.get("completed", 0))
            total = max(1, int(event.get("total", 1)))
            eligible_completed = int(event.get("eligible_completed", 0))
            eligible_total = int(event.get("eligible_total", 0))
            elapsed = max(0.0, float(event.get("batch_elapsed_seconds", 0.0)))
            elapsed_total = max(0.0, time.perf_counter() - started_at)
            llm_completed = int(
                event.get("llm_completed_invocations", eligible_completed)
            )
            llm_total = max(
                0,
                int(event.get("llm_total_invocations", max(0, eligible_total))),
            )
            llm_remaining = max(
                0,
                int(
                    event.get(
                        "llm_remaining_invocations", max(0, llm_total - llm_completed)
                    )
                ),
            )
            observed_avg_seconds_raw = event.get("llm_avg_invocation_seconds")
            observed_avg_seconds = (
                max(0.0, float(observed_avg_seconds_raw))
                if observed_avg_seconds_raw is not None
                else 0.0
            )
            if observed_avg_seconds <= 0.0 and llm_completed > 0:
                observed_avg_seconds = elapsed / float(max(1, llm_completed))

            if llm_total > 0:
                ratio = min(1.0, float(llm_completed) / float(llm_total))
            else:
                ratio = min(1.0, float(completed) / float(max(1, total)))

            blended_avg_seconds = historical_llm_avg_seconds
            if observed_avg_seconds > 0.0:
                sample_weight = 0.8 if llm_completed >= 3 else 0.6
                blended_avg_seconds = (sample_weight * observed_avg_seconds) + (
                    (1.0 - sample_weight) * historical_llm_avg_seconds
                )

            ratio_progress = 20.0 + (65.0 * ratio)
            eta_core = (
                float(llm_remaining)
                * blended_avg_seconds
                / float(narrative_concurrency)
            )
            if llm_completed <= 0 and llm_total > 0:
                cold_start_floor = (
                    float(llm_remaining)
                    / float(narrative_concurrency)
                    * max(1.5, timeout_budget_seconds * 0.35)
                )
                eta_core = max(eta_core, cold_start_floor)
            eta_post = 4.0
            eta_seconds = int(round(max(0.0, eta_core + eta_post)))
            total_estimate = max(
                elapsed_total + float(eta_seconds), elapsed_total + 1.0
            )
            time_progress = min(85.0, (elapsed_total / total_estimate) * 100.0)
            progress_percent = min(
                85.0,
                max(last_narrative_progress, min(ratio_progress, time_progress + 6.0)),
            )
            last_narrative_progress = progress_percent
            eta_margin = int(round(max(2.0, eta_seconds * 0.25)))
            _emit_simulation_progress_event(
                {
                    "phase": "narrative",
                    "progress_percent": progress_percent,
                    "progress_source": "llm_invocation_progress",
                    "eta_seconds": eta_seconds,
                    "eta_source": "llm_invocation_projection",
                    "eta_margin_seconds": eta_margin,
                    "llm_expected_invocations": llm_total,
                    "llm_completed_invocations": llm_completed,
                    "llm_remaining_invocations": llm_remaining,
                    "llm_avg_invocation_seconds": round(blended_avg_seconds, 3),
                    "llm_concurrency": narrative_concurrency,
                    "estimated_duration_seconds": elapsed_total + float(eta_seconds),
                }
            )

        await generator.generate_batch(
            raw_segments, progress_callback=_on_narrative_progress
        )
    _emit_simulation_progress_event(
        {
            "phase": "postprocess",
            "progress_percent": 85.0,
            "eta_seconds": 4,
            "eta_margin_seconds": 2,
        }
    )

    # Fill remaining segments with templates
    for i, seg in enumerate(raw_segments):
        llm_invoked = bool(seg.pop("_llm_invoked", False))
        llm_fallback = bool(seg.pop("_llm_fallback", False))
        llm_fallback_reason = seg.pop("_llm_fallback_reason", None)
        llm_latency_ms = seg.pop("_llm_latency_ms", None)
        if llm_invoked:
            llm_used = True
            llm_trigger = trigger_by_idx.get(i)
            if llm_trigger is None:
                llm_trigger = (
                    "REM_POLICY"
                    if str(seg.get("stage") or "N2") == "REM"
                    else "NREM_BIZARRENESS"
                )
            seg["llm_latency_ms"] = llm_latency_ms
            seg["llm_trigger_type"] = llm_trigger
            seg["llm_fallback_reason"] = (
                str(llm_fallback_reason) if llm_fallback_reason else None
            )
            seg["generation_mode"] = "LLM_FALLBACK" if llm_fallback else "LLM"
            seg["template_bank"] = (
                f"TEMPLATE_{seg.get('stage', 'N2')}" if llm_fallback else ""
            )
        else:
            seg.setdefault("generation_mode", "TEMPLATE")
            seg.setdefault("llm_trigger_type", None)
            seg.setdefault("llm_latency_ms", None)
            seg.setdefault("llm_fallback_reason", None)
            seg.setdefault("template_bank", f"TEMPLATE_{seg.get('stage', 'N2')}")

        if not seg.get("narrative"):
            n, s, template_id = _template_narrative(
                seg,
                config,
                segment_index=i,
                narrative_cache=narrative_cache,
            )
            seg["narrative"] = n
            seg["scene_description"] = s
            seg["template_bank"] = template_id

    quality_scores, quality_summary = summarize_narrative_quality(raw_segments)
    for seg, score in zip(raw_segments, quality_scores):
        seg["narrative_quality"] = score
    _emit_simulation_progress_event(
        {
            "phase": "postprocess",
            "progress_percent": 92.0,
            "eta_seconds": 2,
            "eta_margin_seconds": 1,
        }
    )

    # Step 3: build response
    neuro_ticks = _build_neurochemistry_ticks(raw_segments)
    segments_out = [DreamSegmentResponse(**seg) for seg in raw_segments]

    # Summary statistics
    stages = [s.stage for s in segments_out]
    stage_counts = {st: stages.count(st) for st in set(stages)}
    total = len(stages)
    stage_pct = {st: round(cnt / total, 3) for st, cnt in stage_counts.items()}

    all_bizarre = [s.bizarreness_score for s in segments_out]
    all_emotions = [s.dominant_emotion for s in segments_out]
    dominant_emotion = max(set(all_emotions), key=all_emotions.count)
    llm_success_segments = sum(
        1 for s in raw_segments if s.get("generation_mode") == "LLM"
    )
    llm_fallback_segments = sum(
        1 for s in raw_segments if s.get("generation_mode") == "LLM_FALLBACK"
    )
    llm_total_invocations = llm_success_segments + llm_fallback_segments
    llm_latency_seconds_samples = [
        max(0.0, float(latency_ms) / 1000.0)
        for latency_ms in (
            seg.get("llm_latency_ms")
            for seg in raw_segments
            if seg.get("generation_mode") in {"LLM", "LLM_FALLBACK"}
        )
        if latency_ms is not None
    ]

    summary = {
        "total_segments": total,
        "night_span_hours": config.duration_hours,
        "sleep_start_hour": config.sleep_start_hour,
        "stage_distribution": stage_pct,
        "mean_bizarreness": round(float(np.mean(all_bizarre)), 3),
        "max_bizarreness": round(float(np.max(all_bizarre)), 3),
        "dominant_emotion": dominant_emotion,
        "rem_fraction": stage_pct.get("REM", 0.0),
        "llm_segments_generated": llm_success_segments,
        "llm_calls_total": llm_total_invocations,
        "llm_success_segments": llm_success_segments,
        "llm_fallback_segments": llm_fallback_segments,
        "llm_total_invocations": llm_total_invocations,
        "llm_triggered_segments": sum(
            1
            for s in raw_segments
            if s.get("generation_mode") in {"LLM", "LLM_FALLBACK"}
            and s.get("llm_trigger_type")
        ),
        "lucid_event_count": len(lucid_events),
        "pharmacology_profile": {
            "ssri_strength": config.ssri_strength,
            "melatonin": config.melatonin,
            "cannabis": config.cannabis,
        },
        "style_preset": config.style_preset,
        "prompt_profile": config.prompt_profile,
        **quality_summary,
    }

    result = SimulationResponse(
        id=sim_id,
        config=config,
        segments=segments_out,
        summary=summary,
        neurochemistry_ticks=neuro_ticks,
        neurochemistry_series=neuro_ticks,
        memory_activations=memory_activations,
        memory_activation_series=memory_activations,
        memory_graph=memory_graph,
        lucid_events=lucid_events,
        llm_used=llm_used,
        llm_model=client.config.model if llm_used else None,
    )
    result_payload = result.model_dump()
    _emit_simulation_progress_event(
        {
            "phase": "persist_export",
            "progress_percent": 96.0,
            "eta_seconds": 1,
            "eta_margin_seconds": 1,
        }
    )

    output_dir = _resolve_output_dir(sim_id)
    try:
        export_neurochemistry_csv(result_payload, output_dir / "neurochemistry.csv")
    except (ValueError, OSError) as exc:
        _record_export_failure()
        logger.error(
            "neurochemistry.csv export failed for simulation %s: %s", sim_id, exc
        )
    try:
        export_memory_activations_csv(
            result_payload, output_dir / "memory_activations.csv"
        )
    except (ValueError, OSError) as exc:
        _record_export_failure()
        logger.error(
            "memory_activations.csv export failed for simulation %s: %s", sim_id, exc
        )

    # Persist to in-memory store
    _simulations[sim_id] = result_payload
    _persist_simulation(sim_id, result_payload)
    _upsert_output_metadata(sim_id, result_payload, output_dir)
    _enforce_outputs_retention()
    _emit_simulation_progress_event(
        {
            "phase": "finalizing",
            "progress_percent": 99.0,
            "eta_seconds": 0,
            "eta_margin_seconds": 0,
        }
    )
    _record_simulation_completion(
        duration_seconds=time.perf_counter() - started_at,
        llm_invocations=llm_total_invocations,
        llm_fallback_segments=llm_fallback_segments,
        total_segments=total,
        llm_latency_seconds_samples=llm_latency_seconds_samples,
    )
    _audit(
        "simulation_completed",
        simulation_id=sim_id,
        total_segments=total,
        llm_used=llm_used,
    )
    logger.info("Simulation %s complete — %d segments, LLM=%s", sim_id, total, llm_used)

    return result


@app.post(
    "/api/simulation/multi-night",
    response_model=MultiNightSimulationResponse,
    tags=["Simulation"],
)
@app.post(
    "/api/v1/simulation/multi-night",
    response_model=MultiNightSimulationResponse,
    tags=["Simulation"],
)
async def simulate_multi_night(req: MultiNightSimulationRequest):
    series_id = str(uuid.uuid4())
    base_prior_events = [
        str(ev).strip() for ev in req.config.prior_day_events if str(ev).strip()
    ]
    night_results: List[SimulationResponse] = []
    night_payloads: List[dict[str, Any]] = []
    carryover_events_by_night: List[Dict[str, Any]] = []

    for night_idx in range(int(req.nights)):
        carryover_events: List[str] = []
        if req.carryover_memory and night_payloads:
            carryover_events = _derive_carryover_events_from_simulation(
                night_payloads[-1], top_k=int(req.carryover_top_k)
            )

        merged_prior_events: List[str] = []
        for event in [*base_prior_events, *carryover_events]:
            event_str = str(event).strip()
            if not event_str or event_str in merged_prior_events:
                continue
            merged_prior_events.append(event_str)
        merged_prior_events = merged_prior_events[: int(req.max_prior_events)]

        night_config = req.config.model_copy(
            update={"prior_day_events": merged_prior_events}
        )
        result = await simulate_night(night_config)
        result_payload = result.model_dump()
        summary_obj = result_payload.get("summary")
        if isinstance(summary_obj, dict):
            summary_obj["multi_night_series_id"] = series_id
            summary_obj["night_index"] = night_idx + 1
            summary_obj["carryover_event_count"] = len(carryover_events)
        if isinstance(result.summary, dict):
            result.summary["multi_night_series_id"] = series_id
            result.summary["night_index"] = night_idx + 1
            result.summary["carryover_event_count"] = len(carryover_events)
        night_payloads.append(result_payload)
        night_results.append(result)
        carryover_events_by_night.append(
            {
                "night_index": night_idx + 1,
                "carryover_events": carryover_events,
            }
        )

    continuity = _build_multi_night_continuity_payload(night_payloads)
    continuity["carryover_events_by_night"] = carryover_events_by_night
    summary = {
        "series_id": series_id,
        "night_count": len(night_results),
        "carryover_memory_enabled": bool(req.carryover_memory),
        "recurring_memory_count": int(continuity.get("recurring_memory_count", 0)),
        "night_averages": continuity.get("night_averages", {}),
    }
    _audit(
        "simulation_multi_night_completed",
        series_id=series_id,
        night_count=len(night_results),
        carryover_memory=bool(req.carryover_memory),
    )
    return MultiNightSimulationResponse(
        series_id=series_id,
        nights=night_results,
        summary=summary,
        continuity=continuity,
    )


@app.post(
    "/api/simulation/night/async",
    response_model=AsyncSimulationSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Simulation"],
)
@app.post(
    "/api/v1/simulation/night/async",
    response_model=AsyncSimulationSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Simulation"],
)
async def submit_simulation_night_async(config: SimulationConfig):
    job_id = str(uuid.uuid4())
    narrative_cfg = NarrativeGeneratorConfig.from_settings()
    llm_concurrency = max(1, int(narrative_cfg.concurrency))
    llm_timeout_seconds = max(5.0, float(narrative_cfg.timeout_seconds))
    dt = max(0.1, float(config.dt_minutes))
    ticks = max(1.0, (float(config.duration_hours) * 60.0) / dt)
    llm_density_guess = _estimate_llm_invocation_density()
    llm_expected_guess = (
        max(1, int(round(ticks * llm_density_guess))) if bool(config.use_llm) else 0
    )
    llm_avg_guess = (
        _estimate_llm_invocation_seconds(llm_timeout_seconds)
        if bool(config.use_llm)
        else 0.0
    )
    estimated_duration_seconds = _estimate_job_duration_seconds(
        config,
        expected_llm_invocations=(llm_expected_guess if bool(config.use_llm) else None),
        llm_concurrency=llm_concurrency,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    with _jobs_lock:
        running_jobs = sum(
            1
            for payload in _simulation_jobs.values()
            if str(payload.get("status", "")) == "running"
        )
        pending_jobs = sum(
            1
            for payload in _simulation_jobs.values()
            if str(payload.get("status", "")) == "pending"
        )
        queue_pressure = running_jobs >= _ASYNC_MAX_RUNNING_JOBS
        if pending_jobs >= _ASYNC_MAX_PENDING_JOBS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Async queue is full. "
                    f"pending={pending_jobs} max_pending={_ASYNC_MAX_PENDING_JOBS}"
                ),
            )
        _simulation_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "schema_version": "v2",
            "created_at": time.time(),
            "phase": "queued",
            "progress_percent": 0.0,
            "progress_source": "event_stream",
            "eta_seconds": int(round(estimated_duration_seconds)),
            "eta_source": (
                "queue_pressure_projection"
                if queue_pressure
                else "historical_projection"
            ),
            "eta_margin_seconds": int(
                round(max(3.0, estimated_duration_seconds * 0.2))
            ),
            "estimated_duration_seconds": round(estimated_duration_seconds, 2),
            "use_llm": bool(config.use_llm),
            "running_jobs_at_submit": running_jobs,
            "queue_pressure": queue_pressure,
            "duration_hours": float(config.duration_hours),
            "dt_minutes": float(config.dt_minutes),
            "llm_expected_invocations": llm_expected_guess,
            "llm_completed_invocations": 0,
            "llm_remaining_invocations": llm_expected_guess,
            "llm_avg_invocation_seconds": (
                round(llm_avg_guess, 3) if bool(config.use_llm) else None
            ),
            "llm_concurrency": llm_concurrency if bool(config.use_llm) else None,
        }
        _persist_job(job_id, dict(_simulation_jobs[job_id]))
    task = asyncio.create_task(_run_simulation_job(job_id, config))
    with _jobs_lock:
        _simulation_job_tasks[job_id] = task
    _audit("simulation_job_submitted", job_id=job_id)
    return AsyncSimulationSubmitResponse(
        job_id=job_id,
        status="pending",
        status_url=f"/api/simulation/jobs/{job_id}",
    )


@app.get(
    "/api/simulation/jobs/{job_id}",
    response_model=AsyncSimulationJobResponse,
    tags=["Simulation"],
)
@app.get(
    "/api/v1/simulation/jobs/{job_id}",
    response_model=AsyncSimulationJobResponse,
    tags=["Simulation"],
)
async def get_simulation_job(job_id: str):
    with _jobs_lock:
        job = _simulation_jobs.get(job_id)
    if not job:
        persisted = _load_persisted_job(job_id)
        if not persisted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        with _jobs_lock:
            _simulation_jobs[job_id] = persisted
            job = persisted
    if not isinstance(job, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    status_val = str(job.get("status", "pending"))
    progress_percent, eta_seconds = _job_progress_snapshot(job, status_val)
    progress_source, eta_source, provenance = _job_provenance_snapshot(job, status_val)
    return AsyncSimulationJobResponse(
        job_id=job_id,
        status=status_val,
        created_at=float(job.get("created_at", 0.0)),
        progress_percent=float(progress_percent),
        schema_version=str(job.get("schema_version", "v2")),
        phase=(str(job["phase"]) if job.get("phase") is not None else None),
        progress_source=progress_source,
        eta_seconds=eta_seconds,
        eta_source=eta_source,
        eta_margin_seconds=(
            int(job["eta_margin_seconds"])
            if job.get("eta_margin_seconds") is not None
            else None
        ),
        estimated_duration_seconds=(
            float(job["estimated_duration_seconds"])
            if job.get("estimated_duration_seconds") is not None
            else None
        ),
        provenance=provenance,
        started_at=(
            float(job["started_at"]) if job.get("started_at") is not None else None
        ),
        completed_at=(
            float(job["completed_at"]) if job.get("completed_at") is not None else None
        ),
        error_code=(
            str(job["error_code"]) if job.get("error_code") is not None else None
        ),
        error_message=(
            str(job["error_message"]) if job.get("error_message") is not None else None
        ),
        simulation_id=(
            str(job["simulation_id"]) if job.get("simulation_id") is not None else None
        ),
    )


@app.post(
    "/api/simulation/jobs/{job_id}/cancel",
    response_model=AsyncSimulationCancelResponse,
    tags=["Simulation"],
)
@app.post(
    "/api/v1/simulation/jobs/{job_id}/cancel",
    response_model=AsyncSimulationCancelResponse,
    tags=["Simulation"],
)
async def cancel_simulation_job(job_id: str):
    with _jobs_lock:
        job = _simulation_jobs.get(job_id)
    if not job:
        persisted = _load_persisted_job(job_id)
        if not persisted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        with _jobs_lock:
            _simulation_jobs[job_id] = persisted
            job = persisted
    if not isinstance(job, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    status_val = str(job.get("status", "pending"))
    if status_val in {"completed", "failed", "cancelled"}:
        return AsyncSimulationCancelResponse(
            job_id=job_id,
            status=status_val,
            message=f"Job already {status_val}.",
        )

    with _jobs_lock:
        task = _simulation_job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job["status"] = "cancelling"
            job["phase"] = "cancelling"
            job["error_code"] = "cancelled"
            job["error_message"] = "Cancellation requested by user."
            job["eta_seconds"] = None
            job["progress_source"] = "terminal"
            job["eta_source"] = "terminal"
            _persist_job(job_id, dict(job))
            _audit("simulation_job_cancellation_requested", job_id=job_id)
            return AsyncSimulationCancelResponse(
                job_id=job_id,
                status="cancelling",
                message="Cancellation requested.",
            )

    with _jobs_lock:
        job["status"] = "cancelled"
        job["completed_at"] = time.time()
        job["error_code"] = "cancelled"
        job["error_message"] = "Simulation cancelled by user."
        job["phase"] = "cancelled"
        job["eta_seconds"] = None
        job["progress_source"] = "terminal"
        job["eta_source"] = "terminal"
        _persist_job(job_id, dict(job))
    _audit("simulation_job_cancelled_without_task", job_id=job_id)
    return AsyncSimulationCancelResponse(
        job_id=job_id,
        status="cancelled",
        message="Job marked as cancelled.",
    )


@app.get(
    "/api/simulation/{sim_id}",
    response_model=SimulationResponse,
    tags=["Simulation"],
)
@app.get(
    "/api/v1/simulation/{sim_id}",
    response_model=SimulationResponse,
    tags=["Simulation"],
)
async def get_simulation(sim_id: str):
    """Retrieve a previously run simulation by ID."""
    payload = _resolve_simulation(sim_id)
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation '{sim_id}' not found. Run POST /api/simulation/night first.",
        )
    return payload


@app.get("/simulation/{sim_id}", response_model=SimulationResponse, tags=["Simulation"])
async def get_simulation_alias(sim_id: str):
    return await get_simulation(sim_id)


@app.get("/simulation/{sim_id}/segments", tags=["Simulation"])
@app.get("/api/simulation/{sim_id}/segments", tags=["Simulation"])
@app.get("/api/v1/simulation/{sim_id}/segments", tags=["Simulation"])
async def get_simulation_segments(sim_id: str, offset: int = 0, limit: int = 200):
    data = await get_simulation(sim_id)
    segments = data.get("segments", [])
    start = max(0, int(offset))
    end = max(start, start + max(1, min(int(limit), 1000)))
    return {
        "simulation_id": sim_id,
        "offset": start,
        "limit": end - start,
        "total": len(segments),
        "items": segments[start:end],
    }


@app.get("/simulation/{sim_id}/neurochemistry", tags=["Simulation"])
@app.get("/api/simulation/{sim_id}/neurochemistry", tags=["Simulation"])
@app.get("/api/v1/simulation/{sim_id}/neurochemistry", tags=["Simulation"])
async def get_simulation_neurochemistry(sim_id: str):
    data = await get_simulation(sim_id)
    rows = []
    for seg in data.get("segments", []):
        neuro = seg.get("neurochemistry") or {}
        rows.append(
            {
                "time_hours": seg.get("start_time_hours"),
                "ach": neuro.get("ach"),
                "serotonin": neuro.get("serotonin"),
                "ne": neuro.get("ne"),
                "cortisol": neuro.get("cortisol"),
            }
        )
    return {"simulation_id": sim_id, "series": rows}


@app.get("/simulation/{sim_id}/hypnogram", tags=["Simulation"])
@app.get("/api/simulation/{sim_id}/hypnogram", tags=["Simulation"])
@app.get("/api/v1/simulation/{sim_id}/hypnogram", tags=["Simulation"])
async def get_simulation_hypnogram(sim_id: str):
    data = await get_simulation(sim_id)
    rows = []
    for seg in data.get("segments", []):
        rows.append(
            {
                "time_hours": seg.get("start_time_hours"),
                "stage": seg.get("stage"),
            }
        )
    return {"simulation_id": sim_id, "hypnogram": rows}


@app.post("/simulate-night/stream", tags=["Simulation"])
@app.post("/api/v1/simulate-night/stream", tags=["Simulation"])
async def simulate_night_stream(config: SimulationConfig):
    async def _event_stream():
        yield "data: " + json.dumps(
            {"progress": 0.0, "stage": "init", "message": "Starting simulation"}
        ) + "\n\n"
        result = await simulate_night(config)
        yield (
            "data: "
            + json.dumps(
                {
                    "progress": 1.0,
                    "stage": "complete",
                    "message": "Simulation complete",
                    "result": result.model_dump(),
                }
            )
            + "\n\n"
        )

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/api/dreams", tags=["Simulation"])
@app.get("/api/v1/dreams", tags=["Simulation"])
async def list_dreams():
    """List all stored simulation summaries."""
    summaries = []
    sim_ids = set(_simulations.keys()) | set(_list_persisted_sim_ids())
    for sim_id in sorted(sim_ids):
        data = _resolve_simulation(sim_id)
        if not isinstance(data, dict):
            continue
        summaries.append(
            {
                "id": sim_id,
                "duration_hours": data["config"]["duration_hours"],
                "segment_count": len(data["segments"]),
                "dominant_emotion": data["summary"].get("dominant_emotion"),
                "mean_bizarreness": data["summary"].get("mean_bizarreness"),
                "llm_used": data["llm_used"],
                "llm_model": data.get("llm_model"),
                "rem_fraction": data["summary"].get("rem_fraction"),
            }
        )
    return {"count": len(summaries), "simulations": summaries}


@app.post(
    "/api/workspaces",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Collaboration"],
)
@app.post(
    "/api/v1/workspaces",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Collaboration"],
)
async def create_workspace(request: Request, body: WorkspaceCreateRequest):
    if not _scope_allowed(request, "workspace:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: workspace:write",
        )
    workspace_id = str(uuid.uuid4())
    now = time.time()
    payload = {
        "id": workspace_id,
        "name": body.name.strip(),
        "description": (body.description or "").strip() or None,
        "tags": [str(tag).strip() for tag in body.tags if str(tag).strip()][:20],
        "created_at": now,
        "updated_at": now,
        "run_ids": [],
        "run_meta": {},
    }
    with _workspaces_lock:
        _workspaces[workspace_id] = payload
    _persist_workspace(workspace_id, payload)
    _audit("workspace_created", workspace_id=workspace_id)
    return WorkspaceResponse(
        id=workspace_id,
        name=payload["name"],
        description=payload["description"],
        tags=payload["tags"],
        created_at=now,
        updated_at=now,
        run_ids=[],
    )


@app.get(
    "/api/workspaces/{workspace_id}",
    response_model=WorkspaceResponse,
    tags=["Collaboration"],
)
@app.get(
    "/api/v1/workspaces/{workspace_id}",
    response_model=WorkspaceResponse,
    tags=["Collaboration"],
)
async def get_workspace(workspace_id: str, request: Request):
    if not _scope_allowed(request, "workspace:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: workspace:read",
        )
    payload = _resolve_workspace(workspace_id)
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    return WorkspaceResponse(
        id=str(payload.get("id", workspace_id)),
        name=str(payload.get("name", "Untitled workspace")),
        description=(
            str(payload.get("description"))
            if payload.get("description") is not None
            else None
        ),
        tags=[str(tag) for tag in payload.get("tags", []) if str(tag).strip()],
        created_at=float(payload.get("created_at", 0.0)),
        updated_at=float(payload.get("updated_at", 0.0)),
        run_ids=[
            str(run_id) for run_id in payload.get("run_ids", []) if str(run_id).strip()
        ],
    )


@app.get("/api/workspaces", tags=["Collaboration"])
@app.get("/api/v1/workspaces", tags=["Collaboration"])
async def list_workspaces(request: Request, limit: int = 50):
    if not _scope_allowed(request, "workspace:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: workspace:read",
        )
    max_limit = max(1, min(int(limit), 500))
    ids = set(_workspaces.keys()) | set(_list_persisted_workspace_ids())
    rows: List[Dict[str, Any]] = []
    for workspace_id in ids:
        payload = _resolve_workspace(workspace_id)
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "id": str(payload.get("id", workspace_id)),
                "name": str(payload.get("name", "Untitled workspace")),
                "description": payload.get("description"),
                "tags": [
                    str(tag) for tag in payload.get("tags", []) if str(tag).strip()
                ],
                "created_at": float(payload.get("created_at", 0.0)),
                "updated_at": float(payload.get("updated_at", 0.0)),
                "run_count": len(payload.get("run_ids", [])),
            }
        )
    rows.sort(key=lambda item: float(item.get("updated_at", 0.0)), reverse=True)
    return {"count": len(rows), "items": rows[:max_limit]}


@app.post(
    "/api/workspaces/{workspace_id}/runs",
    response_model=WorkspaceAttachRunResponse,
    tags=["Collaboration"],
)
@app.post(
    "/api/v1/workspaces/{workspace_id}/runs",
    response_model=WorkspaceAttachRunResponse,
    tags=["Collaboration"],
)
async def attach_workspace_run(
    workspace_id: str,
    request: Request,
    body: WorkspaceAttachRunRequest,
):
    if not _scope_allowed(request, "workspace:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: workspace:write",
        )
    simulation_id = str(body.simulation_id).strip()
    simulation = _resolve_simulation(simulation_id)
    if not isinstance(simulation, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation '{simulation_id}' not found.",
        )
    with _workspaces_lock:
        payload = _resolve_workspace(workspace_id)
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )
        run_ids = [
            str(run_id) for run_id in payload.get("run_ids", []) if str(run_id).strip()
        ]
        attached = simulation_id not in run_ids
        if attached:
            run_ids.append(simulation_id)
        payload["run_ids"] = run_ids
        run_meta = payload.get("run_meta")
        if not isinstance(run_meta, dict):
            run_meta = {}
        run_meta[simulation_id] = {
            "label": (body.label or "").strip() or None,
            "attached_at": time.time(),
            "rem_fraction": float(
                ((simulation.get("summary") or {}).get("rem_fraction", 0.0) or 0.0)
            ),
            "mean_bizarreness": float(
                ((simulation.get("summary") or {}).get("mean_bizarreness", 0.0) or 0.0)
            ),
        }
        payload["run_meta"] = run_meta
        payload["updated_at"] = time.time()
        _workspaces[workspace_id] = payload
    _persist_workspace(workspace_id, payload)
    _audit(
        "workspace_run_attached",
        workspace_id=workspace_id,
        simulation_id=simulation_id,
        attached=attached,
    )
    return WorkspaceAttachRunResponse(
        workspace_id=workspace_id,
        simulation_id=simulation_id,
        attached=attached,
        run_count=len(payload.get("run_ids", [])),
    )


@app.get("/api/workspaces/{workspace_id}/runs", tags=["Collaboration"])
@app.get("/api/v1/workspaces/{workspace_id}/runs", tags=["Collaboration"])
async def list_workspace_runs(workspace_id: str, request: Request):
    if not _scope_allowed(request, "workspace:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: workspace:read",
        )
    payload = _resolve_workspace(workspace_id)
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    run_ids = [
        str(run_id) for run_id in payload.get("run_ids", []) if str(run_id).strip()
    ]
    run_meta = payload.get("run_meta")
    if not isinstance(run_meta, dict):
        run_meta = {}
    rows: List[Dict[str, Any]] = []
    for simulation_id in run_ids:
        sim_payload = _resolve_simulation(simulation_id)
        if not isinstance(sim_payload, dict):
            continue
        summary = sim_payload.get("summary") or {}
        meta = (
            run_meta.get(simulation_id)
            if isinstance(run_meta.get(simulation_id), dict)
            else {}
        )
        rows.append(
            {
                "simulation_id": simulation_id,
                "label": meta.get("label"),
                "attached_at": meta.get("attached_at"),
                "duration_hours": (sim_payload.get("config") or {}).get(
                    "duration_hours"
                ),
                "segment_count": len(sim_payload.get("segments") or []),
                "llm_used": bool(sim_payload.get("llm_used", False)),
                "mean_bizarreness": summary.get("mean_bizarreness"),
                "rem_fraction": summary.get("rem_fraction"),
                "narrative_quality_mean": summary.get("narrative_quality_mean"),
            }
        )
    rows.sort(key=lambda item: float(item.get("attached_at") or 0.0), reverse=True)
    return {
        "workspace_id": workspace_id,
        "count": len(rows),
        "items": rows,
    }


@app.post("/api/simulation/counterfactual", tags=["Simulation"])
@app.post("/api/v1/simulation/counterfactual", tags=["Simulation"])
async def counterfactual(req: CounterfactualRequest):
    """
    Run a counterfactual dream variant based on an existing simulation.

    Applies `perturbations` (parameter overrides) to the original config
    and re-runs the simulation, allowing side-by-side comparison.
    """
    base = _resolve_simulation(req.base_simulation_id)
    if not isinstance(base, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Base simulation '{req.base_simulation_id}' not found.",
        )

    base_config_dict = dict(base["config"])

    # Apply perturbations
    for key, val in req.perturbations.items():
        if key in base_config_dict:
            base_config_dict[key] = val

    base_config_dict["use_llm"] = req.use_llm
    new_config = SimulationConfig(**base_config_dict)

    return await simulate_night(new_config)


@app.post("/api/simulation/compare", tags=["Simulation"])
@app.post("/api/v1/simulation/compare", tags=["Simulation"])
async def compare_simulations(req: CompareRequest):
    baseline = _resolve_simulation(req.baseline_simulation_id)
    if not isinstance(baseline, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline simulation '{req.baseline_simulation_id}' not found.",
        )
    candidate = _resolve_simulation(req.candidate_simulation_id)
    if not isinstance(candidate, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate simulation '{req.candidate_simulation_id}' not found.",
        )
    payload = _build_comparison_payload(baseline, candidate)
    _audit(
        "simulation_compared",
        baseline_id=req.baseline_simulation_id,
        candidate_id=req.candidate_simulation_id,
    )
    return payload


@app.get("/api/simulation/{sim_id}/report", tags=["Simulation"])
@app.get("/api/v1/simulation/{sim_id}/report", tags=["Simulation"])
async def get_simulation_report(sim_id: str):
    sim = await get_simulation(sim_id)
    return _build_simulation_report_payload(sim_id=sim_id, sim=sim)


@app.get("/api/simulation/{sim_id}/report/bundle", tags=["Simulation"])
@app.get("/api/v1/simulation/{sim_id}/report/bundle", tags=["Simulation"])
async def get_simulation_report_bundle(sim_id: str):
    sim = await get_simulation(sim_id)
    report = _build_simulation_report_payload(sim_id=sim_id, sim=sim)
    segments = sim.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(
        zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "report.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "summary.json",
            json.dumps(sim.get("summary", {}), ensure_ascii=False, indent=2),
        )
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(
            [
                "segment_index",
                "start_time_hours",
                "end_time_hours",
                "stage",
                "dominant_emotion",
                "bizarreness_score",
                "lucidity_probability",
                "is_lucid",
                "generation_mode",
                "llm_fallback_reason",
                "narrative_quality_overall",
            ]
        )
        for idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            narrative_quality = seg.get("narrative_quality")
            narrative_quality_overall = (
                float(narrative_quality.get("overall", 0.0))
                if isinstance(narrative_quality, dict)
                else 0.0
            )
            writer.writerow(
                [
                    idx,
                    seg.get("start_time_hours"),
                    seg.get("end_time_hours"),
                    seg.get("stage"),
                    seg.get("dominant_emotion"),
                    seg.get("bizarreness_score"),
                    seg.get("lucidity_probability"),
                    bool(seg.get("is_lucid", False)),
                    seg.get("generation_mode"),
                    seg.get("llm_fallback_reason"),
                    narrative_quality_overall,
                ]
            )
        archive.writestr("segments_overview.csv", csv_buf.getvalue())

        methodology = report.get("methodology", {})
        methodology_lines = [
            "# DreamForge Simulation Methodology",
            "",
            f"Simulation ID: {sim_id}",
            "",
            f"- Sleep model: {methodology.get('sleep_model', '')}",
            f"- Narrative model: {methodology.get('narrative_model', '')}",
            f"- Quality scoring: {methodology.get('quality_scoring', '')}",
            "",
            "## Metric definitions",
        ]
        metric_defs = methodology.get("metric_definitions", {})
        if isinstance(metric_defs, dict):
            for key, value in metric_defs.items():
                methodology_lines.append(f"- {key}: {value}")
        methodology_lines.extend(["", "## Release targets"])
        release_targets = methodology.get("release_targets", {})
        if isinstance(release_targets, dict):
            for key, value in release_targets.items():
                methodology_lines.append(f"- {key}: {value}")
        archive.writestr("methodology.txt", "\n".join(methodology_lines).strip() + "\n")

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="dreamforge-report-bundle-{sim_id[:8]}.zip"'
            )
        },
    )


def _build_simulation_report_payload(
    sim_id: str, sim: dict[str, Any]
) -> dict[str, Any]:
    summary = sim.get("summary", {})
    quality = {
        "narrative_quality_mean": summary.get("narrative_quality_mean", 0.0),
        "narrative_length_compliance_mean": summary.get(
            "narrative_length_compliance_mean", 0.0
        ),
        "narrative_artifact_score_mean": summary.get(
            "narrative_artifact_score_mean", 0.0
        ),
        "narrative_memory_grounding_mean": summary.get(
            "narrative_memory_grounding_mean", 0.0
        ),
    }
    report = {
        "simulation_id": sim_id,
        "api_contract": API_CONTRACT_VERSION,
        "prompt_profile_version": PROMPT_PROFILE_VERSION,
        "generated_at_unix": time.time(),
        "summary": summary,
        "quality": quality,
        "methodology": {
            "sleep_model": "Borbély two-process + stage scheduler",
            "narrative_model": "Template/LLM hybrid with sanitization and fallback taxonomy",
            "quality_scoring": "Length compliance, artifact score, memory grounding, coherence proxy",
            "metric_definitions": {
                "narrative_quality_mean": "Mean of per-segment weighted quality score.",
                "narrative_memory_grounding_mean": "Mean token-overlap grounding confidence vs active memory IDs.",
                "llm_fallback_rate": "llm_fallback_segments / max(llm_total_invocations, 1).",
            },
            "release_targets": {
                "narrative_quality_min": 0.55,
                "memory_grounding_min": 0.10,
                "llm_fallback_rate_max": 0.35,
            },
        },
        "notes": [
            "This report is for product analytics and research exploration only.",
            "Not a medical or diagnostic report.",
        ],
    }
    return report


@app.get("/health")
async def health_check():
    client = get_llm_client()
    llm_status = await client.check_health()
    return {
        "status": "ok",
        "service": "dreamforge-api",
        "api_contract": API_CONTRACT_VERSION,
        "prompt_profile_version": PROMPT_PROFILE_VERSION,
        "llm_connected": bool(llm_status.get("ok", False)),
        "llm_provider": client.config.provider,
        "llm_model": client.config.model,
        "auth_enabled": bool(_API_ACCESS_TOKEN or _API_TOKEN_ROLE_MAP),
        "runtime_metrics": _read_runtime_metrics(),
    }


@app.get("/metrics")
@app.get("/api/v1/metrics")
async def metrics():
    return {
        "status": "ok",
        "service": "dreamforge-api",
        "metrics": _read_runtime_metrics(),
    }


@app.get("/metrics/prometheus")
@app.get("/api/v1/metrics/prometheus")
async def metrics_prometheus():
    metrics_snapshot = _read_runtime_metrics()
    return PlainTextResponse(_prometheus_metrics_text(metrics_snapshot))


@app.post(
    "/api/psg/connectors/channel-qa",
    response_model=PSGChannelQAResponse,
    tags=["PSG"],
)
@app.post(
    "/api/v1/psg/connectors/channel-qa",
    response_model=PSGChannelQAResponse,
    tags=["PSG"],
)
async def psg_channel_qa(payload: PSGChannelQARequest):
    return _psg_channel_qa(payload)


@app.get("/api/plugins/evaluators", tags=["System"])
@app.get("/api/v1/plugins/evaluators", tags=["System"])
async def list_plugin_evaluators():
    return {
        "count": len(_EVALUATOR_REGISTRY),
        "items": [
            {
                "name": name,
                "kind": "builtin",
                "input": "simulation summary payload",
                "output": "scored metric dictionary",
            }
            for name in sorted(_EVALUATOR_REGISTRY.keys())
        ],
    }


@app.post("/api/plugins/evaluators/run", tags=["System"])
@app.post("/api/v1/plugins/evaluators/run", tags=["System"])
async def run_plugin_evaluator(request: EvaluatorRunRequest):
    evaluator_name = str(request.evaluator).strip()
    evaluator = _EVALUATOR_REGISTRY.get(evaluator_name)
    if evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluator '{evaluator_name}' not found.",
        )
    return {
        "evaluator": evaluator_name,
        "scores": evaluator(request.summary),
    }


@app.get("/api/artifacts/health", tags=["System"])
@app.get("/api/v1/artifacts/health", tags=["System"])
async def artifact_health():
    return _artifact_health_snapshot(force_refresh=True)


@app.get("/api/version", tags=["System"])
@app.get("/api/v1/version", tags=["System"])
async def api_version():
    artifact_health_snapshot = _artifact_health_snapshot()
    return {
        "api_contract": API_CONTRACT_VERSION,
        "prompt_profile_version": PROMPT_PROFILE_VERSION,
        "async_job_schema_version": "v2",
        "async_limits": {
            "max_pending_jobs": _ASYNC_MAX_PENDING_JOBS,
            "max_running_jobs": _ASYNC_MAX_RUNNING_JOBS,
        },
        "artifact_manifest_pass": bool(artifact_health_snapshot.get("pass", False)),
        "artifact_manifest_path": str(
            artifact_health_snapshot.get("manifest_path", "")
        ),
        "deprecated": [],
        "changelog_policy": "Document every contract-level change in CHANGELOG.md",
    }


@app.get("/api/slo", tags=["System"])
@app.get("/api/v1/slo", tags=["System"])
async def api_slo():
    metrics_snapshot = _read_runtime_metrics()
    return {"targets": SLO_TARGETS, "current": metrics_snapshot}


@app.get("/api/error-taxonomy", tags=["System"])
@app.get("/api/v1/error-taxonomy", tags=["System"])
async def error_taxonomy():
    metrics_snapshot = _read_runtime_metrics()
    return {
        "taxonomy": ERROR_TAXONOMY,
        "observed_counts": metrics_snapshot.get("error_codes", {}),
    }


@app.get("/api/release-gate", tags=["System"])
@app.get("/api/v1/release-gate", tags=["System"])
async def release_gate_status():
    metrics_snapshot = _read_runtime_metrics()
    return _build_release_gate_status(metrics_snapshot)


@app.get("/api/audit/events", tags=["System"])
@app.get("/api/v1/audit/events", tags=["System"])
async def audit_events(
    request: Request,
    event: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 100,
):
    if not _scope_allowed(request, "audit:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scope: audit:read",
        )
    max_limit = max(1, min(int(limit), 1000))
    entries = list(_audit_events)
    if len(entries) < max_limit:
        entries.extend(_load_persisted_audit_events(limit=max_limit))
    filtered: List[Dict[str, Any]] = []
    event_filter = str(event).strip() if event else ""
    since_ts = float(since) if since is not None else None
    for row in entries:
        if not isinstance(row, dict):
            continue
        row_event = str(row.get("event", ""))
        row_ts = row.get("timestamp")
        try:
            row_ts_f = float(row_ts) if row_ts is not None else 0.0
        except (TypeError, ValueError):
            row_ts_f = 0.0
        if event_filter and row_event != event_filter:
            continue
        if since_ts is not None and row_ts_f < since_ts:
            continue
        filtered.append(row)
    filtered.sort(
        key=lambda item: float(item.get("timestamp", 0.0)),
        reverse=True,
    )
    return {
        "count": len(filtered[:max_limit]),
        "items": filtered[:max_limit],
    }


@app.get("/api/enterprise", tags=["System"])
@app.get("/api/v1/enterprise", tags=["System"])
async def enterprise_surface():
    return {
        "status": "active",
        "editions": ["community", "pro", "enterprise"],
        "value_props": [
            "API + dashboard baseline in OSS",
            "Team collaboration and analytics in Pro",
            "RBAC, audit, SLA, deployment controls in Enterprise",
        ],
        "links": {
            "waitlist": os.getenv("ENTERPRISE_WAITLIST_URL", ""),
            "trial": os.getenv("PRO_TRIAL_URL", ""),
            "sla_sheet": os.getenv("ENTERPRISE_SLA_URL", ""),
        },
    }
