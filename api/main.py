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
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from core.data.template_loader import SchemaValidationError, TemplateBank
from core.models.sleep_cycle import SleepCycleModel, SleepStage
from core.models.neurochemistry import cortisol_profile
from core.generation.narrative_generator import NarrativeGenerator
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
_api_error_codes: Dict[str, float] = defaultdict(float)
_request_windows: Dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()
_jobs_lock = threading.Lock()
_simulation_jobs: Dict[str, Dict[str, Any]] = {}
_simulation_job_tasks: Dict[str, asyncio.Task[Any]] = {}
_audit_events: deque[Dict[str, Any]] = deque(maxlen=2000)
_redis_lock = threading.Lock()
_redis_client: Any = None
_redis_disabled = False

_REDIS_KEY_SIMULATION_PREFIX = "dreamforge:simulation:"
_REDIS_KEY_SIMULATION_INDEX = "dreamforge:simulations"
_REDIS_KEY_JOB_PREFIX = "dreamforge:job:"
_REDIS_KEY_AUDIT_EVENTS = "dreamforge:audit:events"


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
    if client is None:
        return
    try:
        raw = json.dumps(payload, ensure_ascii=False)
        client.set(f"{_REDIS_KEY_SIMULATION_PREFIX}{sim_id}", raw)
        client.sadd(_REDIS_KEY_SIMULATION_INDEX, sim_id)
    except Exception as exc:
        logger.warning("Failed to persist simulation %s to Redis: %s", sim_id, exc)


def _load_persisted_simulation(sim_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(f"{_REDIS_KEY_SIMULATION_PREFIX}{sim_id}")
        if not raw:
            return None
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid persisted simulation payload for %s: %s", sim_id, exc)
        return None
    except Exception as exc:
        logger.warning("Failed loading simulation %s from Redis: %s", sim_id, exc)
        return None


def _persist_job(job_id: str, payload: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.set(
            f"{_REDIS_KEY_JOB_PREFIX}{job_id}", json.dumps(payload, ensure_ascii=False)
        )
    except Exception as exc:
        logger.warning("Failed to persist job %s to Redis: %s", job_id, exc)


def _load_persisted_job(job_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(f"{_REDIS_KEY_JOB_PREFIX}{job_id}")
        if not raw:
            return None
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid persisted job payload for %s: %s", job_id, exc)
        return None
    except Exception as exc:
        logger.warning("Failed loading job %s from Redis: %s", job_id, exc)
        return None


def _list_persisted_sim_ids() -> List[str]:
    client = _get_redis_client()
    if client is None:
        return []
    try:
        values = client.smembers(_REDIS_KEY_SIMULATION_INDEX) or []
        return [str(item) for item in values if str(item)]
    except Exception as exc:
        logger.warning("Failed listing persisted simulation IDs: %s", exc)
        return []


def _persist_audit_event(entry: Dict[str, Any]) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.lpush(_REDIS_KEY_AUDIT_EVENTS, json.dumps(entry, ensure_ascii=False))
        client.ltrim(_REDIS_KEY_AUDIT_EVENTS, 0, 1999)
    except Exception as exc:
        logger.warning("Failed to persist audit event: %s", exc)


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
) -> None:
    with _metrics_lock:
        _runtime_metrics["simulation_completed_total"] += 1.0
        _runtime_metrics["simulation_duration_seconds_total"] += max(
            0.0, float(duration_seconds)
        )
        _runtime_metrics["llm_invocations_total"] += float(max(0, llm_invocations))
        _runtime_metrics["llm_fallback_segments_total"] += float(
            max(0, llm_fallback_segments)
        )


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
    if path in {"/", "/health", "/metrics", "/metrics/prometheus"}:
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
    latency_proxy = float(metrics_snapshot.get("simulation_duration_seconds_avg", 0.0))
    export_success_rate = float(metrics_snapshot.get("export_success_rate", 0.0))

    checks = {
        "api_success_rate_pass": api_success_rate
        >= float(SLO_TARGETS["api_success_rate_min"]),
        "simulation_completion_rate_pass": simulation_completion_rate
        >= float(SLO_TARGETS["simulation_completion_rate_min"]),
        "simulation_latency_proxy_pass": latency_proxy
        <= float(SLO_TARGETS["simulation_p95_latency_seconds_max"]),
        "export_success_rate_pass": export_success_rate
        >= float(SLO_TARGETS["export_success_rate_min"]),
    }
    breaches = [name for name, ok in checks.items() if not ok]
    return {
        "pass": len(breaches) == 0,
        "checks": checks,
        "breaches": breaches,
        "targets": SLO_TARGETS,
        "current": metrics_snapshot,
        "notes": [
            "simulation_latency_proxy_pass uses average latency until p95 histograms are wired.",
            "Release gate should fail when any check is false.",
        ],
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
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    simulation_id: Optional[str] = None


class AsyncSimulationCancelResponse(BaseModel):
    job_id: str
    status: str
    message: str


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
    edges: List[Dict[str, Any]] = []
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
            if weight >= 0.15:
                edges.append(
                    {
                        "source": src["id"],
                        "target": dst["id"],
                        "weight": round(weight, 4),
                    }
                )

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
                boost = float(node["salience"]) * ach * 0.4
                node["activation"] = round(
                    min(1.0, float(node["activation"]) + boost), 4
                )

        target_valence = (
            -1.0
            if emotion in {"fearful", "anxious", "melancholic", "dread"}
            else 1.0 if emotion in {"joyful", "serene", "wonder"} else 0.0
        )
        for node in nodes:
            if target_valence != 0.0 and float(node["valence"]) * target_valence > 0:
                node["activation"] = round(
                    min(1.0, float(node["activation"]) + 0.15), 4
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


def _strip_thinking_tags(response: str) -> str:
    if not response:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


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
    return {
        "baseline_id": baseline.get("id"),
        "candidate_id": candidate.get("id"),
        "delta": {
            "mean_bizarreness": delta_mean_bizarreness,
            "rem_fraction": delta_rem_fraction,
            "lucid_event_count": delta_lucid_event_count,
            "narrative_quality_mean": delta_narrative_quality,
        },
        "confidence": {
            "sample_size": segment_count,
            "metric_confidence": {
                "mean_bizarreness": sample_confidence,
                "rem_fraction": sample_confidence,
                "lucid_event_count": sample_confidence,
                "narrative_quality_mean": sample_confidence,
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
            "baseline_llm_fallback_segments": int(
                b_summary.get("llm_fallback_segments", 0)
            ),
            "candidate_llm_fallback_segments": int(
                c_summary.get("llm_fallback_segments", 0)
            ),
        },
        "anomaly_flags": anomaly_flags,
        "generated_at_unix": time.time(),
    }


async def _run_simulation_job(job_id: str, config: SimulationConfig) -> None:
    with _jobs_lock:
        job = _simulation_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = time.time()
        _persist_job(job_id, dict(job))
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
            _persist_job(job_id, dict(job))
    finally:
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

    # Step 1: biophysical simulation
    raw_segments = _simulate_night_physics(config)
    memory_graph, memory_activations = _build_memory_outputs(
        raw_segments, config.prior_day_events
    )
    lucidity_threshold = LucidityModel.from_settings().threshold
    lucid_events = _annotate_lucid_events(raw_segments, lucidity_threshold)

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
        await generator.generate_batch(raw_segments)

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
    _record_simulation_completion(
        duration_seconds=time.perf_counter() - started_at,
        llm_invocations=llm_total_invocations,
        llm_fallback_segments=llm_fallback_segments,
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
    with _jobs_lock:
        _simulation_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
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
    return AsyncSimulationJobResponse(
        job_id=job_id,
        status=str(job.get("status", "pending")),
        created_at=float(job.get("created_at", 0.0)),
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
            job["error_code"] = "cancelled"
            job["error_message"] = "Cancellation requested by user."
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


@app.get("/api/version", tags=["System"])
@app.get("/api/v1/version", tags=["System"])
async def api_version():
    return {
        "api_contract": API_CONTRACT_VERSION,
        "prompt_profile_version": PROMPT_PROFILE_VERSION,
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
