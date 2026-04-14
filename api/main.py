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
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from core.models.sleep_cycle import SleepCycleModel
from core.simulation.llm_client import parse_narrative_response

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dreamforge.api")

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
    sleep_start_hour: float = Field(23.0, ge=18.0, le=26.0)
    prior_day_events: List[str] = Field(default_factory=list)
    emotional_state: str = Field("neutral")
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
    neurochemistry: Optional[Dict[str, float]] = None
    active_memory_ids: List[str] = []
    generation_mode: str = "TEMPLATE"


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
    llm_used: bool = False
    llm_model: Optional[str] = None


class CounterfactualRequest(BaseModel):
    base_simulation_id: str
    perturbations: Dict[str, Any] = Field(...)
    use_llm: bool = True


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

    segments = []

    # Simple neurochemistry baselines by stage
    neuro_by_stage = {
        "N1": {"ach": 0.50, "serotonin": 0.30, "ne": 0.30, "cortisol": 0.40},
        "N2": {"ach": 0.45, "serotonin": 0.28, "ne": 0.28, "cortisol": 0.38},
        "N3": {"ach": 0.30, "serotonin": 0.20, "ne": 0.20, "cortisol": 0.35},
        "REM": {"ach": 0.90, "serotonin": 0.05, "ne": 0.05, "cortisol": 0.50},
        "WAKE": {"ach": 0.70, "serotonin": 0.80, "ne": 0.80, "cortisol": 0.60},
    }

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

        # Neurochemistry with noise + SSRI modulation
        neuro = dict(neuro_by_stage[stage])
        neuro["serotonin"] = min(1.0, neuro["serotonin"] * config.ssri_strength)
        neuro = {k: max(0.0, v + random.gauss(0, 0.02)) for k, v in neuro.items()}

        # Bizarreness: primarily driven by ACh, with secondary contribution from
        # suppressed monoamines and stress; add small noise for variability.
        alpha = 0.6
        mono_coeff = 0.25
        ne_coeff = 0.1
        stress_coeff = 0.05
        noise_std = 0.03

        biz_val = (
            alpha * neuro["ach"]
            + mono_coeff * (1.0 - neuro["serotonin"])
            + ne_coeff * (1.0 - neuro["ne"])
            + stress_coeff * config.stress_level
        )
        bizarreness = round(min(1.0, max(0.0, biz_val + random.gauss(0, noise_std))), 3)

        # Lucidity probability: highest during REM with lower stress
        lucidity = (
            0.05
            if stage != "REM"
            else max(0.0, 0.15 + neuro["ach"] * 0.10 - config.stress_level * 0.08)
        )
        lucidity = round(min(1.0, max(0.0, lucidity + random.gauss(0, 0.01))), 4)

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
                "neurochemistry": {k: round(v, 4) for k, v in neuro.items()},
                "active_memory_ids": [],
                # Placeholders — will be replaced by LLM below
                "narrative": "",
                "scene_description": "",
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
    edge_weights: Dict[tuple[str, str], float] = {}
    snapshots: List[Dict[str, Any]] = []
    prev_node_id: Optional[str] = None

    for event in prior_day_events:
        label = event.strip()[:80]
        if not label:
            continue
        node_id = f"event::{label.lower()}"
        node_map[node_id] = {
            "id": node_id,
            "label": label,
            "emotion": "neutral",
            "activation": 0.25,
            "salience": 0.45,
        }

    for seg in segments:
        emotion = str(seg.get("dominant_emotion") or "neutral")
        node_id = f"emotion::{emotion}"
        if node_id not in node_map:
            node_map[node_id] = {
                "id": node_id,
                "label": emotion,
                "emotion": emotion,
                "activation": 0.35,
                "salience": 0.50,
            }

        if prev_node_id and prev_node_id != node_id:
            edge = (prev_node_id, node_id)
            edge_weights[edge] = edge_weights.get(edge, 0.0) + 1.0
        prev_node_id = node_id

        if seg.get("stage") == "REM":
            biz = float(seg.get("bizarreness_score") or 0.0)
            activation = round(min(1.0, max(0.05, 0.2 + 0.8 * biz)), 4)
            snapshots.append(
                {
                    "time_hours": seg.get("start_time_hours"),
                    "stage": "REM",
                    "activations": [
                        {"id": node_id, "label": emotion, "activation": activation}
                    ],
                }
            )

    nodes = list(node_map.values())
    edges = [
        {"source": src, "target": dst, "weight": round(weight, 3)}
        for (src, dst), weight in edge_weights.items()
    ]
    return {"nodes": nodes, "edges": edges}, snapshots


def _template_narrative(seg: dict, config: SimulationConfig) -> tuple[str, str]:
    """Fallback template-based narrative when LLM is disabled."""
    stage = seg["stage"]
    emotion = seg["dominant_emotion"]
    bizarre = seg["bizarreness_score"]

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
    return templates.get(stage, ""), scene_templates.get(stage, "")


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


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", tags=["System"])
async def root():
    """Health ping."""
    return {"service": "DreamForge AI", "status": "running", "version": "0.2.0"}


@app.get("/api/health/llm", response_model=LLMHealthResponse, tags=["LLM"])
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
async def llm_health_alias():
    return await llm_health()


@app.get("/api/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
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


@app.post("/api/llm/config", response_model=LLMConfigResponse, tags=["LLM"])
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
    logger.info(
        "Starting simulation %s (%.1fh, dt=%.1fmin, LLM=%s)",
        sim_id,
        config.duration_hours,
        config.dt_minutes,
        config.use_llm,
    )

    # Step 1: biophysical simulation
    raw_segments = _simulate_night_physics(config)

    # Step 2: narrative generation
    client = get_llm_client()
    llm_used = False

    # Determine which segments get LLM treatment
    if config.use_llm:
        if config.llm_segments_only:
            llm_indices = [i for i, s in enumerate(raw_segments) if s["stage"] == "REM"]
            # Source: DreamForge architecture spec v4.0 (target ~15–30 LLM calls / 8h)
            target_calls = max(10, min(50, int(config.duration_hours * 3)))
            if len(llm_indices) > target_calls:
                step = max(1, len(llm_indices) // target_calls)
                llm_indices = llm_indices[::step][:target_calls]
        else:
            # Source: DreamForge architecture spec v4.0 (target ~15–30 LLM calls / 8h)
            target_calls = max(10, min(50, int(config.duration_hours * 3)))
            sample_step = max(1, len(raw_segments) // target_calls)
            llm_indices = list(range(0, len(raw_segments), sample_step))[:target_calls]
    else:
        llm_indices = []

    # Call LLM for selected segments (with concurrency limit)
    semaphore = asyncio.Semaphore(3)  # max 3 concurrent LLM calls

    async def _safe_llm(idx: int):
        nonlocal llm_used
        seg = raw_segments[idx]
        async with semaphore:
            try:
                narrative, scene = await _generate_llm_narrative(seg, config, client)
                llm_used = True
                raw_segments[idx]["generation_mode"] = "LLM"
            except Exception as exc:
                logger.error("LLM narrative failed for segment %d: %s", idx, exc)
                narrative, scene = _template_narrative(seg, config)
                raw_segments[idx]["generation_mode"] = "LLM_FALLBACK"
        raw_segments[idx]["narrative"] = narrative
        raw_segments[idx]["scene_description"] = scene

    if llm_indices:
        await asyncio.gather(*[_safe_llm(i) for i in llm_indices])

    # Fill remaining segments with templates
    for i, seg in enumerate(raw_segments):
        seg.setdefault("generation_mode", "TEMPLATE")
        if not seg["narrative"]:
            n, s = _template_narrative(seg, config)
            seg["narrative"] = n
            seg["scene_description"] = s

    # Step 3: build response
    segments_out = [DreamSegmentResponse(**seg) for seg in raw_segments]
    neuro_ticks = _build_neurochemistry_ticks(raw_segments)
    memory_graph, memory_activations = _build_memory_outputs(
        raw_segments, config.prior_day_events
    )

    # Summary statistics
    stages = [s.stage for s in segments_out]
    stage_counts = {st: stages.count(st) for st in set(stages)}
    total = len(stages)
    stage_pct = {st: round(cnt / total, 3) for st, cnt in stage_counts.items()}

    all_bizarre = [s.bizarreness_score for s in segments_out]
    all_emotions = [s.dominant_emotion for s in segments_out]
    dominant_emotion = max(set(all_emotions), key=all_emotions.count)

    summary = {
        "total_segments": total,
        "night_span_hours": config.duration_hours,
        "stage_distribution": stage_pct,
        "mean_bizarreness": round(float(np.mean(all_bizarre)), 3),
        "max_bizarreness": round(float(np.max(all_bizarre)), 3),
        "dominant_emotion": dominant_emotion,
        "rem_fraction": stage_pct.get("REM", 0.0),
        "llm_segments_generated": len(llm_indices) if llm_used else 0,
        "llm_calls_total": len(llm_indices) if llm_used else 0,
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
        llm_used=llm_used,
        llm_model=client.config.model if llm_used else None,
    )

    # Persist to in-memory store
    _simulations[sim_id] = result.model_dump()
    logger.info("Simulation %s complete — %d segments, LLM=%s", sim_id, total, llm_used)

    return result


@app.get(
    "/api/simulation/{sim_id}",
    response_model=SimulationResponse,
    tags=["Simulation"],
)
async def get_simulation(sim_id: str):
    """Retrieve a previously run simulation by ID."""
    if sim_id not in _simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation '{sim_id}' not found. Run POST /api/simulation/night first.",
        )
    return _simulations[sim_id]


@app.get("/simulation/{sim_id}", response_model=SimulationResponse, tags=["Simulation"])
async def get_simulation_alias(sim_id: str):
    return await get_simulation(sim_id)


@app.get("/simulation/{sim_id}/segments", tags=["Simulation"])
@app.get("/api/simulation/{sim_id}/segments", tags=["Simulation"])
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
async def list_dreams():
    """List all stored simulation summaries."""
    summaries = []
    for sim_id, data in _simulations.items():
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
async def counterfactual(req: CounterfactualRequest):
    """
    Run a counterfactual dream variant based on an existing simulation.

    Applies `perturbations` (parameter overrides) to the original config
    and re-runs the simulation, allowing side-by-side comparison.
    """
    if req.base_simulation_id not in _simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Base simulation '{req.base_simulation_id}' not found.",
        )

    base = _simulations[req.base_simulation_id]
    base_config_dict = dict(base["config"])

    # Apply perturbations
    for key, val in req.perturbations.items():
        if key in base_config_dict:
            base_config_dict[key] = val

    base_config_dict["use_llm"] = req.use_llm
    new_config = SimulationConfig(**base_config_dict)

    return await simulate_night(new_config)


@app.get("/health")
async def health_check():
    client = get_llm_client()
    llm_status = await client.check_health()
    return {
        "status": "ok",
        "service": "dreamforge-api",
        "llm_connected": bool(llm_status.get("ok", False)),
        "llm_provider": client.config.provider,
        "llm_model": client.config.model,
    }
