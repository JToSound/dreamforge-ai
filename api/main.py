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

import json
import logging
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from core.data.template_loader import SchemaValidationError, TemplateBank
from core.models.sleep_cycle import SleepCycleModel, SleepStage
from core.models.neurochemistry import cortisol_profile
from core.generation.narrative_generator import NarrativeGenerator
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
    is_lucid: bool = False
    neurochemistry: Optional[Dict[str, float]] = None
    active_memory_ids: List[str] = []
    generation_mode: str = "TEMPLATE"
    llm_trigger_type: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    llm_fallback_reason: Optional[str] = None
    template_bank: Optional[str] = None


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
        text: str, stage_name: str, bizarre_score: float, marker_hours: float
    ) -> str:
        min_words, max_words = _window_for_stage(stage_name, bizarre_score)
        base = " ".join(str(text).split())
        if stage_name in {"N1", "N2"}:
            base = f"{base} At {marker_hours:.3f}h, the image tilts and resets."
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
                _normalize_template(narrative, stage, float(bizarre), time_marker),
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
        _normalize_template(narrative, stage, float(bizarre), time_marker),
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
        logger.error(
            "neurochemistry.csv export failed for simulation %s: %s", sim_id, exc
        )
    try:
        export_memory_activations_csv(
            result_payload, output_dir / "memory_activations.csv"
        )
    except (ValueError, OSError) as exc:
        logger.error(
            "memory_activations.csv export failed for simulation %s: %s", sim_id, exc
        )

    # Persist to in-memory store
    _simulations[sim_id] = result_payload
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
