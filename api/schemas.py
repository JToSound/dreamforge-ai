from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── LLM config ──────────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = Field(default=None, description="API key; omit to use env var.")
    base_url: Optional[str] = Field(default=None, description="Override base URL (e.g. for Ollama).")
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=64, le=4096)


# ── Sub-configs ──────────────────────────────────────────────────────────────

class SleepConfig(BaseModel):
    duration_hours: float = Field(default=8.0, ge=1.0, le=12.0)
    sleep_start_clock_time: float = Field(default=23.0, ge=18.0, le=26.0)
    dt_minutes: float = Field(default=0.5, ge=0.1, le=5.0)
    tau_wake: float = Field(default=18.0, gt=0.0)
    tau_sleep: float = Field(default=4.5, gt=0.0)


class NeurochemistryConfig(BaseModel):
    noise_std: float = Field(default=0.05, ge=0.0, le=0.5)
    ssri_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    cortisol_amplitude: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryConfig(BaseModel):
    max_nodes: int = Field(default=200, ge=10, le=1000)
    decay_rate: float = Field(default=0.02, ge=0.0, le=0.5)
    prune_threshold: float = Field(default=0.1, ge=0.0, le=0.5)
    replay_max_length: int = Field(default=10, ge=2, le=30)


# ── Main request / response ──────────────────────────────────────────────────

class SimulateNightRequest(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sleep: SleepConfig = Field(default_factory=SleepConfig)
    neurochemistry: NeurochemistryConfig = Field(default_factory=NeurochemistryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    prior_day_events: list[str] = Field(default_factory=list)
    stress_level: float = Field(default=0.5, ge=0.0, le=1.0)
    user_id: Optional[str] = None


class DreamSegmentOut(BaseModel):
    segment_index: int
    time_hours: float
    stage: str
    narrative: str
    dominant_emotion: str
    bizarreness_score: float
    lucidity_probability: float
    active_memory_ids: list[str]
    neurochemistry: dict[str, float]


class HypnogramPoint(BaseModel):
    time_hours: float
    stage: str
    process_s: float
    process_c: float


class SimulateNightResponse(BaseModel):
    simulation_id: str
    duration_hours: float
    total_segments: int
    hypnogram: list[HypnogramPoint]
    neurochemistry_series: list[dict[str, Any]]
    dream_segments: list[DreamSegmentOut]
    memory_graph: dict[str, Any]
    summary_narrative: str
    mean_bizarreness: float
    dominant_emotion: str


class ProgressEvent(BaseModel):
    simulation_id: str
    progress: float           # 0.0 – 1.0
    stage: str
    message: str
    segment: Optional[DreamSegmentOut] = None