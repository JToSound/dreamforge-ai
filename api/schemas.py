"""Pydantic schemas for DreamForge AI REST API.

All request/response models are defined here to keep api/routes/* thin.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMConfig(BaseModel):
    """LLM backend configuration supplied by the user at runtime."""

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM provider to use.",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model name/ID (e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022', 'llama3').",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider. Leave null to read from environment variable.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for Ollama or OpenAI-compatible endpoints (e.g. http://localhost:11434/v1).",
    )
    temperature: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the LLM.",
    )
    max_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Max tokens per LLM call.",
    )


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

class PharmacologyInput(BaseModel):
    ssri: bool = Field(default=False, description="Whether subject is on an SSRI.")
    ssri_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    melatonin: bool = Field(default=False)
    cannabis: bool = Field(default=False)


class DayInput(BaseModel):
    """Events / emotional state of the day prior to sleep."""

    events: list[str] = Field(
        default_factory=list,
        description="Brief text descriptions of notable events from the day.",
    )
    stress_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Subjective stress level (0 = calm, 1 = extremely stressed).",
    )
    mood: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Baseline mood valence (0 = very negative, 1 = very positive).",
    )
    pharmacology: PharmacologyInput = Field(default_factory=PharmacologyInput)


class SimulationConfig(BaseModel):
    duration_hours: float = Field(default=8.0, ge=1.0, le=12.0)
    dt_minutes: float = Field(default=0.5, ge=0.1, le=5.0)
    sleep_start_clock_time: float = Field(default=23.0, ge=18.0, le=30.0)
    tau_wake: float = Field(default=18.0)
    tau_sleep: float = Field(default=4.5)
    noise_std: float = Field(default=0.05)


class SimulationRequest(BaseModel):
    """Full request body for POST /simulate-night."""

    day_input: DayInput = Field(default_factory=DayInput)
    config: SimulationConfig = Field(default_factory=SimulationConfig)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM backend to use for narrative/phenomenology generation.",
    )
    generate_narrative: bool = Field(
        default=True,
        description="If True, call the LLM to generate dream narratives. If False, skip LLM calls (faster).",
    )


# ---------------------------------------------------------------------------
# Simulation response
# ---------------------------------------------------------------------------

class NeurochemSnapshot(BaseModel):
    time_hours: float
    ach: float
    serotonin: float
    ne: float
    cortisol: float


class DreamSegmentOut(BaseModel):
    segment_index: int
    time_hours: float
    stage: str
    narrative: str
    dominant_emotion: str
    bizarreness_score: float
    lucidity_score: float
    active_memory_labels: list[str]
    neurochemistry: NeurochemSnapshot


class MemoryNodeOut(BaseModel):
    id: str
    label: str
    memory_type: str
    activation: float
    salience: float
    emotion: str
    arousal: float


class MemoryEdgeOut(BaseModel):
    source: str
    target: str
    weight: float


class AgentActivityPoint(BaseModel):
    time_hours: float
    agent: str
    event: str
    confidence: float


class SimulationResult(BaseModel):
    simulation_id: str
    duration_hours: float
    total_segments: int
    hypnogram: list[dict[str, Any]]
    neurochemistry_series: list[NeurochemSnapshot]
    dream_segments: list[DreamSegmentOut]
    memory_graph: dict[str, Any]
    agent_activity: list[AgentActivityPoint]
    llm_provider_used: str
    llm_model_used: str
    metadata: dict[str, Any] = Field(default_factory=dict)
