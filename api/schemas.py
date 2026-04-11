from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.models.dream_segment import DreamNight


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class DreamSimulationRequest(BaseModel):
    """Request body for running a single-night simulation."""

    duration_hours: float = Field(8.0, gt=0.0)
    dt_minutes: float = Field(0.5, gt=0.0)
    ssri_strength: float = Field(1.0, gt=0.0)
    stress_level: float = Field(0.0, ge=0.0)

    llm_enabled: bool = Field(False)
    llm_provider: Optional[str] = Field(None)
    llm_model: Optional[str] = Field(None)
    llm_important_only: bool = Field(True)
    llm_api_key: Optional[str] = Field(
        None,
        description="Optional API key passed from the frontend; overrides the server env var.",
    )


# ---------------------------------------------------------------------------
# Segment / Summary
# ---------------------------------------------------------------------------

class DreamSegmentSchema(BaseModel):
    id: str
    start_time_hours: float
    end_time_hours: float
    stage: str
    narrative: str
    scene_description: str
    dominant_emotion: str
    bizarreness_score: float
    lucidity_probability: float


class NightSummarySchema(BaseModel):
    sleep_stages: Dict[str, float]
    neurochemistry: Dict[str, Dict[str, float]]
    bizarreness: Dict[str, Any]
    memory: Dict[str, Any]


# ---------------------------------------------------------------------------
# Timeseries schemas (new — for hypnogram / neuro-flux / memory-graph viz)
# ---------------------------------------------------------------------------

class SleepStateSchema(BaseModel):
    time_hours: float
    stage: str
    process_s: float
    process_c: float


class NeuroStateSchema(BaseModel):
    time_hours: float
    ach: float
    serotonin: float
    ne: float
    cortisol: float


class MemoryGraphSchema(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class DreamNightSchema(BaseModel):
    id: str
    config: Dict[str, Any]
    segments: List[DreamSegmentSchema]
    summary: NightSummarySchema
    # Raw timeseries for frontend charts
    sleep_history: List[SleepStateSchema] = Field(default_factory=list)
    neuro_history: List[NeuroStateSchema] = Field(default_factory=list)
    memory_graph: MemoryGraphSchema = Field(
        default_factory=lambda: MemoryGraphSchema(nodes=[], edges=[])
    )


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def serialize_dream_night(night: DreamNight) -> DreamNightSchema:
    meta: Dict[str, Any] = night.metadata or {}
    summary: Dict[str, Any] = meta.get("summary", {})

    sleep_raw: List[Dict[str, Any]] = meta.get("sleep_history", [])
    neuro_raw: List[Dict[str, Any]] = meta.get("neuro_history", [])
    memory_raw: Dict[str, Any] = meta.get("memory_graph", {"nodes": [], "edges": []})

    return DreamNightSchema(
        id=night.id,
        config=night.config,
        segments=[
            DreamSegmentSchema(
                id=s.id,
                start_time_hours=s.start_time_hours,
                end_time_hours=s.end_time_hours,
                stage=s.stage.value,
                narrative=s.narrative,
                scene_description=s.scene_description,
                dominant_emotion=s.dominant_emotion.value,
                bizarreness_score=s.bizarreness_score,
                lucidity_probability=s.lucidity_probability,
            )
            for s in night.segments
        ],
        summary=NightSummarySchema(
            sleep_stages=summary.get("sleep_stages", {}),
            neurochemistry=summary.get("neurochemistry", {}),
            bizarreness=summary.get("bizarreness", {}),
            memory=summary.get("memory", {}),
        ),
        sleep_history=[SleepStateSchema(**s) for s in sleep_raw],
        neuro_history=[NeuroStateSchema(**n) for n in neuro_raw],
        memory_graph=MemoryGraphSchema(
            nodes=memory_raw.get("nodes", []),
            edges=memory_raw.get("edges", []),
        ),
    )
