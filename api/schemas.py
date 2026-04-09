from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.models.dream_segment import DreamNight, DreamSegment


class DreamSimulationRequest(BaseModel):
    """Request body for running a single-night simulation."""

    duration_hours: float = Field(8.0, gt=0.0)
    dt_minutes: float = Field(0.5, gt=0.0)
    ssri_strength: float = Field(1.0, gt=0.0)
    stress_level: float = Field(0.0, ge=0.0)

    llm_enabled: bool = Field(False, description="Enable LLM-backed dream narratives.")
    llm_provider: Optional[str] = Field(None, description="LLM provider identifier (openai, lmstudio, ollama, ...).")
    llm_model: Optional[str] = Field(None, description="Model name for the selected provider.")
    llm_important_only: bool = Field(
        True,
        description="If true, only use the LLM for REM or high-replay segments to save tokens.",
    )


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


class DreamNightSchema(BaseModel):
    id: str
    config: Dict[str, Any]
    segments: List[DreamSegmentSchema]
    summary: NightSummarySchema


def serialize_dream_night(night: DreamNight) -> DreamNightSchema:
    summary: Dict[str, Any] = night.metadata.get("summary", {})
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
    )
