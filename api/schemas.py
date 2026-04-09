from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from core.models.dream_segment import DreamNight, DreamSegment


class DreamSimulationRequest(BaseModel):
    """Request body for running a single-night simulation."""

    duration_hours: float = Field(8.0, gt=0.0)
    dt_minutes: float = Field(0.5, gt=0.0)
    ssri_strength: float = Field(1.0, gt=0.0)
    stress_level: float = Field(0.0, ge=0.0)


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


class DreamNightSchema(BaseModel):
    id: str
    config: Dict[str, Any]
    segments: List[DreamSegmentSchema]


def serialize_dream_night(night: DreamNight) -> DreamNightSchema:
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
    )
