from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from core.models.memory_graph import EmotionLabel
from core.models.sleep_cycle import SleepStage


class DreamSegment(BaseModel):
    """Structured representation of a dream segment for downstream analysis."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time_hours: float
    end_time_hours: float
    stage: SleepStage

    narrative: str = Field("", description="First-person description of the dream segment.")
    scene_description: str = Field("", description="Short, third-person scene summary.")

    active_memory_ids: List[str] = Field(default_factory=list)
    dominant_emotion: EmotionLabel = EmotionLabel.NEUTRAL

    bizarreness_score: float = Field(0.0, ge=0.0, le=1.0)
    lucidity_probability: float = Field(0.0, ge=0.0, le=1.0)

    metadata: dict = Field(default_factory=dict, description="Free-form metadata for analysis/visualization.")


class DreamNight(BaseModel):
    """Container for an entire simulated night of dreaming."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segments: List[DreamSegment]
    config: dict = Field(default_factory=dict)
    notes: Optional[str] = None
