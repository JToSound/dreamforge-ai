from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models.dream_segment import DreamSegment
from core.models.memory_graph import ReplaySequence, EmotionLabel
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState, SleepStage
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class DreamConstructorConfig:
    min_stage_for_dreaming: SleepStage = SleepStage.N1


class DreamConstructorAgent:
    """Agent that turns current brain state into a dream segment.

    For now, this uses simple template-based narratives with explicit hooks for
    future LLM integration.
    """

    def __init__(self, event_bus: Optional[EventBus] = None, config: Optional[DreamConstructorConfig] = None) -> None:
        self.event_bus = event_bus or EventBus()
        self.config = config or DreamConstructorConfig()
        self._last_time_hours: float = 0.0

    def step(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        replay_seq: Optional[ReplaySequence],
    ) -> Optional[DreamSegment]:
        # Only generate dream content in sleep (optionally only REM/N2/N3).
        if sleep_state.stage == SleepStage.WAKE:
            self._last_time_hours = sleep_state.time_hours
            return None

        segment = DreamSegment(
            start_time_hours=self._last_time_hours,
            end_time_hours=sleep_state.time_hours,
            stage=sleep_state.stage,
        )

        # Very simple heuristics for now.
        dominant_emotion = replay_seq.dominant_emotion if replay_seq is not None else EmotionLabel.NEUTRAL
        segment.dominant_emotion = dominant_emotion
        if replay_seq is not None:
            segment.active_memory_ids = replay_seq.node_ids

        segment.narrative = self._build_narrative(sleep_state, neuro_state, segment)
        segment.scene_description = self._build_scene_description(sleep_state, segment)
        segment.bizarreness_score = self._estimate_bizarreness(segment)

        # Emit event for downstream consumers.
        self._emit_event(segment)
        self._last_time_hours = sleep_state.time_hours
        return segment

    # ------------------------------------------------------------------
    # Internal helpers (LLM hooks could be added here later)
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        segment: DreamSegment,
    ) -> str:
        # Placeholder: combine stage, emotion, and time into a short narrative.
        emotion = segment.dominant_emotion.value
        return (
            f"You find yourself in a dream during {sleep_state.stage.value}. "
            f"The emotional tone feels {emotion}. Time flows strangely as the night progresses."
        )

    def _build_scene_description(self, sleep_state: SleepState, segment: DreamSegment) -> str:
        return (
            f"A vignette from {sleep_state.stage.value} sleep, colored by {segment.dominant_emotion.value} emotion."
        )

    def _estimate_bizarreness(self, segment: DreamSegment) -> float:
        # Simple placeholder heuristic based on number of active memories.
        n = len(segment.active_memory_ids)
        return min(1.0, 0.1 * n)

    def _emit_event(self, segment: DreamSegment) -> None:
        event = Event(
            type=EventType.MEMORY_REPLAY_EVENT,
            payload={"segment_id": segment.id},
            timestamp_hours=segment.end_time_hours,
        )
        self.event_bus.publish(event)
