from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from core.models.dream_segment import DreamSegment
from core.models.memory_graph import ReplaySequence, EmotionLabel
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState, SleepStage
from core.simulation.event_bus import EventBus, Event, EventType


class LLMCallable(Protocol):
    """Protocol for pluggable LLM backends.

    Any callable that takes a prompt string and returns a completion string can
    satisfy this protocol.
    """

    def __call__(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class DreamConstructorConfig:
    min_stage_for_dreaming: SleepStage = SleepStage.N1
    use_llm: bool = True
    important_only: bool = True


_STAGE_ORDER = {
    SleepStage.WAKE: 0,
    SleepStage.N1: 1,
    SleepStage.N2: 2,
    SleepStage.N3: 3,
    SleepStage.REM: 4,
}


class DreamConstructorAgent:
    """Agent that turns current brain state into a dream segment.

    By default it uses lightweight template-based narratives, but it exposes a
    simple hook for plugging in any LLM backend (OpenAI, Anthropic, local
    models via Ollama/LM Studio, etc.).
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[DreamConstructorConfig] = None,
        llm: Optional[LLMCallable] = None,
        important_only: Optional[bool] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.config = config or DreamConstructorConfig()
        if important_only is not None:
            self.config.important_only = important_only
        self._last_time_hours: float = 0.0
        self._llm: Optional[LLMCallable] = llm

    def set_llm_backend(self, llm: LLMCallable) -> None:
        """Inject or replace the LLM backend at runtime."""

        self._llm = llm

    def _is_important_segment(self, sleep_state: SleepState, replay_seq: Optional[ReplaySequence]) -> bool:
        if sleep_state.stage == SleepStage.REM:
            return True
        if replay_seq is None:
            return False
        return len(replay_seq.node_ids) >= 2

    def step(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        replay_seq: Optional[ReplaySequence],
    ) -> Optional[DreamSegment]:
        if sleep_state.stage == SleepStage.WAKE:
            self._last_time_hours = sleep_state.time_hours
            return None

        if _STAGE_ORDER[sleep_state.stage] < _STAGE_ORDER[self.config.min_stage_for_dreaming]:
            self._last_time_hours = sleep_state.time_hours
            return None

        segment = DreamSegment(
            start_time_hours=self._last_time_hours,
            end_time_hours=sleep_state.time_hours,
            stage=sleep_state.stage,
        )

        dominant_emotion = replay_seq.dominant_emotion if replay_seq is not None else EmotionLabel.NEUTRAL
        segment.dominant_emotion = dominant_emotion
        if replay_seq is not None:
            segment.active_memory_ids = replay_seq.node_ids

        use_llm = self.config.use_llm and self._llm is not None
        if self.config.important_only and not self._is_important_segment(sleep_state, replay_seq):
            use_llm = False

        if use_llm:
            segment.narrative = self._build_narrative_llm(sleep_state, neuro_state, segment)
            segment.scene_description = self._build_scene_description_llm(sleep_state, segment)
        else:
            segment.narrative = self._build_narrative_template(sleep_state, neuro_state, segment)
            segment.scene_description = self._build_scene_description_template(sleep_state, segment)

        segment.bizarreness_score = self._estimate_bizarreness(segment)

        self._emit_event(segment)
        self._last_time_hours = sleep_state.time_hours
        return segment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_narrative_template(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        segment: DreamSegment,
    ) -> str:
        emotion = segment.dominant_emotion.value
        return (
            f"You find yourself in a dream during {sleep_state.stage.value} sleep. "
            f"The emotional tone feels {emotion}. Time flows strangely as the night progresses."
        )

    def _build_scene_description_template(self, sleep_state: SleepState, segment: DreamSegment) -> str:
        return f"A vignette from {sleep_state.stage.value} sleep, colored by {segment.dominant_emotion.value} emotion."

    def _build_narrative_llm(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        segment: DreamSegment,
    ) -> str:
        assert self._llm is not None
        prompt = (
            "You are a dream narrator. Given the current sleep stage, neuromodulator levels, "
            "and a list of activated memories, write a short first-person dream description.\n\n"
            f"Sleep stage: {sleep_state.stage.value}\n"
            f"ACh level (relative): {neuro_state.ach:.2f}\n"
            f"Serotonin level (relative): {neuro_state.serotonin:.2f}\n"
            f"Noradrenaline level (relative): {neuro_state.ne:.2f}\n"
            f"Cortisol level (relative): {neuro_state.cortisol:.2f}\n"
            f"Dominant emotion: {segment.dominant_emotion.value}\n"
            f"Active memory fragment IDs: {segment.active_memory_ids}\n\n"
            "Write 2–4 sentences in the first person, vivid but concise."
        )
        return self._llm(prompt)

    def _build_scene_description_llm(self, sleep_state: SleepState, segment: DreamSegment) -> str:
        assert self._llm is not None
        prompt = (
            "You are a visual scene summarizer. Given a dream segment, return a short, "
            "third-person scene description in one sentence. Focus on setting and mood.\n\n"
            f"Sleep stage: {sleep_state.stage.value}\n"
            f"Dominant emotion: {segment.dominant_emotion.value}\n"
            f"Existing narrative (if any): {segment.narrative}\n"
        )
        return self._llm(prompt)

    def _estimate_bizarreness(self, segment: DreamSegment) -> float:
        n = len(segment.active_memory_ids)
        # Heuristic: more distinct memories, especially with conflicting emotions,
        # produce more bizarre content. Cap at 1.0.
        base = 0.1 * n
        if n >= 2:
            base += 0.1
        return min(1.0, base)

    def _emit_event(self, segment: DreamSegment) -> None:
        event = Event(
            type=EventType.DREAM_SEGMENT_GENERATED,
            payload={"segment_id": segment.id},
            timestamp_hours=segment.end_time_hours,
        )
        self.event_bus.publish(event)
