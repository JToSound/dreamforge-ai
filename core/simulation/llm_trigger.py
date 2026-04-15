from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.models.sleep_cycle import SleepStage


class LLMTriggerType(str, Enum):
    REM_EPISODE_ONSET = "rem_episode_onset"
    LUCIDITY_THRESHOLD = "lucidity_threshold"
    BIZARRENESS_SPIKE = "bizarreness_spike"
    MEMORY_REPLAY = "memory_replay"
    NIGHT_END_REPORT = "night_end_report"


@dataclass
class LLMTrigger:
    trigger_type: LLMTriggerType
    time_hours: float
    context: dict[str, object]
    priority: int = 3


@dataclass
class LLMTriggerDetector:
    """Detect semantically meaningful moments that warrant LLM invocation."""

    lucidity_threshold: float = 0.6
    bizarreness_threshold: float = 0.95
    memory_replay_salience_threshold: float = 0.8
    _prev_stage: Optional[SleepStage] = None
    _lucidity_fired_this_rem: bool = False
    _last_biz_spike_hour: Optional[float] = None
    _biz_spike_cooldown_hours: float = 0.25

    def detect(
        self,
        time_hours: float,
        stage: SleepStage,
        bizarreness_score: float,
        lucidity_probability: float,
        memory_replay_occurred: bool,
        neurochemistry: dict[str, float],
        memory_fragments: list[dict[str, object]],
    ) -> Optional[LLMTrigger]:
        trigger: Optional[LLMTrigger] = None

        # Track REM transitions.
        entering_rem = stage == SleepStage.REM and self._prev_stage != SleepStage.REM
        leaving_rem = stage != SleepStage.REM and self._prev_stage == SleepStage.REM
        if leaving_rem:
            self._lucidity_fired_this_rem = False

        if entering_rem:
            self._lucidity_fired_this_rem = False
            trigger = LLMTrigger(
                trigger_type=LLMTriggerType.REM_EPISODE_ONSET,
                time_hours=float(time_hours),
                priority=1,
                context={
                    "stage": stage.value,
                    "bizarreness_score": float(bizarreness_score),
                    "lucidity_probability": float(lucidity_probability),
                    "neurochemistry": neurochemistry,
                },
            )
        elif (
            stage == SleepStage.REM
            and not self._lucidity_fired_this_rem
            and float(lucidity_probability) >= self.lucidity_threshold
        ):
            self._lucidity_fired_this_rem = True
            trigger = LLMTrigger(
                trigger_type=LLMTriggerType.LUCIDITY_THRESHOLD,
                time_hours=float(time_hours),
                priority=1,
                context={
                    "stage": stage.value,
                    "bizarreness_score": float(bizarreness_score),
                    "lucidity_probability": float(lucidity_probability),
                    "neurochemistry": neurochemistry,
                },
            )
        elif float(bizarreness_score) >= self.bizarreness_threshold:
            if self._last_biz_spike_hour is None or (
                float(time_hours) - self._last_biz_spike_hour
                >= self._biz_spike_cooldown_hours
            ):
                self._last_biz_spike_hour = float(time_hours)
                trigger = LLMTrigger(
                    trigger_type=LLMTriggerType.BIZARRENESS_SPIKE,
                    time_hours=float(time_hours),
                    priority=2,
                    context={
                        "stage": stage.value,
                        "bizarreness_score": float(bizarreness_score),
                        "lucidity_probability": float(lucidity_probability),
                        "neurochemistry": neurochemistry,
                    },
                )
        elif memory_replay_occurred:

            def _salience_value(fragment: dict[str, object]) -> float:
                raw = fragment.get("salience", 0.0)
                if isinstance(raw, (int, float, str, bytes)):
                    return float(raw)
                return 0.0

            has_high_salience = any(
                _salience_value(m) >= self.memory_replay_salience_threshold
                for m in memory_fragments
            )
            if has_high_salience or not memory_fragments:
                trigger = LLMTrigger(
                    trigger_type=LLMTriggerType.MEMORY_REPLAY,
                    time_hours=float(time_hours),
                    priority=3,
                    context={
                        "stage": stage.value,
                        "bizarreness_score": float(bizarreness_score),
                        "lucidity_probability": float(lucidity_probability),
                        "neurochemistry": neurochemistry,
                        "memory_fragments": memory_fragments,
                    },
                )

        self._prev_stage = stage
        return trigger
