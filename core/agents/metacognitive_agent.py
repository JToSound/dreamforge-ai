from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import random
import numpy as np

from core.models.dream_segment import DreamSegment
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState, SleepStage
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class MetacognitiveConfig:
    """Configuration for lucidity model and metacognitive tracking."""

    # Training level (0.0–1.0) models user-level induction practice (MILD/WILD)
    # Set a small default so some lucid events occur in validation runs.
    lucidity_training_level: float = 0.25

    # Window (in segments) to consider for recent reality-check failures
    rc_window: int = 5


class MetacognitiveAgent:
    """Estimates lucidity probability using a multi-factor gated model.

    Implements a rolling "reality-check failures" counter that increases when
    high-bizarreness REM segments do not produce lucidity. The counter is
    exposed to the lucidity model as `reality_check_failures_recent`.
    """

    def __init__(self, event_bus: Optional[EventBus] = None, config: Optional[MetacognitiveConfig] = None) -> None:
        self.event_bus = event_bus or EventBus()
        self.config = config or MetacognitiveConfig()

        # Rolling history of recent reality-check failures (True/False)
        self._rc_history = deque(maxlen=self.config.rc_window)

        # Consecutive failures counter (for burst-sensitive dynamics)
        self.consecutive_rc_failures = 0

    # --- Utility functions -------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return float(1.0 / (1.0 + np.exp(-x)))
        except Exception:
            return 0.0

    def compute_lucidity_probability(
        self,
        stage: SleepStage,
        ach_level: float,
        cortisol_level: float,
        bizarreness_score: float,
        time_in_night_hours: float,
        reality_check_failures_recent: int,
        lucidity_training_level: float = 0.0,
    ) -> float:
        """Multi-factor lucidity probability model.

        Stage gate: lucidity is near-impossible outside REM.
        Neurochemical gate: requires high ACh (>0.65) + moderate cortisol (~0.5).
        Temporal bias: late-night REM increases base probability.
        Bizarreness is a weak trigger (~20% weight). History and training
        contribute additional weight.

        Scientific citations:
        - Stage gating & lucidity: Voss et al. (2009), Sleep 32(9):1191–1200
        - ACh dependency: Hobson (2009), Nature Reviews Neuroscience 10:803–813
        """
        # Stage gate — low baseline outside REM (small chance)
        if stage != SleepStage.REM:
            return float(np.clip(0.04 * random.gauss(1.0, 0.20), 0.0, 1.0))

        # Neurochemical gate (keep as a soft influence rather than hard multiplier)
        ach_gate = self._sigmoid((ach_level - 0.65) * 10.0)
        cort_gate = 1.0 - abs(cortisol_level - 0.50) * 2.0
        neuro_factor = ach_gate * max(0.0, cort_gate)
        # map neuro_factor into a milder scaling range [0.6, 1.0]
        neuro_scale = 0.6 + 0.4 * float(np.clip(neuro_factor, 0.0, 1.0))

        # Temporal bias (late-night REM more prone to lucidity)
        temporal_factor = 0.5 + 0.5 * min(1.0, time_in_night_hours / 6.0)

        # Weak bizarreness trigger — keep low so lucidity remains decoupled
        bizarreness_trigger = 0.10 * float(np.clip(bizarreness_score, 0.0, 1.0))

        # Reality-check failure accumulation (moderate weight)
        rc_factor = 0.30 * min(1.0, reality_check_failures_recent / 5.0)

        # Training level (strong influence; default raised above)
        training_factor = 0.45 * float(np.clip(lucidity_training_level, 0.0, 1.0))

        # Base probability increases modestly across the night
        base = 0.06 + 0.03 * min(1.0, time_in_night_hours / 6.0)

        # Combine factors, using neuro_scale to avoid hard suppression when neurochemistry is moderate
        raw = (base + bizarreness_trigger + rc_factor + training_factor) * temporal_factor * neuro_scale

        # Time-dependent stochastic bursts (more likely late-night)
        burst_prob = 0.06 + 0.14 * min(1.0, time_in_night_hours / 6.0)
        if random.random() < float(np.clip(burst_prob, 0.0, 1.0)):
            raw = min(1.0, raw + abs(random.gauss(0.50, 0.15)))

        return float(np.clip(raw, 0.0, 1.0))

    # --- Public interface --------------------------------------------------
    def update_for_segment(
        self,
        segment: DreamSegment,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
    ) -> None:
        """Update `segment.lucidity_probability` using the multi-factor model
        and maintain reality-check failure tracking.
        """
        # Determine recent failure count (number of True in history)
        recent_failures = int(sum(1 for v in self._rc_history if v))

        # Compute lucidity
        try:
            time_hours = float(getattr(segment, "time_hours", getattr(segment, "start_time_hours", 0.0)))
        except Exception:
            time_hours = 0.0

        lucidity = self.compute_lucidity_probability(
            stage=sleep_state.stage,
            ach_level=float(getattr(neuro_state, "ach", 0.0)),
            cortisol_level=float(getattr(neuro_state, "cortisol", 0.0)),
            bizarreness_score=float(getattr(segment, "bizarreness_score", 0.0)),
            time_in_night_hours=float(time_hours),
            reality_check_failures_recent=recent_failures,
            lucidity_training_level=self.config.lucidity_training_level,
        )

        # Assign back to the segment (works for dict-like or pydantic models)
        try:
            setattr(segment, "lucidity_probability", float(lucidity))
        except Exception:
            try:
                segment["lucidity_probability"] = float(lucidity)
            except Exception:
                pass

        # Reality-check failure detection: increment if high biz + REM without lucidity
        high_biz = float(getattr(segment, "bizarreness_score", 0.0)) > 0.85
        lucidity_threshold = 0.35
        failure = (sleep_state.stage == SleepStage.REM) and high_biz and (lucidity < lucidity_threshold)

        if failure:
            self.consecutive_rc_failures += 1
            self._rc_history.append(True)
        else:
            # reset consecutive on success or non-applicable segment
            self.consecutive_rc_failures = 0
            self._rc_history.append(False)

        # Publish event
        event = Event(
            type=EventType.LUCIDITY_UPDATED,
            payload={"segment_id": getattr(segment, "id", None), "lucidity_probability": float(lucidity)},
            timestamp_hours=time_hours,
        )
        try:
            self.event_bus.publish(event)
        except Exception:
            pass

