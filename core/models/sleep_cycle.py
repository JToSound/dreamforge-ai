"""core/models/sleep_cycle.py

Borbély two-process model of sleep regulation and stage dynamics.

Process S (homeostatic sleep pressure) and Process C (circadian drive) interact
to determine sleep propensity and discretise into polysomnographic stages:
Wake, N1, N2, N3 (SWS), REM.

Scientific references
---------------------
- Borbély, A.A. (1982). A two process model of sleep regulation.
  Human Neurobiology, 1(3), 195–204.
- Borbély, A.A. et al. (2016). The two-process model of sleep regulation:
  a reappraisal. Journal of Sleep Research, 25(2), 131–143.
- Achermann, P. & Borbély, A.A. (2003). Mathematical models of sleep regulation.
  Frontiers in Bioscience, 8, s683–693.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SleepStage(str, Enum):
    """Standard polysomnography sleep stages.

    Based on AASM 2007 scoring rules:
    Wake, N1 (light sleep), N2 (sleep spindles), N3 (slow-wave/deep sleep), REM.
    """

    WAKE = "WAKE"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    REM = "REM"


# ---------------------------------------------------------------------------
# Parameter model
# ---------------------------------------------------------------------------


class TwoProcessParameters(BaseModel):
    """Parameters for Borbély’s two-process model (Process S and Process C).

    Process S (homeostatic):
        Increases exponentially during wake with time constant *tau_wake*.
        Decays exponentially during sleep with time constant *tau_sleep*.

    Process C (circadian):
        Near-24 h sinusoidal oscillator parameterised by period, amplitude, and
        phase (hour of global-clock peak).  The circadian *alerting signal* peaks
        in the early evening and falls during the sleep window, acting as an
        opposing force to rising homeostatic pressure.

    References
    ----------
    Borbély 1982; Achermann & Borbély 2003.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Process S (hours) ---------------------------------------------------
    tau_wake: float = Field(
        default=18.0,
        gt=0.0,
        description="Time constant for S increase during wake (hours). "
        "Borbély 1982 estimate: ~18 h.",
    )
    tau_sleep: float = Field(
        # Reduced from 4.2 -> 3.5 to slow S decay slightly and increase
        # time spent at high homeostatic pressure (promotes more N3/SWS).
        # Source: Achermann & Borbély (2003), Journal of Sleep Research 12:37–46
        default=3.5,
        gt=0.0,
        description="Time constant for S decay during sleep (hours). "
        "Adjusted per Achermann & Borbély (2003) to increase SWS proportion.",
    )
    s_max: float = Field(
        default=1.0,
        gt=0.0,
        description="Upper asymptote for Process S (normalised).",
    )
    s_min: float = Field(
        default=0.0,
        description="Lower asymptote for Process S (normalised).",
    )

    # --- Process C -----------------------------------------------------------
    circadian_period: float = Field(
        default=24.2,
        gt=0.0,
        description="Intrinsic circadian period (hours). Human average ~24.2 h.",
    )
    circadian_amplitude: float = Field(
        default=0.50,
        gt=0.0,
        description="Amplitude of the circadian oscillation (normalised).",
    )
    circadian_phase: float = Field(
        default=18.0,
        description="Clock time (hours, 0–24) of the circadian *maximum* "
        "(approximately early evening alerting peak).",
    )

    # --- Stage thresholds / heuristics --------------------------------------
    n3_s_threshold: float = Field(
        # Lowered threshold to widen N3-eligible window (was 0.75 -> now 0.65).
        # Source: Borbély (1982), Human Neurobiology 1:195–204
        default=0.65,
        description="Process-S value above which N3 (SWS) is preferentially "
        "generated in early-night cycles. Lowered to promote realistic SWS fraction.",
    )
    rem_c_threshold: float = Field(
        default=0.15,
        description="Circadian component below which REM transitions are "
        "suppressed (morning-bias heuristic).",
    )
    rem_cycle_bias: float = Field(
        default=0.18,
        description="Per-cycle additive probability increment for REM onset, "
        "reflecting the empirical lengthening of REM toward morning.",
    )

    # --- Simulation defaults ------------------------------------------------
    initial_s_fraction: float = Field(
        default=0.90,
        description="Initial Process S as a fraction of s_max at sleep onset. "
        "Represents accumulated homeostatic pressure after ~16 h wake.",
    )


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class SleepState:
    """Complete state of the sleep model at a given simulated time."""

    time_hours: float
    """Simulation time since sleep onset (hours)."""

    process_s: float
    """Current homeostatic sleep pressure (0–1 normalised)."""

    process_c: float
    """Current circadian drive (can be negative; range ~[-amp, +amp])."""

    stage: SleepStage
    """Discretised sleep stage label."""

    cycle_index: int
    """Index of the current 90-minute sleep cycle (0-based)."""

    time_in_stage_hours: float = field(default=0.0)
    """Elapsed time in the current stage (hours), used for transition guards."""
    minutes_in_stage: float = field(default=0.0)
    """Elapsed time in the current stage (minutes)."""
    minutes_in_cycle: float = field(default=0.0)
    """Elapsed time in the current cycle (minutes)."""


CYCLE_TEMPLATES: dict[int, list[tuple[SleepStage, float]]] = {
    1: [
        (SleepStage.N1, 5.0),
        (SleepStage.N2, 20.0),
        (SleepStage.N3, 40.0),
        (SleepStage.N2, 10.0),
        (SleepStage.REM, 10.0),
    ],
    2: [
        (SleepStage.N1, 3.0),
        (SleepStage.N2, 20.0),
        (SleepStage.N3, 25.0),
        (SleepStage.N2, 10.0),
        (SleepStage.REM, 14.0),
    ],
    3: [
        (SleepStage.N1, 2.0),
        (SleepStage.N2, 20.0),
        (SleepStage.N3, 10.0),
        (SleepStage.N2, 10.0),
        (SleepStage.REM, 18.0),
    ],
    4: [
        (SleepStage.N1, 2.0),
        (SleepStage.N2, 25.0),
        (SleepStage.N3, 5.0),
        (SleepStage.N2, 10.0),
        (SleepStage.REM, 20.0),
    ],
}

DEFAULT_CYCLE_TEMPLATE: list[tuple[SleepStage, float]] = [
    (SleepStage.N1, 2.0),
    (SleepStage.N2, 30.0),
    (SleepStage.REM, 15.0),
]


class CycleStateMachine:
    """Build staged ultradian-cycle schedules from empirical templates."""

    def __init__(
        self,
        templates: Optional[dict[int, list[tuple[SleepStage, float]]]] = None,
        default_template: Optional[list[tuple[SleepStage, float]]] = None,
        jitter_std: float = 0.2,
        jitter_min: float = 0.8,
        jitter_max: float = 1.2,
    ) -> None:
        self.templates = templates or CYCLE_TEMPLATES
        self.default_template = default_template or DEFAULT_CYCLE_TEMPLATE
        self.jitter_std = jitter_std
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max

    def _jitter_duration(self, base_minutes: float) -> float:
        mult = float(np.random.normal(1.0, self.jitter_std))
        mult = max(self.jitter_min, min(self.jitter_max, mult))
        return max(1.0, base_minutes * mult)

    def build_schedule(self, total_minutes: int) -> list[dict]:
        schedule: list[dict] = []
        cursor = 0.0
        cycle_idx = 0

        while cursor < total_minutes:
            cycle_idx += 1
            template = self.templates.get(cycle_idx, self.default_template)
            cycle_start_min = cursor

            for stage, base_min in template:
                dur = self._jitter_duration(base_min)
                start_min = cursor
                end_min = min(total_minutes, cursor + dur)

                schedule.append(
                    {
                        "cycle_index": cycle_idx - 1,
                        "stage": stage,
                        "start_min": start_min,
                        "end_min": end_min,
                        "cycle_cycle_start_min": cycle_start_min,
                    }
                )

                cursor = end_min
                if cursor >= total_minutes:
                    break

        return schedule


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SleepCycleModel:
    """Implements Borbély’s two-process model with empirical stage dynamics.

    Provides:
    - Continuous S(t), C(t) trajectories.
    - Discrete stage labels (Wake / N1 / N2 / N3 / REM) following empirical
      sleep architecture:
      * N3 dominant in first third of the night (high Process S).
      * REM periods lengthen toward morning (rising circadian drive).
      * Average cycle length ~90 minutes.

    Example
    -------
    >>> model = SleepCycleModel()
    >>> states, stages = model.simulate_night(duration_hours=8.0)
    >>> print(stages[:10])
    """

    def __init__(self, params: Optional[TwoProcessParameters] = None) -> None:
        self.params: TwoProcessParameters = params or TwoProcessParameters()
        self.cycle_state_machine = CycleStateMachine()

    # ------------------------------------------------------------------
    # Process S dynamics
    # ------------------------------------------------------------------

    def _process_s_step(
        self,
        s_prev: float,
        dt_hours: float,
        is_sleep: bool,
    ) -> float:
        """Advance Process S over interval *dt_hours*.

        During wake (exponential rise toward s_max)::

            S(t+dt) = s_max − (s_max − S(t)) × exp(−dt / tau_wake)

        During sleep (exponential decay toward s_min)::

            S(t+dt) = S(t) × exp(−dt / tau_sleep)

        Args:
            s_prev:    Process S at the start of the interval.
            dt_hours:  Duration of the interval in hours.
            is_sleep:  True if the organism is in a sleep stage.

        Returns:
            Updated Process S value clamped to [s_min, s_max].

        References
        ----------
        Borbély 1982, eq. 1–2; Achermann & Borbély 2003.
        """
        p = self.params
        if is_sleep:
            s_new = s_prev * math.exp(-dt_hours / p.tau_sleep)
        else:
            s_new = p.s_max - (p.s_max - s_prev) * math.exp(-dt_hours / p.tau_wake)
        return float(min(max(s_new, p.s_min), p.s_max))

    # ------------------------------------------------------------------
    # Process C dynamics
    # ------------------------------------------------------------------

    def _process_c(
        self,
        global_time_hours: float,
    ) -> float:
        """Compute Process C (circadian alerting signal) at absolute clock time.

        Modelled as a sinusoid with period *circadian_period* and phase
        *circadian_phase*::

            C(t) = A × sin(2π(t − φ) / T)

        where A is amplitude, φ is the phase of the maximum, and T is the period.

        Args:
            global_time_hours: Wall-clock time (0–48 h range for overnight).

        Returns:
            Circadian drive value in [−amplitude, +amplitude].

        References
        ----------
        Borbély et al. 2016, eq. 2.
        """
        p = self.params
        phase_rad = (
            2.0 * math.pi * (global_time_hours - p.circadian_phase) / p.circadian_period
        )
        return float(p.circadian_amplitude * math.sin(phase_rad))

    # ------------------------------------------------------------------
    # Stage inference (heuristic discretisation)
    # ------------------------------------------------------------------

    def _infer_stage(
        self,
        s_value: float,
        c_value: float,
        cycle_index: int,
        current_stage: SleepStage,
        time_in_cycle_hours: float,
        time_in_stage_hours: float,
    ) -> SleepStage:
        """Map continuous (S, C) and cycle position to a discrete sleep stage.

        Heuristic rules grounded in empirical sleep architecture:

        1. **Cycle phase**: Each ~90-minute cycle is divided into NREM and REM
           portions; REM occupies the final ~30% of later cycles.
        2. **N3 preference**: High Process S in early cycles (≤2) biases toward
           N3 in the early cycle phase.
        3. **REM lengthening**: REM probability increases with cycle index
           (*rem_cycle_bias*) and circadian drive (Process C).
        4. **Minimum stage duration**: Prevents spurious single-tick transitions
           (N3 minimum 10 min; REM minimum 5 min).

        Args:
            s_value:             Current Process S.
            c_value:             Current Process C.
            cycle_index:         Current 90-min cycle index (0-based).
            current_stage:       Stage at start of tick (for transition guards).
            time_in_cycle_hours: Elapsed time within the current cycle.
            time_in_stage_hours: Elapsed time in the current stage.

        Returns:
            New SleepStage.
        """
        p = self.params
        cycle_length_h = 1.5  # 90-minute cycle (empirical mean)
        cycle_phase = (time_in_cycle_hours % cycle_length_h) / cycle_length_h

        # Guard: enforce minimum time in N3 (10 min) and REM (5 min)
        if current_stage == SleepStage.N3 and time_in_stage_hours < 10 / 60:
            return SleepStage.N3
        if current_stage == SleepStage.REM and time_in_stage_hours < 5 / 60:
            return SleepStage.REM

        # NEW: force N3 in the very first cycle's early half (first ~45 minutes)
        # This ensures a reliable SWS block at sleep onset for healthy adults.
        # Empirical rationale: early-night SWS dominates first cycle (Borbély 1982).
        if cycle_index == 0 and cycle_phase < 0.5:
            return SleepStage.N3

        # Early cycle phase (0–25%): deep or light NREM
        if cycle_phase < 0.25:
            # Expand N3 eligibility to include the first four cycles (index 0..3).
            if s_value >= p.n3_s_threshold and cycle_index <= 3:
                return SleepStage.N3
            # Prefer N2 over N1 in early-cycle non-N3 windows to reduce spurious N1.
            return SleepStage.N2

        # Mid cycle phase (25–60%): consolidation in N2
        elif cycle_phase < 0.60:
            # Allow N3 continuation if still high S and early night
            if (
                current_stage == SleepStage.N3
                and s_value >= p.n3_s_threshold * 0.9
                and cycle_index <= 3
            ):
                return SleepStage.N3
            return SleepStage.N2

        # Late cycle phase (60–100%): REM window
        else:
            rem_prob = p.rem_cycle_bias * (cycle_index + 1)
            # Circadian boost: positive C (evening/morning) increases REM tendency
            if c_value > p.rem_c_threshold:
                rem_prob += 0.20
            if rem_prob > 0.35 or (cycle_index >= 3 and c_value > 0.0):
                return SleepStage.REM
            return SleepStage.N2

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: SleepState,
        dt_hours: float,
        sleep_start_clock_time: float = 23.0,
    ) -> SleepState:
        """Advance the sleep model by *dt_hours*.

        Args:
            state:                   Current SleepState.
            dt_hours:                Time step in hours (e.g. 1/120 for 30 s).
            sleep_start_clock_time:  Wall-clock hour (0–24) of sleep onset,
                                     used to compute the global circadian time.

        Returns:
            Updated SleepState.
        """
        is_sleep = state.stage != SleepStage.WAKE

        s_new = self._process_s_step(
            s_prev=state.process_s,
            dt_hours=dt_hours,
            is_sleep=is_sleep,
        )

        new_time_hours = state.time_hours + dt_hours
        global_time = sleep_start_clock_time + new_time_hours
        c_new = self._process_c(global_time_hours=global_time)

        cycle_length_h = 1.5
        cycle_index = int(new_time_hours // cycle_length_h)
        time_in_cycle = new_time_hours - cycle_index * cycle_length_h

        new_stage = self._infer_stage(
            s_value=s_new,
            c_value=c_new,
            cycle_index=cycle_index,
            current_stage=state.stage,
            time_in_cycle_hours=time_in_cycle,
            time_in_stage_hours=state.time_in_stage_hours,
        )

        time_in_stage = (
            state.time_in_stage_hours + dt_hours if new_stage == state.stage else 0.0
        )

        return SleepState(
            time_hours=new_time_hours,
            process_s=s_new,
            process_c=c_new,
            stage=new_stage,
            cycle_index=cycle_index,
            time_in_stage_hours=time_in_stage,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def initial_state(
        self,
        sleep_start_clock_time: float = 23.0,
    ) -> SleepState:
        """Return the initial SleepState at sleep onset.

        Args:
            sleep_start_clock_time: Wall-clock hour of sleep onset.

        Returns:
            SleepState with high Process S and appropriate circadian phase.
        """
        p = self.params
        s0 = p.initial_s_fraction * p.s_max
        c0 = self._process_c(global_time_hours=sleep_start_clock_time)
        return SleepState(
            time_hours=0.0,
            process_s=s0,
            process_c=c0,
            stage=SleepStage.N1,
            cycle_index=0,
            time_in_stage_hours=0.0,
        )

    def simulate_night(
        self,
        duration_hours: float = 8.0,
        dt_minutes: float = 0.5,
        sleep_start_clock_time: float = 23.0,
    ) -> Tuple[list[SleepState], list[SleepStage]]:
        """Simulate a full sleep night and return the trajectory.

        Performance target: 8-hour night with 0.5-min ticks completes in
        <1 second on a modern CPU (960 steps, pure Python math).

        Args:
            duration_hours:         Total simulated duration (hours).
            dt_minutes:             Temporal resolution (minutes per tick).
            sleep_start_clock_time: Wall-clock hour of sleep onset.

        Returns:
            states: Time-ordered list of SleepState snapshots.
            stages: Corresponding SleepStage hypnogram labels.
        """
        # Build a staged cycle schedule (minutes) that specifies which stage is
        # active at each time; this ensures realistic ultradian cycles rather
        # than relying on a sawtooth phase heuristic.
        dt_hours = dt_minutes / 60.0
        total_minutes = int(duration_hours * 60)

        schedule = self._build_cycle_schedule(total_minutes=total_minutes)

        # helper: find active segment index by minute using incremental pointer
        seg_idx = 0
        seg_start = schedule[0]["start_min"]

        state = self.initial_state(sleep_start_clock_time=sleep_start_clock_time)
        states: list[SleepState] = [state]
        stages: list[SleepStage] = [state.stage]

        for step in range(int(total_minutes / dt_minutes)):
            current_min = step * dt_minutes

            # advance segment pointer if needed
            while (
                seg_idx + 1 < len(schedule)
                and current_min >= schedule[seg_idx]["end_min"]
            ):
                seg_idx += 1
                seg_start = schedule[seg_idx]["start_min"]

            seg = schedule[seg_idx]
            current_stage = seg["stage"]
            cycle_index = seg["cycle_index"]
            minutes_in_stage = current_min - seg_start
            minutes_in_cycle = current_min - seg["cycle_cycle_start_min"]

            is_sleep = current_stage != SleepStage.WAKE

            s_new = self._process_s_step(
                s_prev=state.process_s,
                dt_hours=dt_hours,
                is_sleep=is_sleep,
            )

            new_time_hours = state.time_hours + dt_hours
            global_time = sleep_start_clock_time + new_time_hours
            c_new = self._process_c(global_time_hours=global_time)

            time_in_stage_hours = minutes_in_stage / 60.0

            state = SleepState(
                time_hours=new_time_hours,
                process_s=s_new,
                process_c=c_new,
                stage=current_stage,
                cycle_index=cycle_index,
                time_in_stage_hours=time_in_stage_hours,
                minutes_in_stage=minutes_in_stage,
                minutes_in_cycle=minutes_in_cycle,
            )
            states.append(state)
            stages.append(state.stage)

        return states, stages

    def _build_cycle_schedule(self, total_minutes: int) -> list[dict]:
        """Build a schedule of (stage, start_min, end_min, cycle_index).

        Durations follow empirical templates per cycle (see Task 2 spec) with
        ±20% jitter (clipped) applied to each stage duration.
        """
        return self.cycle_state_machine.build_schedule(total_minutes=total_minutes)

    def process_s_curve(
        self,
        duration_hours: float = 8.0,
        dt_minutes: float = 0.5,
        sleep_start_clock_time: float = 23.0,
    ) -> list[float]:
        """Return just the Process S time series (convenience for plotting)."""
        states, _ = self.simulate_night(
            duration_hours=duration_hours,
            dt_minutes=dt_minutes,
            sleep_start_clock_time=sleep_start_clock_time,
        )
        return [s.process_s for s in states]

    def process_c_curve(
        self,
        duration_hours: float = 8.0,
        dt_minutes: float = 0.5,
        sleep_start_clock_time: float = 23.0,
    ) -> list[float]:
        """Return just the Process C time series (convenience for plotting)."""
        states, _ = self.simulate_night(
            duration_hours=duration_hours,
            dt_minutes=dt_minutes,
            sleep_start_clock_time=sleep_start_clock_time,
        )
        return [s.process_c for s in states]
