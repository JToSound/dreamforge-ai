from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


class SleepStage(str, Enum):
    """Discrete sleep stages used by the simulation.

    Based on standard polysomnography staging: Wake, N1, N2, N3 (SWS), REM.
    """

    WAKE = "WAKE"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    REM = "REM"


class TwoProcessParameters(BaseModel):
    """Parameters for Borbély's two-process model (Process S and Process C).

    Process S (homeostatic drive) increases during wake with time constant
    ``tau_wake`` and decays during sleep with ``tau_sleep``. Process C
    (circadian) is modeled as a sinusoidal oscillator with period ~24 h and
    configurable phase and amplitude. [cite:20]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Process S parameters (hours)
    tau_wake: float = Field(default=18.0, gt=0.0, description="Time constant for S increase during wake (hours).")
    tau_sleep: float = Field(default=4.5, gt=0.0, description="Time constant for S decay during sleep (hours).")

    s_max: float = Field(default=1.0, gt=0.0, description="Upper asymptote for Process S.")
    s_min: float = Field(default=0.0, description="Lower asymptote for Process S.")

    # Process C parameters
    circadian_period: float = Field(default=24.2, gt=0.0, description="Circadian period (hours).")
    circadian_amplitude: float = Field(default=0.5, gt=0.0, description="Amplitude of Process C.")
    circadian_phase: float = Field(
        default=18.0,
        description="Phase (hours) of the circadian maximum (e.g., ~18 h for early evening peak).",
    )

    # Heuristic thresholds for stage discretization; these can be refined.
    rem_bias: float = Field(
        default=0.15,
        description="Bias added to REM probability when Process C is high and S is moderate.",
    )
    n3_threshold: float = Field(
        default=0.8,
        description="Threshold of Process S above which deep N3 is favored in early night.",
    )


@dataclass
class SleepState:
    """State variables for the sleep model at a given simulated time."""

    time_hours: float            # Simulation time since start of night, in hours
    process_s: float             # Homeostatic sleep pressure
    process_c: float             # Circadian drive
    stage: SleepStage            # Current sleep stage
    cycle_index: int             # Index of current sleep cycle (0-based)


class SleepCycleModel:
    """Implements the two-process model of sleep regulation and stage dynamics.

    Process S (homeostatic) and Process C (circadian) interact to determine
    sleep propensity. This model provides continuous S(t), C(t) trajectories
    and discrete stage labels (Wake / N1 / N2 / N3 / REM). [cite:20]
    """

    def __init__(self, params: Optional[TwoProcessParameters] = None) -> None:
        self.params = params or TwoProcessParameters()
        self._initial_s = 0.9 * self.params.s_max

    def initial_state(self, sleep_start_circadian_time: float = 23.0) -> SleepState:
        """Return initial state at sleep onset.

        Args:
            sleep_start_circadian_time: Clock time (hours, 0–24) at which sleep starts.
        """
        c0 = self._process_c_global_time(global_time_hours=sleep_start_circadian_time)
        return SleepState(
            time_hours=0.0,
            process_s=self._initial_s,
            process_c=c0,
            stage=SleepStage.N1,
            cycle_index=0,
        )

    # ------------------------------------------------------------------
    # Process S and C dynamics
    # ------------------------------------------------------------------

    def _process_s(self, s_prev: float, dt_hours: float, is_sleep: bool) -> float:
        """Update Process S over interval dt given wake/sleep state. [cite:20]

        During wake:
            S(t+dt) = s_max - (s_max - S(t)) * exp(-dt / tau_wake)

        During sleep:
            S(t+dt) = S(t) * exp(-dt / tau_sleep)
        """
        p = self.params
        if is_sleep:
            s = s_prev * math.exp(-dt_hours / p.tau_sleep)
        else:
            s = p.s_max - (p.s_max - s_prev) * math.exp(-dt_hours / p.tau_wake)
        return min(max(s, p.s_min), p.s_max)

    def _process_c_global_time(self, global_time_hours: float) -> float:
        """Circadian component as sinusoid over absolute time. [cite:20]

        Process C is a near-24 h oscillator with phase and amplitude.
        """
        p = self.params
        phase = 2.0 * math.pi * (global_time_hours - p.circadian_phase) / p.circadian_period
        return p.circadian_amplitude * math.sin(phase)

    def _process_c(self, current_state: SleepState, dt_hours: float, sleep_start_clock_time: float) -> float:
        global_time = sleep_start_clock_time + current_state.time_hours + dt_hours
        return self._process_c_global_time(global_time_hours=global_time)

    # ------------------------------------------------------------------
    # Stage dynamics and cycle structure
    # ------------------------------------------------------------------

    def _infer_stage(
        self,
        s_value: float,
        c_value: float,
        cycle_index: int,
        current_stage: SleepStage,
        time_in_cycle_hours: float,
    ) -> SleepStage:
        """Heuristic mapping from S, C, and cycle structure to discrete sleep stage.

        Empirical constraints:
          - N3 dominant early in the night when S is high.
          - REM periods lengthen toward morning, aligned with circadian peaks.
          - Cycles are ~90 minutes on average. [cite:18]
        """
        cycle_length_h = 1.5
        cycle_phase = (time_in_cycle_hours % cycle_length_h) / cycle_length_h

        high_s = s_value >= self.params.n3_threshold
        high_c = c_value >= 0.2

        if cycle_phase < 0.25:
            if high_s and cycle_index <= 2:
                return SleepStage.N3
            return SleepStage.N2
        elif cycle_phase < 0.6:
            return SleepStage.N2
        else:
            rem_prob_bias = self.params.rem_bias + 0.2 * cycle_index
            if high_c or rem_prob_bias > 0.3:
                return SleepStage.REM
            return SleepStage.N2

    def step(
        self,
        state: SleepState,
        dt_hours: float,
        sleep_start_clock_time: float = 23.0,
    ) -> SleepState:
        """Advance the sleep model by dt_hours and return the new state."""
        is_sleep = state.stage != SleepStage.WAKE

        s_new = self._process_s(state.process_s, dt_hours=dt_hours, is_sleep=is_sleep)
        c_new = self._process_c(state, dt_hours=dt_hours, sleep_start_clock_time=sleep_start_clock_time)

        cycle_length_h = 1.5
        total_time = state.time_hours + dt_hours
        cycle_index = int(total_time // cycle_length_h)
        time_in_cycle = total_time - cycle_index * cycle_length_h

        new_stage = self._infer_stage(
            s_value=s_new,
            c_value=c_new,
            cycle_index=cycle_index,
            current_stage=state.stage,
            time_in_cycle_hours=time_in_cycle,
        )

        return SleepState(
            time_hours=total_time,
            process_s=s_new,
            process_c=c_new,
            stage=new_stage,
            cycle_index=cycle_index,
        )

    def simulate_night(
        self,
        duration_hours: float = 8.0,
        dt_minutes: float = 0.5,
        sleep_start_clock_time: float = 23.0,
    ) -> Tuple[list[SleepState], list[SleepStage]]:
        """Simulate an entire night and return state trajectory and hypnogram."""
        dt_hours = dt_minutes / 60.0
        num_steps = int(duration_hours / dt_hours)

        state = self.initial_state(sleep_start_circadian_time=sleep_start_clock_time)
        states: list[SleepState] = [state]
        stages: list[SleepStage] = [state.stage]

        for _ in range(num_steps):
            state = self.step(state, dt_hours=dt_hours, sleep_start_clock_time=sleep_start_clock_time)
            states.append(state)
            stages.append(state.stage)

        return states, stages
