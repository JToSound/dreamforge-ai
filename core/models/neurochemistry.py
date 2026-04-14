from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from scipy.integrate import solve_ivp

from core.models.sleep_cycle import SleepStage


class Neurotransmitter(str, Enum):
    ACh = "ACh"
    SEROTONIN = "5HT"
    NE = "NE"
    CORTISOL = "CORTISOL"


class NeurochemistryParameters(BaseModel):
    """Parameters controlling neuromodulator and cortisol dynamics.

    Dynamics are stage dependent and qualitatively consistent with the
    monoamine hypothesis of REM sleep: high cholinergic activation with
    near-silencing of serotonin and noradrenaline during REM, plus a
    circadian/stress-modulated cortisol curve. [cite:24][cite:27]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basal production rates (per hour)
    r_ach_wake: float = Field(1.0, description="Basal ACh production rate during wake.")
    r_ach_nrem: float = Field(0.5, description="Basal ACh production rate during NREM.")
    r_ach_rem: float = Field(1.5, description="Basal ACh production rate during REM.")
    # REM-specific scaling factor (easy tuning knob to reduce REM ACh magnitude)
    ach_rem_scale: float = Field(
        0.75, description="Multiplier applied to REM ACh production."
    )

    r_5ht_wake: float = Field(1.0, description="Basal 5-HT production during wake.")
    r_5ht_nrem: float = Field(0.3, description="Basal 5-HT production during NREM.")
    r_5ht_rem: float = Field(
        0.05, description="Basal 5-HT production during REM (near silent)."
    )

    r_ne_wake: float = Field(1.0, description="Basal NE production during wake.")
    r_ne_nrem: float = Field(0.3, description="Basal NE production during NREM.")
    r_ne_rem: float = Field(
        0.05, description="Basal NE production during REM (near silent)."
    )

    # Clearance constants (per hour)
    k_clear_ach: float = Field(1.2, description="Clearance rate for ACh.")
    k_clear_5ht: float = Field(0.8, description="Clearance rate for 5-HT.")
    k_clear_ne: float = Field(0.8, description="Clearance rate for NE.")

    # ACh saturation / upper bound (prevents runaway ACh during long REM)
    ach_max: float = Field(1.0, description="Upper bound for ACh level (saturation).")
    ach_saturating: bool = Field(
        True, description="Whether to use saturating production for ACh."
    )

    # Cortisol circadian-like parameters (simplified)
    cortisol_baseline: float = Field(0.5, description="Baseline cortisol level.")
    cortisol_amplitude: float = Field(
        0.9, description="Amplitude of circadian cortisol oscillation."
    )
    cortisol_phase: float = Field(
        6.0,
        description="(Deprecated) Legacy cortisol phase; use cortisol_rise_time + cortisol_k for asymmetric rise.",
    )
    cortisol_clear: float = Field(0.3, description="Clearance rate for cortisol.")
    # Asymmetric cortisol shape controls (sigma for rise vs fall)
    cortisol_rise_sigma: float = Field(
        0.7, description="Width of cortisol rise (hours)."
    )
    cortisol_fall_sigma: float = Field(
        3.0, description="Width of cortisol fall (hours)."
    )

    # Asymmetric sigmoid-based cortisol rise parameters
    cortisol_rise_time: float = Field(
        5.5,
        description=(
            "Hour (since sleep onset) of the cortisol peak in the asymmetric "
            "sigmoid profile."
        ),
    )
    # Separate steepness parameters for the rising and falling limbs to allow
    # an asymmetric profile (steeper morning rise, slower decline).
    cortisol_k_rise: float = Field(
        3.0,
        description=(
            "Steepness of cortisol rise (sigmoid) during the morning increase; "
            "larger -> steeper rise."
        ),
    )
    cortisol_k_fall: float = Field(
        1.0,
        description=(
            "Steepness of cortisol decline (sigmoid) after the peak; smaller -> slower fall."
        ),
    )

    # Pharmacological modifiers (e.g., SSRIs)
    ssri_factor: float = Field(
        1.0,
        description="Multiplier on effective 5-HT level due to SSRI; >1 for increased availability.",
    )

    # Noise level to simulate biological variability
    noise_std: float = Field(
        0.05, ge=0.0, description="Std of Gaussian noise added to derivatives."
    )


@dataclass
class NeurochemistryState:
    """State of key neuromodulators at a given time (relative units)."""

    time_hours: float
    ach: float
    serotonin: float
    ne: float
    cortisol: float


class NeurochemistryModel:
    """Stage-conditioned ODE model for neuromodulator dynamics during sleep.

    The model is deliberately simple but qualitatively consistent with:
      - High cortical ACh and low monoamines in REM. [cite:24]
      - Near-silencing of noradrenergic neurons during REM. [cite:27]
      - Circadian and stress-related modulation of cortisol.
    """

    def __init__(self, params: Optional[NeurochemistryParameters] = None) -> None:
        self.params = params or NeurochemistryParameters()

    # ------------------------------------------------------------------
    # Stage-dependent production rates
    # ------------------------------------------------------------------

    def _prod_rates(self, stage: SleepStage) -> tuple[float, float, float]:
        p = self.params
        if stage == SleepStage.REM:
            return p.r_ach_rem * p.ach_rem_scale, p.r_5ht_rem, p.r_ne_rem
        elif stage in (SleepStage.N1, SleepStage.N2, SleepStage.N3):
            return p.r_ach_nrem, p.r_5ht_nrem, p.r_ne_nrem
        else:  # WAKE
            return p.r_ach_wake, p.r_5ht_wake, p.r_ne_wake

    def _cortisol_drive(self, time_hours: float) -> float:
        """Asymmetric sigmoid cortisol profile with a fixed peak time.

        The curve is piecewise:
        - Before peak: a rising sigmoid normalized to reach 1.0 at peak.
        - After peak: a decaying sigmoid normalized to start at 1.0 at peak.

        This guarantees the maximum occurs at `cortisol_rise_time` while allowing
        independently tunable rise/fall steepness.
        """
        p = self.params
        peak = float(p.cortisol_rise_time)
        t = float(time_hours)

        if t <= peak:
            z = -float(p.cortisol_k_rise) * (t - peak)
        else:
            z = float(p.cortisol_k_fall) * (t - peak)

        if z > 700:
            raw_sig = 0.0
        elif z < -700:
            raw_sig = 1.0
        else:
            raw_sig = 1.0 / (1.0 + np.exp(z))

        sig = min(1.0, 2.0 * raw_sig)
        return float(p.cortisol_baseline + p.cortisol_amplitude * sig)

    # ------------------------------------------------------------------
    # Staged integrator
    # ------------------------------------------------------------------

    def integrate_staged(
        self,
        state: NeurochemistryState,
        stage_durations: List[Tuple[SleepStage, float]],
    ) -> List[NeurochemistryState]:
        """Integrate neurochemical ODEs across a sequence of sleep stages.

        Each (stage, duration_hours) tuple is integrated independently with the
        stage held constant. This avoids missed stage transitions due to coarse
        solver stepping and ensures production/clearance rates switch exactly
        at stage boundaries.

        Args:
            state: initial NeurochemistryState at start of the first segment.
            stage_durations: list of tuples (SleepStage, duration_hours).

        Returns:
            A list of NeurochemistryState samples (including the initial state
            and intermediate samples) covering the entire staged integration.

        Notes:
            - Uses `max_step=1/120` hours (30 seconds) per segment to capture
              fast transitions accurately.
            - This method was introduced to fix issues where a single long
              solve_ivp call ignored stage transitions due to sparse internal
              time steps.
        """
        trajectory: List[NeurochemistryState] = []
        current_time = float(state.time_hours)
        y_current = np.array(
            [state.ach, state.serotonin, state.ne, state.cortisol], dtype=float
        )

        # small epsilon for time comparisons
        eps = 1e-9

        for stage, duration in stage_durations:
            t0 = current_time
            t1 = t0 + float(duration)

            # constant stage function for this segment
            def const_stage_fn(_t: float, s: SleepStage = stage) -> SleepStage:
                return s

            sol = solve_ivp(
                fun=lambda t, y: self._ode_system(t, y, stage_fn=const_stage_fn),
                t_span=(t0, t1),
                y0=y_current,
                max_step=1.0 / 120.0,
                dense_output=False,
            )

            if not sol.success:
                raise RuntimeError(
                    f"ODE solver failed for stage {stage} from {t0} to {t1}: {sol.message}"
                )

            t_samples = sol.t
            y_samples = sol.y.T

            # append samples, avoiding duplicated boundary time
            for ti, yi in zip(t_samples, y_samples):
                if trajectory and abs(ti - current_time) < eps:
                    # skip duplicate boundary sample
                    continue
                trajectory.append(
                    NeurochemistryState(
                        time_hours=float(ti),
                        ach=float(yi[0]),
                        serotonin=float(yi[1] * self.params.ssri_factor),
                        ne=float(yi[2]),
                        cortisol=float(yi[3]),
                    )
                )

            # update for next segment
            current_time = float(t_samples[-1])
            y_current = y_samples[-1]

        return trajectory

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _ode_system(
        self,
        t: float,
        y: np.ndarray,
        stage_fn: Callable[[float], SleepStage],
    ) -> np.ndarray:
        p = self.params
        ach, five_ht, ne, cort = y
        stage = stage_fn(t)
        r_ach, r_5ht, r_ne = self._prod_rates(stage)

        # Apply optional saturating production for ACh to prevent excessive REM values
        if p.ach_saturating and p.ach_max > 0:
            d_ach = r_ach * (1.0 - ach / p.ach_max) - p.k_clear_ach * ach
        else:
            d_ach = r_ach - p.k_clear_ach * ach
        d_5ht = r_5ht - p.k_clear_5ht * five_ht
        d_ne = r_ne - p.k_clear_ne * ne

        cort_drive = self._cortisol_drive(t)
        d_cort = cort_drive - p.cortisol_clear * cort

        if p.noise_std > 0.0:
            noise = np.random.normal(0.0, p.noise_std, size=4)
        else:
            noise = np.zeros(4)

        return np.array([d_ach, d_5ht, d_ne, d_cort]) + noise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initial_state(
        self,
        time_hours: float = 0.0,
        ach: float = 0.8,
        serotonin: float = 0.8,
        ne: float = 0.8,
        cortisol: float = 0.5,
    ) -> NeurochemistryState:
        return NeurochemistryState(
            time_hours=time_hours,
            ach=ach,
            serotonin=serotonin,
            ne=ne,
            cortisol=cortisol,
        )

    def integrate(
        self,
        state: NeurochemistryState,
        t_end: float,
        stage_fn: Callable[[float], SleepStage],
        max_step: float = 1.0 / 60.0,
    ) -> tuple[list[NeurochemistryState], np.ndarray]:
        """Integrate from state.time_hours to t_end and return trajectory.

        Args:
            state: Initial NeurochemistryState.
            t_end: Final time in hours.
            stage_fn: Function mapping time -> sleep stage.
            max_step: Max integration step (hours), e.g. 1/60 for 1 minute.
        """
        y0 = np.array(
            [state.ach, state.serotonin, state.ne, state.cortisol], dtype=float
        )

        sol = solve_ivp(
            fun=lambda t, y: self._ode_system(t, y, stage_fn=stage_fn),
            t_span=(state.time_hours, t_end),
            y0=y0,
            max_step=max_step,
            dense_output=False,
        )

        t_samples = sol.t
        y_samples = sol.y.T

        trajectory: list[NeurochemistryState] = []
        for t, y in zip(t_samples, y_samples):
            trajectory.append(
                NeurochemistryState(
                    time_hours=float(t),
                    ach=float(y[0]),
                    serotonin=float(y[1] * self.params.ssri_factor),
                    ne=float(y[2]),
                    cortisol=float(y[3]),
                )
            )

        return trajectory, t_samples
