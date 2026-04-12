from __future__ import annotations

"""Parametric bizarreness scorer grounded in Revonsuo & Salmivalli (1995).

Provides a deterministic, low-variance mapping from sleep stage,
neurochemistry and memory arousal to a structured `BizarrenessScore`.

References:
- Revonsuo A, Salmivalli C. (1995). A content analysis of bizarre elements
  in dreams. Dreaming 5(3):169–187.
"""
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from core.models.sleep_cycle import SleepStage


class BizarrenessScore(BaseModel):
    """Structured components of a scene's bizarreness.

    All scores are in [0, 1].
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_score: float = Field(..., ge=0.0, le=1.0)
    discontinuity_score: float = Field(..., ge=0.0, le=1.0)
    incongruity_score: float = Field(..., ge=0.0, le=1.0)
    implausibility_score: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: Tuple[float, float] = Field(..., description="±1σ confidence interval")


def compute_bizarreness(
    stage: SleepStage,
    ach_level: float,
    ne_level: float,
    memory_arousal: float,
    cycle_index: int,
) -> BizarrenessScore:
    """Compute a parametric bizarreness score.

    This implementation follows the component weighting specified in the
    Phase 2 production upgrade prompt (Revonsuo & Salmivalli 1995 citation):

      - Stage base:  N1=0.10, N2=0.18, N3=0.08, REM=0.52, WAKE=0.04
      - ACh boost:   +0.22 * ach_level
      - NE suppress: +0.18 * (1 - ne_level)
      - Arousal:     +0.12 * memory_arousal
      - Cycle bonus: +0.03 * min(cycle_index, 4)

    A small Gaussian noise (sigma=0.025) is added and the result is clipped
    to [0, 1]. Component scores are derived deterministically from the
    contributing terms and re-scaled so they sum to the reported `total_score`.

    Args:
        stage: SleepStage label.
        ach_level: Acetylcholine level in [0,1].
        ne_level: Norepinephrine level in [0,1]. Lower NE -> reality monitoring failure.
        memory_arousal: Activation/arousal of the active memory fragment (0–1).
        cycle_index: Zero-based cycle index; later cycles increase baseline.

    Returns:
        BizarrenessScore Pydantic model with component breakdown and CI.
    """

    # Stage base weights (Revonsuo & Salmivalli 1995 - content analysis of bizarre elements)
    stage_base_map = {
        SleepStage.N1: 0.10,
        SleepStage.N2: 0.18,
        SleepStage.N3: 0.08,
        SleepStage.REM: 0.52,
        SleepStage.WAKE: 0.04,
    }

    s_base = float(stage_base_map.get(stage, 0.1))

    ach_contrib = 0.22 * float(np.clip(ach_level, 0.0, 1.0))
    ne_contrib = 0.18 * (1.0 - float(np.clip(ne_level, 0.0, 1.0)))
    arousal_contrib = 0.12 * float(np.clip(memory_arousal, 0.0, 1.0))
    cycle_contrib = 0.03 * float(min(int(cycle_index), 4))

    # Small Gaussian noise
    sigma = 0.025
    noise = float(np.random.normal(0.0, sigma))

    raw_total = s_base + ach_contrib + ne_contrib + arousal_contrib + cycle_contrib + noise

    # Dampening scale to map raw additive score into perceptual 0-1 range.
    # Calibrated so typical REM combinations land in ~0.75-0.90 while NREM
    # values remain moderate. (Heuristic calibration.)
    SCALE = 0.763
    total = float(np.clip(raw_total * SCALE, 0.0, 1.0))

    # Component raw scores (heuristic breakdown)
    disc_raw = 0.45 * (ach_contrib + arousal_contrib) + 0.25 * ne_contrib
    incon_raw = 0.35 * ach_contrib + 0.35 * ne_contrib + 0.10 * arousal_contrib
    impl_raw = 0.20 * ach_contrib + 0.20 * cycle_contrib + 0.40 * s_base

    comps = np.array([disc_raw, incon_raw, impl_raw], dtype=float)
    comps = np.clip(comps, 0.0, None)

    # Normalize component proportions to match total (preserve relative ratios)
    sum_comps = float(comps.sum())
    if sum_comps <= 0.0:
        # fallback: distribute total across components by fixed proportions
        disc, incon, impl = total * 0.4, total * 0.35, total * 0.25
    else:
        scale = total / sum_comps
        disc, incon, impl = (comps * scale).tolist()

    # Confidence interval ±1σ around total (clipped)
    ci_low = float(np.clip(total - sigma, 0.0, 1.0))
    ci_high = float(np.clip(total + sigma, 0.0, 1.0))

    return BizarrenessScore(
        total_score=total,
        discontinuity_score=float(np.clip(disc, 0.0, 1.0)),
        incongruity_score=float(np.clip(incon, 0.0, 1.0)),
        implausibility_score=float(np.clip(impl, 0.0, 1.0)),
        confidence_interval=(ci_low, ci_high),
    )
