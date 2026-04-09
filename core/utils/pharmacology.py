from __future__ import annotations

from dataclasses import dataclass

from core.models.neurochemistry import NeurochemistryParameters


@dataclass
class PharmacologyProfile:
    """Simplified pharmacology configuration.

    This is intentionally coarse; it captures only high-level effects.
    """

    ssri_strength: float = 1.0  # >1.0 increases effective 5-HT
    stress_level: float = 0.0   # 0–1, increases cortisol amplitude


def apply_pharmacology(params: NeurochemistryParameters, profile: PharmacologyProfile) -> NeurochemistryParameters:
    """Return a new NeurochemistryParameters with pharmacology applied."""

    new_params = params.model_copy(deep=True)
    new_params.ssri_factor = profile.ssri_strength
    new_params.cortisol_amplitude *= (1.0 + 0.5 * profile.stress_level)
    return new_params
