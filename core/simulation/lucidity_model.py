from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_yaml = importlib.import_module("yaml")


def _load_settings() -> dict[str, Any]:
    settings_path = Path("settings.yaml")
    if not settings_path.exists():
        return {}
    with settings_path.open("r", encoding="utf-8") as handle:
        loaded = _yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


@dataclass(frozen=True)
class LucidityTickState:
    stage: str
    rem_depth: float
    t_rem_fraction: float
    cycle_index: int


@dataclass
class LucidityModel:
    threshold: float = 0.60
    steepness: float = 8.0

    @classmethod
    def from_settings(cls) -> "LucidityModel":
        settings = _load_settings()
        threshold = float(settings.get("lucidity_threshold", 0.60))
        return cls(threshold=threshold, steepness=8.0)

    def compute_lucidity(self, tick_state: LucidityTickState) -> float:
        if tick_state.stage != "REM":
            return 0.0

        rem_depth = max(0.0, min(1.0, float(tick_state.rem_depth)))
        t_rem_fraction = max(0.0, min(1.0, float(tick_state.t_rem_fraction)))

        sigmoid = 1.0 / (1.0 + math.exp(-self.steepness * (rem_depth - self.threshold)))
        awareness_boost = 1.0 + 0.4 * math.sin(2.0 * math.pi * t_rem_fraction)

        # Cap peak lucidity in a biologically plausible window for late REM.
        probability = float(max(0.0, min(0.75, sigmoid * awareness_boost)))
        logger.debug(
            "lucidity factors: stage=%s rem_depth=%.3f sigmoid=%.3f awareness=%.3f prob=%.3f",
            tick_state.stage,
            rem_depth,
            sigmoid,
            awareness_boost,
            probability,
        )
        return probability
