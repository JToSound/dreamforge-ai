from __future__ import annotations

import logging
import random

import numpy as np

from core.models.sleep_cycle import (
    CycleStateMachine,
    SleepCycleModel,
    SleepStage,
    enforce_n3_floor,
)


def _fraction(stages: list[SleepStage], target: SleepStage) -> float:
    return sum(1 for stage in stages if stage == target) / max(1, len(stages))


def test_n3_floor_enforced_across_10_seeds() -> None:
    for seed in range(10):
        np.random.seed(seed)
        random.seed(seed)
        model = SleepCycleModel()
        _, stages = model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
        assert _fraction(stages, SleepStage.N3) >= 0.10


def test_front_weighting_of_n3() -> None:
    np.random.seed(42)
    random.seed(42)
    model = SleepCycleModel()
    states, _ = model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
    total_n3_minutes = 0.0
    first_cycle_n3_minutes = 0.0
    for state in states:
        if state.stage == SleepStage.N3:
            total_n3_minutes += 0.5
            if state.time_hours <= 2.0:
                first_cycle_n3_minutes += 0.5
    assert first_cycle_n3_minutes >= 0.40 * max(1.0, total_n3_minutes)


def test_rem_not_converted() -> None:
    schedule = [SleepStage.N2] * 90 + [SleepStage.REM] * 20 + [SleepStage.N2] * 90
    rem_before = schedule.count(SleepStage.REM)
    updated = enforce_n3_floor(schedule, n3_min_fraction=0.20)
    rem_after = updated.count(SleepStage.REM)
    assert rem_after == rem_before


def test_n2_ceiling_still_respected() -> None:
    for seed in (0, 7, 42):
        np.random.seed(seed)
        random.seed(seed)
        model = SleepCycleModel()
        _, stages = model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
        assert _fraction(stages, SleepStage.N2) <= 0.58


def test_homeostatic_debt_log_emitted(caplog) -> None:
    templates = {
        idx: [
            (SleepStage.N1, 2.0),
            (SleepStage.N2, 70.0),
            (SleepStage.REM, 18.0),
        ]
        for idx in range(1, 7)
    }
    machine = CycleStateMachine(
        templates=templates,
        default_template=[
            (SleepStage.N1, 2.0),
            (SleepStage.N2, 70.0),
            (SleepStage.REM, 18.0),
        ],
        sws_debt_threshold=0.0,
    )
    model = SleepCycleModel()
    model.cycle_state_machine = machine

    with caplog.at_level(logging.DEBUG, logger="core.models.sleep_cycle"):
        model.simulate_night(duration_hours=8.0, dt_minutes=0.5)
    assert "N2_rebalance fired" in caplog.text
