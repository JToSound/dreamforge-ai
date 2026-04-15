from __future__ import annotations

import random
from pathlib import Path

import pytest

from core.data.template_loader import TemplateBank, TemplateNotFoundError


def _data_dir() -> Path:
    return Path("core") / "data"


def test_all_yaml_files_validate_against_schema() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    assert len(bank.entries) >= 32


def test_select_returns_stage_match() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    selected = bank.select(
        stage="N3",
        emotion="neutral",
        nchem={"ach": 0.3, "serotonin": 0.25},
        rng=random.Random(1),
    )
    assert selected.stage == "N3"


def test_select_prefers_emotion_match() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    selected = bank.select(
        stage="N2",
        emotion="curious",
        nchem={"ach": 0.5, "serotonin": 0.25},
        rng=random.Random(2),
    )
    assert selected.emotion == "curious"


def test_select_applies_neurochem_filter() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    selected = bank.select(
        stage="REM",
        emotion="fearful",
        nchem={"ach": 0.82, "serotonin": 0.10},
        rng=random.Random(3),
    )
    assert selected.neurochem_filter["ach_min"] is not None
    assert selected.neurochem_filter["serotonin_max"] is not None


def test_select_is_deterministic_with_fixed_seed() -> None:
    bank_a = TemplateBank(_data_dir())
    bank_b = TemplateBank(_data_dir())
    bank_a.load()
    bank_b.load()

    rng_a = random.Random(99)
    rng_b = random.Random(99)
    seq_a = [
        bank_a.select(
            stage="N2",
            emotion="neutral",
            nchem={"ach": 0.45, "serotonin": 0.25},
            rng=rng_a,
        ).id
        for _ in range(5)
    ]
    seq_b = [
        bank_b.select(
            stage="N2",
            emotion="neutral",
            nchem={"ach": 0.45, "serotonin": 0.25},
            rng=rng_b,
        ).id
        for _ in range(5)
    ]
    assert seq_a == seq_b


def test_no_consecutive_duplicates_across_20_calls() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    rng = random.Random(17)
    selected_ids = [
        bank.select(
            stage="REM",
            emotion="neutral",
            nchem={"ach": 0.8, "serotonin": 0.1},
            rng=rng,
        ).id
        for _ in range(20)
    ]
    assert all(a != b for a, b in zip(selected_ids, selected_ids[1:]))


def test_select_raises_on_unknown_stage() -> None:
    bank = TemplateBank(_data_dir())
    bank.load()
    with pytest.raises(TemplateNotFoundError):
        bank.select(
            stage="WAKE",
            emotion="neutral",
            nchem={"ach": 0.5, "serotonin": 0.3},
            rng=random.Random(7),
        )
