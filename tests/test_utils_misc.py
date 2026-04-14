from pathlib import Path

from core.models.neurochemistry import NeurochemistryParameters
from core.models.sleep_cycle import SleepStage
from core.simulation.llm_trigger import LLMTriggerType
from core.simulation.narrative_cache import NarrativeCache
from core.utils.journal_store import append_journal_entry, load_journal_entries
from core.utils.pharmacology import PharmacologyProfile, apply_pharmacology


def test_journal_store_append_and_load(tmp_path, monkeypatch):
    import core.utils.journal_store as journal_store

    journal_path = Path(tmp_path) / "journal.jsonl"
    monkeypatch.setattr(journal_store, "_JOURNAL_PATH", journal_path)

    assert load_journal_entries() == []

    append_journal_entry("  hello world  ", "joy", 0.2, ["work", "meeting"])
    with journal_path.open("a", encoding="utf-8") as f:
        f.write("not-json\n")
        f.write("\n")

    entries = load_journal_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == "hello world"
    assert entries[0]["emotion"] == "joy"
    assert entries[0]["created_at"].endswith("Z")


def test_apply_pharmacology_uses_profile_and_preserves_original():
    params = NeurochemistryParameters(ssri_factor=1.0, cortisol_amplitude=0.6)
    profile = PharmacologyProfile(ssri_strength=1.7, stress_level=0.4)

    updated = apply_pharmacology(params, profile)
    assert updated.ssri_factor == 1.7
    assert updated.cortisol_amplitude == 0.6 * (1.0 + 0.5 * 0.4)

    # original object is unchanged (apply_pharmacology uses a deep copy)
    assert params.ssri_factor == 1.0
    assert params.cortisol_amplitude == 0.6


def test_narrative_cache_blueprint_and_template_fallback(monkeypatch):
    cache = NarrativeCache()
    monkeypatch.setattr(
        "core.simulation.narrative_cache.random.choice", lambda xs: xs[0]
    )

    # Non-REM path should use a stage template fallback.
    text = cache.get_segment_narrative(
        segment_index=0, emotion="unknown", stage=SleepStage.N2
    )
    assert isinstance(text, str)
    assert len(text) > 0

    payload = {
        "opening_scene": "opening",
        "narrative_thread": "thread",
        "peak_bizarre_moment": "peak",
        "emotional_climax": "climax",
        "atmosphere": "atmo",
    }
    cache.update_from_llm(LLMTriggerType.REM_EPISODE_ONSET, payload)
    assert cache.last_trigger_type == LLMTriggerType.REM_EPISODE_ONSET

    rem_text = cache.get_segment_narrative(
        segment_index=3, emotion="joy", stage=SleepStage.REM
    )
    assert rem_text in {"opening", "thread", "peak", "climax", "atmo"}
