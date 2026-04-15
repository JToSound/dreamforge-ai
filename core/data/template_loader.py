from __future__ import annotations

import random
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_yaml = importlib.import_module("yaml")


class SchemaValidationError(ValueError):
    """Raised when one or more template entries fail schema validation."""


class TemplateNotFoundError(LookupError):
    """Raised when no templates match the requested selection context."""


@dataclass(frozen=True)
class TemplateEntry:
    id: str
    stage: str
    emotion: str | None
    neurochem_filter: dict[str, float | None]
    min_words: int
    text: str


class TemplateBank:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.templates_dir = self.data_dir / "templates"
        self.entries: list[TemplateEntry] = []
        self._loaded = False
        self._last_selected_by_key: dict[tuple[str, str], str] = {}
        self._last_selected_id: str | None = None

    def load(self) -> None:
        """Load and validate all template YAML files under data_dir/templates.

        Raises:
            SchemaValidationError: If one or more entries are invalid.
        """
        yaml_files = sorted(self.templates_dir.glob("*_templates.yaml"))
        errors: list[str] = []
        loaded_entries: list[TemplateEntry] = []

        for path in yaml_files:
            with path.open("r", encoding="utf-8") as handle:
                raw = _yaml.safe_load(handle) or []
            if not isinstance(raw, list):
                errors.append(f"{path.name}: root must be a list of entries")
                continue

            for index, item in enumerate(raw):
                if not isinstance(item, dict):
                    errors.append(f"{path.name}[{index}]: entry must be a mapping")
                    continue
                try:
                    loaded_entries.append(self._validate_entry(item, path.name, index))
                except SchemaValidationError as exc:
                    errors.append(str(exc))

        if errors:
            raise SchemaValidationError("\n".join(errors))

        self.entries = loaded_entries
        self._loaded = True

    def select(
        self,
        stage: str,
        emotion: str,
        nchem: dict[str, float],
        rng: random.Random | None = None,
    ) -> TemplateEntry:
        """Select a template based on stage, emotion, and neurochem filters."""
        if not self._loaded:
            self.load()

        normalized_stage = stage.upper()
        normalized_emotion = (emotion or "neutral").lower()
        stage_matches = [
            entry for entry in self.entries if entry.stage == normalized_stage
        ]
        if not stage_matches:
            raise TemplateNotFoundError(f"No templates available for stage={stage}")

        emotion_matches = [
            entry
            for entry in stage_matches
            if entry.emotion in {None, normalized_emotion}
        ]
        if not emotion_matches:
            emotion_matches = [
                entry for entry in stage_matches if entry.emotion is None
            ]
        if not emotion_matches:
            emotion_matches = stage_matches

        filtered = [
            entry
            for entry in emotion_matches
            if self._matches_neurochem_filter(entry.neurochem_filter, nchem)
        ]
        constrained_filtered = [
            entry
            for entry in filtered
            if any(value is not None for value in entry.neurochem_filter.values())
        ]
        candidates = constrained_filtered or filtered or emotion_matches
        key = (normalized_stage, normalized_emotion)
        last_id = self._last_selected_by_key.get(key)
        if self._last_selected_id and len(candidates) > 1:
            deduped_global = [
                entry for entry in candidates if entry.id != self._last_selected_id
            ]
            if deduped_global:
                candidates = deduped_global
        if (
            self._last_selected_id
            and len(candidates) == 1
            and candidates[0].id == self._last_selected_id
        ):
            fallback_pool = [entry for entry in (filtered or emotion_matches)]
            alternatives = [
                entry for entry in fallback_pool if entry.id != self._last_selected_id
            ]
            if alternatives:
                candidates = alternatives
        if last_id and len(candidates) > 1:
            deduped = [entry for entry in candidates if entry.id != last_id]
            if deduped:
                candidates = deduped

        chooser = rng if rng is not None else random
        selected = chooser.choice(candidates)
        self._last_selected_by_key[key] = selected.id
        self._last_selected_id = selected.id
        return selected

    @staticmethod
    def _validate_entry(
        entry: dict[str, Any], file_name: str, index: int
    ) -> TemplateEntry:
        allowed_stages = {"N1", "N2", "N3", "REM"}
        allowed_emotions = {
            "neutral",
            "anxious",
            "fearful",
            "melancholic",
            "joyful",
            "serene",
            "curious",
        }
        required_keys = {
            "id",
            "stage",
            "emotion",
            "neurochem_filter",
            "min_words",
            "text",
        }
        missing = required_keys - set(entry.keys())
        if missing:
            raise SchemaValidationError(
                f"{file_name}[{index}]: missing required keys {sorted(missing)}"
            )

        entry_id = str(entry["id"])
        stage = str(entry["stage"]).upper()
        if stage not in allowed_stages:
            raise SchemaValidationError(
                f"{file_name}[{index}]: invalid stage '{entry['stage']}'"
            )

        emotion_raw = entry["emotion"]
        emotion: str | None = None
        if emotion_raw is not None:
            emotion = str(emotion_raw).lower()
            if emotion not in allowed_emotions:
                raise SchemaValidationError(
                    f"{file_name}[{index}]: invalid emotion '{emotion_raw}'"
                )

        filter_raw = entry["neurochem_filter"]
        if not isinstance(filter_raw, dict):
            raise SchemaValidationError(
                f"{file_name}[{index}]: neurochem_filter must be a mapping"
            )
        neurochem_filter: dict[str, float | None] = {
            "ach_min": TemplateBank._as_optional_float(filter_raw.get("ach_min")),
            "ach_max": TemplateBank._as_optional_float(filter_raw.get("ach_max")),
            "serotonin_max": TemplateBank._as_optional_float(
                filter_raw.get("serotonin_max")
            ),
        }

        min_words = int(entry["min_words"])
        text = str(entry["text"]).strip()
        if min_words < 40:
            raise SchemaValidationError(
                f"{file_name}[{index}]: min_words must be >= 40 (got {min_words})"
            )

        return TemplateEntry(
            id=entry_id,
            stage=stage,
            emotion=emotion,
            neurochem_filter=neurochem_filter,
            min_words=min_words,
            text=text,
        )

    @staticmethod
    def _as_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _matches_neurochem_filter(
        neurochem_filter: dict[str, float | None], nchem: dict[str, float]
    ) -> bool:
        ach = float(nchem.get("ach", 0.0))
        serotonin = float(nchem.get("serotonin", 0.0))

        ach_min = neurochem_filter.get("ach_min")
        ach_max = neurochem_filter.get("ach_max")
        serotonin_max = neurochem_filter.get("serotonin_max")

        if ach_min is not None and ach < ach_min:
            return False
        if ach_max is not None and ach > ach_max:
            return False
        if serotonin_max is not None and serotonin > serotonin_max:
            return False
        return True
