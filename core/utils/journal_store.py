from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


_JOURNAL_PATH = Path(__file__).resolve().parents[2] / "data" / "journal.jsonl"


@dataclass
class JournalEntry:
    text: str
    emotion: str
    stress_level: float
    tags: List[str]
    created_at: str


def _ensure_dir() -> None:
    _JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_journal_entry(text: str, emotion: str, stress_level: float, tags: List[str]) -> None:
    _ensure_dir()
    entry = JournalEntry(
        text=text.strip(),
        emotion=emotion,
        stress_level=stress_level,
        tags=tags,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    with _JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load_journal_entries() -> List[Dict[str, Any]]:
    if not _JOURNAL_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with _JOURNAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
