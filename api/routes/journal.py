from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.utils.journal_store import append_journal_entry


router = APIRouter(prefix="/memory", tags=["memory"])


class JournalEntryRequest(BaseModel):
    text: str = Field(..., description="Free-text journal entry describing daytime experience.")
    emotion: str = Field("neutral", description="Dominant emotion label (joy, fear, sadness, anger, surprise, disgust, neutral).")
    stress_level: float = Field(0.0, ge=0.0, le=1.0, description="Perceived stress level (0–1).")
    tags: list[str] = Field(default_factory=list, description="Optional tags such as people, places, themes.")


@router.post("/encode-journal")
async def encode_journal(body: JournalEntryRequest) -> dict:
    append_journal_entry(
        text=body.text,
        emotion=body.emotion,
        stress_level=body.stress_level,
        tags=body.tags,
    )
    return {"status": "ok"}
