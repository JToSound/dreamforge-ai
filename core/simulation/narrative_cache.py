from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.models.sleep_cycle import SleepStage
from core.simulation.llm_trigger import LLMTriggerType


@dataclass
class NarrativeCache:
    """Cache for trigger-driven narrative blueprints and template fallback text."""

    active_rem_blueprint: dict = field(default_factory=dict)
    last_trigger_type: LLMTriggerType | None = None

    _stage_templates: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {
            "REM": {
                "neutral": [
                    "I drift through a radiant corridor where gravity loosens and every doorway opens into another impossible room.",
                    "The city folds inward like origami, and I move through glowing streets as if they were a single thought.",
                    "I watch familiar faces dissolve into symbols, then reform as if the dream is editing itself in real time.",
                ],
                "fear": [
                    "I run through a tilted hallway while the walls breathe, and every turn reveals a place I thought I had escaped.",
                    "The sky flashes with alarms no one else can hear, and I realize I have been sprinting in circles for hours.",
                    "A distant voice says my name from inside the mirror, and the floor drops before I can answer.",
                ],
                "joy": [
                    "I leap over shimmering rooftops and land softly on clouds that feel warmer than sunlight.",
                    "Music ripples through the streets and everyone I meet already knows my next line.",
                    "I float above a festival of impossible colors, laughing as the horizon bends to meet me.",
                ],
            },
            "N3": {
                "neutral": [
                    "Everything is heavy and calm, like sinking into warm water with no edges.",
                    "I move slowly through a dark plain where sound arrives long after motion.",
                    "The dream is mostly silence, interrupted by distant fragments of rooms I once knew.",
                ],
                "fear": [
                    "A low hum fills the dark and I cannot tell whether I am awake inside it.",
                    "I sense footsteps in the deep quiet, but every time I turn there is only empty space.",
                    "The room stays still, yet I feel watched from somewhere just beyond the doorway.",
                ],
                "joy": [
                    "A deep calm settles over me, and I rest in a quiet world that asks nothing back.",
                    "I wander a moonlit field with no urgency, only the feeling of being safely held.",
                    "The silence itself feels kind, as if the night is gently carrying me forward.",
                ],
            },
            "N2": {
                "neutral": [
                    "Short scenes flicker in and out: a hallway, a train platform, a conversation I almost remember.",
                    "I cross between small moments stitched together by faint echoes of the day.",
                    "Objects appear and vanish before I can focus, as if memory is changing channels.",
                ],
                "fear": [
                    "Doors keep opening into near-identical rooms, each one slightly more wrong than the last.",
                    "I recognize the place but not the rules, and that uncertainty keeps tightening around me.",
                    "A missed call keeps appearing on every screen I touch, though none of them have numbers.",
                ],
                "joy": [
                    "I follow warm lights through familiar streets and every corner reveals another gentle surprise.",
                    "Friends appear in fragments of old places, and each moment feels unexpectedly bright.",
                    "I keep finding small gifts in ordinary rooms, as if the dream is quietly celebrating.",
                ],
            },
            "N1": {
                "neutral": [
                    "The boundary of sleep blurs and tiny images drift past like sparks behind closed eyes.",
                    "I hover at the edge of sleep while brief scenes flicker and fade before they settle.",
                    "Shapes and whispers overlap as I drift between wakefulness and dream.",
                ],
                "fear": [
                    "A sudden jolt of unease passes through me as shadows sharpen and disappear again.",
                    "I hear my name at the edge of sleep and cannot tell where it came from.",
                    "The room feels near and far at once, and I wake inside the uncertainty.",
                ],
                "joy": [
                    "Soft colors pulse behind my eyes and the world loosens into a gentle glow.",
                    "I drift through brief, bright images that feel like tiny promises of the night ahead.",
                    "A warm wave carries me toward sleep as distant voices turn to music.",
                ],
            },
            "WAKE": {
                "neutral": [
                    "I wake for a moment, the room quiet and dark, and the dream dissolves before I can hold it.",
                    "A brief awakening pulls me to the surface, then the fragments slip away again.",
                    "For a second I am fully here, then the night takes me back.",
                ],
                "fear": [
                    "I wake abruptly with my heart racing, unsure whether the threat was real or imagined.",
                    "The dream snaps off and leaves a sharp residue of dread in the room.",
                    "A sudden awakening breaks the scene, but the fear lingers a few breaths longer.",
                ],
                "joy": [
                    "I surface briefly with a smile, carrying a warm trace of the dream into waking.",
                    "A short awakening glows with leftover happiness before sleep returns.",
                    "I wake lightly, still feeling the gentle brightness of the last scene.",
                ],
            },
        }
    )

    def update_from_llm(self, trigger_type: LLMTriggerType, payload: dict) -> None:
        self.last_trigger_type = trigger_type
        if trigger_type == LLMTriggerType.REM_EPISODE_ONSET:
            self.active_rem_blueprint = {
                "opening_scene": str(
                    payload.get("opening_scene") or payload.get("narrative") or ""
                ),
                "narrative_thread": str(payload.get("narrative_thread") or ""),
                "peak_bizarre_moment": str(payload.get("peak_bizarre_moment") or ""),
                "emotional_climax": str(payload.get("emotional_climax") or ""),
                "atmosphere": str(payload.get("atmosphere") or ""),
            }

    def get_segment_narrative(
        self, segment_index: int, emotion: str, stage: SleepStage
    ) -> str:
        stage_key = stage.value if isinstance(stage, SleepStage) else str(stage)
        emotion_key = (emotion or "neutral").lower()
        emotion_bucket = (
            emotion_key if emotion_key in {"neutral", "fear", "joy"} else "neutral"
        )

        # Use the active REM blueprint when available.
        if stage_key == "REM" and self.active_rem_blueprint:
            thread = self.active_rem_blueprint.get("narrative_thread", "")
            opening = self.active_rem_blueprint.get("opening_scene", "")
            peak = self.active_rem_blueprint.get("peak_bizarre_moment", "")
            climax = self.active_rem_blueprint.get("emotional_climax", "")
            atmosphere = self.active_rem_blueprint.get("atmosphere", "")
            fragments = [opening, thread, peak, climax, atmosphere]
            usable = [f for f in fragments if f]
            if usable:
                return usable[segment_index % len(usable)]

        stage_templates = self._stage_templates.get(
            stage_key, self._stage_templates["N2"]
        )
        options = (
            stage_templates.get(emotion_bucket)
            or stage_templates.get("neutral")
            or self._stage_templates["N2"]["neutral"]
        )
        return random.choice(options)
