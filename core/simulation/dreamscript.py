"""Deterministic offline narrative engine used when no live LLM is available."""

from __future__ import annotations

import random
import re
from typing import List, Optional, Sequence

from core.models.memory_graph import MemoryNodeModel
from core.models.sleep_cycle import SleepStage
from core.utils.bizarreness_scorer import BizarrenessScore


class DreamScriptEngine:
    """Template-driven narrative engine for demo and offline generation."""

    _BIZARRE_VOCAB = (
        "non-euclidean",
        "kaleidoscopic",
        "fractal",
        "impossible",
        "liquid",
        "glitching",
        "holographic",
        "paradoxical",
    )
    _REALITY_FAILURE_PHRASES = (
        "Nobody questions the contradiction, and neither do I.",
        "I accept the impossible rule as if it has always been true.",
        "The scene breaks continuity, yet it feels perfectly ordinary.",
        "Causality slips, but my mind treats it as routine.",
        "Time jumps forward and backward without warning, and I remain calm.",
    )
    _ANXIETY_PHRASES = (
        "A background alarm keeps pulsing through the scene.",
        "Everything carries a sharp edge of urgency.",
        "My chest tightens as if I am late for something unnamed.",
        "The atmosphere feels tense, watchful, and over-bright.",
        "A low dread hums beneath every detail.",
    )

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self._carry_regex = re.compile(r'\b([A-Z][a-z]{2,})\b|"([^"]+)"')

        self.NREM_LIGHT = self._build_nrem_light_bank()
        self.NREM_DEEP = self._build_nrem_deep_bank()
        self.REM_EARLY = self._build_rem_early_bank()
        self.REM_LATE = self._build_rem_late_bank()

    def _compose_bank(
        self,
        openings: Sequence[str],
        middles: Sequence[str],
        endings: Sequence[str],
        min_count: int = 32,
    ) -> list[str]:
        bank: list[str] = []
        for opening in openings:
            for middle in middles:
                for ending in endings:
                    bank.append(f"{opening} {middle} {ending}")
                    if len(bank) >= min_count:
                        return bank
        return bank

    def _build_nrem_light_bank(self) -> list[str]:
        openings = (
            "I move slowly through {place}",
            "I stand near a quiet window in {place}",
            "I walk down a familiar corridor beside {place}",
            "I follow soft footsteps into {place}",
            "I sit at the edge of {place}",
            "I drift through an ordinary morning inside {place}",
            "I wait in a pale hallway that opens onto {place}",
            "I cross a courtyard near {place}",
        )
        middles = (
            "while holding a small {object}",
            "while replaying a conversation about {topic}",
            "as someone mentions {person} in passing",
            "as the air fills with the scent of {scent}",
            "while my sleeves brush against threads of {tag}",
            "as a clock clicks softly in another room",
            "while distant voices repeat one unfinished sentence",
            "as a familiar melody loops in the background",
        )
        endings = (
            "Nothing dramatic happens, but the moment feels meaningful.",
            "The scene stays calm, as if memory is settling into place.",
            "I keep expecting a turn, yet the dream remains gentle.",
            "The details blur at the edges, then return with quiet clarity.",
        )
        return self._compose_bank(openings, middles, endings, min_count=32)

    def _build_nrem_deep_bank(self) -> list[str]:
        openings = (
            "I descend into a heavy landscape beneath {place}",
            "I sink through dark water under {place}",
            "I stand in a muted plain that borders {place}",
            "I wait in a silent chamber below {place}",
            "I move through dense fog around {place}",
            "I drift across a dim field beyond {place}",
            "I kneel in still earth near {place}",
            "I rest in a cavern that echoes {place}",
        )
        middles = (
            "while searching for {person}",
            "while listening to distant echoes of {topic}",
            "while tracing symbols carved into {object}",
            "while a faint smell of {scent} rises and falls",
            "while old memories gather around the word {tag}",
            "while every step takes longer than expected",
            "while the horizon pulses once and goes dark",
            "while shapes emerge and vanish without sound",
        )
        endings = (
            "The dream feels primitive, slow, and gravitational.",
            "Everything is reduced to weight, temperature, and pulse.",
            "I remain suspended in deep silence until the scene dissolves.",
            "The world narrows to sensation, then fades to black.",
        )
        return self._compose_bank(openings, middles, endings, min_count=32)

    def _build_rem_early_bank(self) -> list[str]:
        openings = (
            "I step into a bright version of {place}",
            "I run across rooftops above {place}",
            "I float through a market that resembles {place}",
            "I ride a train of light toward {place}",
            "I enter a theater built from mirrors of {place}",
            "I climb stairs that fold into {place}",
            "I wake inside a painting of {place}",
            "I sprint through a carnival orbiting {place}",
        )
        middles = (
            "where {person} hands me {object}",
            "while the crowd chants about {topic}",
            "as banners made of {tag} sweep the sky",
            "while the smell of {scent} turns electric",
            "as gravity bends around my footsteps",
            "while every doorway opens to a new timeline",
            "as streetlights blink in impossible rhythms",
            "while the horizon scrolls like a film reel",
        )
        endings = (
            "I laugh at the impossibility and keep moving.",
            "The scene remains vivid, coherent, and dreamlike.",
            "Color saturates everything until edges start to shimmer.",
            "The narrative accelerates, but I still feel oriented.",
        )
        return self._compose_bank(openings, middles, endings, min_count=32)

    def _build_rem_late_bank(self) -> list[str]:
        openings = (
            "I awaken inside a vast city made from {tag}",
            "I fly over {place} as sunrise fractures into prisms",
            "I walk through a cathedral of moving equations near {place}",
            "I stand on a shoreline where {place} floats in the sky",
            "I drift between moons while watching {place} rotate below",
            "I cross a bridge of glass memories above {place}",
            "I enter a chamber where every wall predicts tomorrow",
            "I follow a beam of light toward the center of {place}",
        )
        middles = (
            "and {person} asks me to decode {object}",
            "while voices debate the meaning of {topic}",
            "as the wind writes notes in the scent of {scent}",
            "while old and future versions of me trade places",
            "as clocks melt into a map I can almost read",
            "while language turns into color and then into motion",
            "as the world folds and unfolds in repeating layers",
            "while distant choirs pronounce one perfect sentence",
        )
        endings = (
            "I feel lucid for a moment, then surrender to the surge.",
            "The dream peaks in scale and symbolic intensity.",
            "Everything converges into one luminous, unstable image.",
            "I sense awakening nearby, but the dream keeps negotiating for time.",
        )
        return self._compose_bank(openings, middles, endings, min_count=32)

    def _select_template_bank(
        self, stage: SleepStage, biz: BizarrenessScore
    ) -> List[str]:
        if stage == SleepStage.N1:
            return self.NREM_LIGHT
        if stage in (SleepStage.N2, SleepStage.N3):
            return self.NREM_DEEP if stage == SleepStage.N3 else self.NREM_LIGHT
        if stage == SleepStage.REM:
            return self.REM_LATE if biz.total_score >= 0.72 else self.REM_EARLY
        return self.NREM_LIGHT

    def _build_vocab(self, active_memories: List[MemoryNodeModel]) -> dict:
        tags: list[str] = []
        people: list[str] = []
        places: list[str] = []
        objects: list[str] = []
        topics: list[str] = []
        scents: list[str] = []

        for memory in active_memories:
            label = (memory.label or "").strip()
            if label:
                objects.append(label[:48])
            tags.extend([tag for tag in memory.tags if tag])
            if getattr(memory, "emotion", None):
                topics.append(str(memory.emotion))

        tags = tags or ["thread", "silver", "echo", "paper", "glass"]
        people = people or ["Anna", "Marco", "Lina", "Elias", "Noor"]
        places = places or [
            "the old station",
            "a quiet harbor",
            "the school atrium",
            "a moonlit avenue",
        ]
        objects = [obj for obj in objects if obj][:12] or [
            "book",
            "clock",
            "compass",
            "key",
            "notebook",
        ]
        topics = topics or ["home", "loss", "work", "departure", "return"]
        scents = scents or ["rain", "coffee", "salt", "smoke", "pine"]

        return {
            "tag": random.choice(tags),
            "person": random.choice(people),
            "place": random.choice(places),
            "object": random.choice(objects),
            "topic": random.choice(topics),
            "scent": random.choice(scents),
        }

    def _choose_template(self, bank: List[str], biz_score: float, ach: float) -> str:
        weights = []
        n = len(bank)
        for idx in range(n):
            w = 1.0 + biz_score * (idx / max(1, n - 1))
            w *= 1.0 + max(0.0, ach - 0.7) * 1.6
            weights.append(w)
        return random.choices(bank, weights=weights, k=1)[0]

    def _inject_modifiers(
        self, text: str, ach: float, ne: float, cortisol: float
    ) -> str:
        out = text
        if ach > 0.7:
            token = random.choice(self._BIZARRE_VOCAB)
            out += f" The geometry turns {token}, and the scene behaves like a living riddle."
        if ne < 0.1:
            out += " " + random.choice(self._REALITY_FAILURE_PHRASES)
        if cortisol > 0.75:
            out += " " + random.choice(self._ANXIETY_PHRASES)
        return out

    def _extract_carry(self, prev_segment_text: Optional[str]) -> Optional[str]:
        if not prev_segment_text:
            return None
        match = self._carry_regex.search(prev_segment_text)
        if not match:
            return None
        return (match.group(1) or match.group(2) or "").strip()

    def generate_narrative(
        self,
        stage: SleepStage,
        neurochemistry,
        active_memories: List[MemoryNodeModel],
        bizarreness: BizarrenessScore,
        prev_segment_text: Optional[str] = None,
    ) -> str:
        bank = self._select_template_bank(stage, bizarreness)
        vocab = self._build_vocab(active_memories)
        template = self._choose_template(
            bank,
            bizarreness.total_score,
            float(getattr(neurochemistry, "ach", 0.5)),
        )

        carry = self._extract_carry(prev_segment_text)
        if carry:
            vocab["person"] = carry

        narrative = template.format(**vocab)
        if carry and carry not in narrative:
            narrative += f" {carry} keeps reappearing at the edge of the scene."
        narrative = self._inject_modifiers(
            narrative,
            float(getattr(neurochemistry, "ach", 0.5)),
            float(getattr(neurochemistry, "ne", 0.5)),
            float(getattr(neurochemistry, "cortisol", 0.5)),
        )

        if bizarreness.total_score > 0.9:
            narrative = narrative.replace(".", "…")

        return narrative
