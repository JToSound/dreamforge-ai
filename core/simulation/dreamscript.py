from __future__ import annotations

"""Deterministic template-based offline dream narrative engine.

The DreamScriptEngine provides an automatic fallback narrative generator that
requires no external LLM. It is intentionally template-driven and parameter
modulated by neurochemistry and bizarreness to produce plausible, varied
segments suitable for demo mode and reproducible GIFs.
"""
from typing import List, Optional
import random
import re

from core.models.memory_graph import MemoryNodeModel
from core.utils.bizarreness_scorer import BizarrenessScore
from core.models.sleep_cycle import SleepStage


class DreamScriptEngine:
    """Template-based dream generator for offline/demo mode.

    The engine exposes `generate_narrative()` which selects and fills templates
    from four banks (NREM_LIGHT, NREM_DEEP, REM_EARLY, REM_LATE). Templates
    contain placeholders populated from `active_memories` tags and emotion.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        # Minimal template banks; banks can be extended for production.
        self.NREM_LIGHT = [
            "I walked through {place}, carrying a small {object} that felt familiar.",
            "A gentle conversation about {topic} unfolded, like a remembered tune.",
            "The room smelled of {scent} and sunlight filtered through the curtains.",
            "I searched for {person} but only found a doorway that led to a garden.",
            "Someone handed me {object} and said it belonged to {person}.",
            "I walked along {place} and noticed the clock had stopped at noon.",
            "There was a quiet parade of people wearing {tag} on their sleeves.",
            "A window opened onto {place}, and I stepped through without thinking.",
        ]

        self.NREM_DEEP = [
            "The house collapsed into a corridor of doors; behind one was {place}.",
            "I was digging for a memory of {person} and found a sealed box.",
            "The floor turned to soft earth and I sank, but it felt comforting.",
            "There was a hush, and the faces of strangers all looked like {person}.",
            "I sat by a river of photographs and watched them float by in silence.",
            "The walls whispered {topic} in a language I almost remembered.",
            "An old melody guided me to a locked room with {object} on a table.",
            "A child laughed and the sound rearranged the furniture into a map.",
        ]

        self.REM_EARLY = [
            "The street bent into an impossible spiral and {person} waved from above.",
            "Clouds turned into letters spelling {topic}, and I read them aloud.",
            "A mirror reflected a different city where {place} was underwater.",
            "I grew wings made of {object} and flew over {place} as people cheered.",
            "A telephone rang and the voice at the other end described tomorrow.",
            "The sky stitched itself into a patchwork of {tag} and memory.",
            "I walked into a painting of {place} and the colors rearranged my pockets.",
            "The ground hummed with the name of {person} until it became a song.",
        ]

        self.REM_LATE = [
            "Buildings folded like paper; inside, {person} handed me a key made of light.",
            "The ocean whispered secrets about {topic} and I understood them all.",
            "I tasted {object} and it turned into an entire afternoon I had lost.",
            "Time bent, and the faces of everyone I knew spoke in chorus about {place}.",
            "A door opened to an impossible sky and I remembered how to fly with {object}.",
            "The city rearranged itself into a poem about {person} and rain.",
            "A staircase led to the rooftop where I watched moons collide gently.",
            "The world folded like a letter and I read the name {person} inside.",
        ]

        # Carryover regex for simple continuity (capitalize words / quoted phrases)
        self._carry_regex = re.compile(r"\b([A-Z][a-z]{2,})\b|\"([^\"]+)\"")

    def _select_template_bank(self, stage: SleepStage, biz: BizarrenessScore) -> List[str]:
        # Choose bank based on stage and bizarreness magnitude
        if stage in (SleepStage.N1,):
            return self.NREM_LIGHT
        if stage in (SleepStage.N2, SleepStage.N3):
            return self.NREM_DEEP
        # REM: early vs late split by cycle index embedded in biz confidence (heuristic)
        # If biz total_score is moderate, use early REM templates; extreme -> late
        if stage == SleepStage.REM:
            return self.REM_LATE if biz.total_score > 0.7 else self.REM_EARLY
        return self.NREM_LIGHT

    def _build_vocab(self, active_memories: List[MemoryNodeModel]) -> dict:
        tags = []
        people = []
        places = []
        objects = []
        topics = []
        scents = []
        for m in active_memories:
            tags.extend([t for t in m.tags if t])
            label = (m.label or "").strip()
            # heuristic extraction
            if "," in label:
                parts = [p.strip() for p in label.split(",")]
                objects.extend(parts)
            else:
                objects.append(label)
            if m.emotion:
                topics.append(str(m.emotion))
        # fallback vocabulary
        tags = tags or ["blue", "quiet", "strange"]
        people = people or ["Anna", "Marco", "Lina"]
        places = places or ["the market", "the shore", "the old school"]
        objects = [o for o in objects if o][:10] or ["book", "clock", "knife", "umbrella"]
        topics = topics or ["home", "loss", "flight"]
        scents = scents or ["coffee", "salt", "rain"]
        return {
            "tag": random.choice(tags),
            "person": random.choice(people),
            "place": random.choice(places),
            "object": random.choice(objects),
            "topic": random.choice(topics),
            "scent": random.choice(scents),
        }

    def _choose_template(self, bank: List[str], biz_score: float, ach: float) -> str:
        # weighting: higher biz -> prefer templates with surreal tokens (we'll
        # randomize selection while biasing by biz_score)
        n = len(bank)
        weights = []
        for i in range(n):
            # bias later templates slightly for higher biz
            w = 1.0 + biz_score * (i / max(1, n - 1))
            # ACh further biases toward transformation templates
            w *= 1.0 + max(0.0, (ach - 0.7)) * 1.5
            weights.append(w)
        # normalize
        tot = sum(weights)
        probs = [w / tot for w in weights]
        return random.choices(bank, probs, k=1)[0]

    def _inject_modifiers(self, text: str, ach: float, ne: float, cortisol: float) -> str:
        # Modify phrasing based on neurochemistry
        if ach > 0.7:
            # add surreal adjectives
            text = text.replace("the ", "the shimmering ")
        if ne < 0.1:
            # inject reality-failure phrase
            text = text + " It felt natural, even though nothing made sense."
        if cortisol > 0.8:
            text = "Urgent: " + text
        return text

    def _extract_carry(self, prev_segment_text: Optional[str]) -> Optional[str]:
        if not prev_segment_text:
            return None
        m = self._carry_regex.search(prev_segment_text)
        if not m:
            return None
        return (m.group(1) or m.group(2) or "").strip()

    def generate_narrative(
        self,
        stage: SleepStage,
        neurochemistry,
        active_memories: List[MemoryNodeModel],
        bizarreness: BizarrenessScore,
        prev_segment_text: Optional[str] = None,
    ) -> str:
        """Generate a single-segment narrative string.

        Args:
            stage: SleepStage label.
            neurochemistry: object with fields `ach`, `ne`, `cortisol` expected.
            active_memories: list of MemoryNodeModel used to seed vocabulary.
            bizarreness: BizarrenessScore object from the scorer.
            prev_segment_text: optional previous segment text to continue continuity.
        """
        bank = self._select_template_bank(stage, bizarreness)
        vocab = self._build_vocab(active_memories)
        template = self._choose_template(bank, bizarreness.total_score, getattr(neurochemistry, "ach", 0.5))

        # carry entity
        carry = self._extract_carry(prev_segment_text)
        if carry:
            vocab["person"] = carry

        narrative = template.format(**vocab)
        narrative = self._inject_modifiers(narrative, getattr(neurochemistry, "ach", 0.5), getattr(neurochemistry, "ne", 0.5), getattr(neurochemistry, "cortisol", 0.5))

        # make short rewrites for higher bizarreness
        if bizarreness.total_score > 0.8:
            narrative = narrative.replace("and", "— and then")
        if bizarreness.total_score > 0.9:
            narrative = narrative.upper()

        return narrative
