from __future__ import annotations

from typing import Dict


def nchem_to_descriptors(
    ach: float, serotonin: float, ne: float, cortisol: float
) -> Dict[str, str]:
    """Map neurochemical levels to qualitative dream descriptors.

    Source: Hobson JA (2009) Nature Reviews Neuroscience 10:803.
    """
    ach_state = (
        "vivid and hyperconnected"
        if ach > 0.70
        else (
            "fluid and associative"
            if ach > 0.45
            else "dim and formless" if ach > 0.25 else "near-absent"
        )
    )
    mood_tone = (
        "serene"
        if serotonin > 0.55
        else (
            "neutral"
            if serotonin > 0.35
            else "dysphoric" if serotonin > 0.15 else "bleak"
        )
    )
    arousal_level = (
        "hyperalert"
        if ne > 0.65
        else "alert" if ne > 0.40 else "calm" if ne > 0.20 else "suppressed"
    )
    stress_signature = (
        "cortisol surge — urgency permeates the scene"
        if cortisol > 0.65
        else "mild background tension" if cortisol > 0.40 else "baseline ease"
    )
    return {
        "ach_state": ach_state,
        "mood_tone": mood_tone,
        "arousal_level": arousal_level,
        "stress_signature": stress_signature,
    }
