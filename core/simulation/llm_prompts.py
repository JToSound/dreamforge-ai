from __future__ import annotations


def prefix_no_think(user_prompt: str) -> str:
    """Prefix a user prompt with Qwen no-think directive.

    Args:
        user_prompt: Prompt text intended for the user role.

    Returns:
        Prompt with `/no_think` prepended.
    """
    return f"/no_think\n\n{user_prompt}"


def build_narrative_messages(
    stage: str,
    emotion: str,
    bizarreness: float,
    ach: float,
    ne: float,
    cortisol: float,
) -> list[dict[str, str]]:
    """Build standardized LLM messages for narrative generation.

    Args:
        stage: Sleep stage label.
        emotion: Dominant emotion label.
        bizarreness: Segment bizarreness score.
        ach: Acetylcholine level.
        ne: Norepinephrine level.
        cortisol: Cortisol level.

    Returns:
        A chat-completions message list with `/no_think` prepended to the user
        message content.
    """
    system_prompt = (
        "You are a dream narrative generator. Return JSON with keys "
        "narrative and scene_description."
    )
    user_prompt = (
        f"Stage: {stage}\n"
        f"Emotion: {emotion}\n"
        f"Bizarreness: {bizarreness:.3f}\n"
        f"ACh: {ach:.3f}\n"
        f"NE: {ne:.3f}\n"
        f"Cortisol: {cortisol:.3f}\n"
        "Respond with JSON only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prefix_no_think(user_prompt)},
    ]
