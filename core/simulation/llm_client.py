from __future__ import annotations

import json
import re


def parse_narrative_response(content: str) -> dict[str, str]:
    """Parse a narrative payload from an LLM response string.

    Args:
        content: Raw model output, potentially containing think tags, markdown
            fences, or truncated JSON.

    Returns:
        Dictionary containing `narrative` and `scene_description` keys. Missing
        fields are returned as empty strings.
    """
    if not content or not content.strip():
        return {"narrative": "", "scene_description": ""}

    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        return {
            "narrative": str(data.get("narrative", "")),
            "scene_description": str(
                data.get("scene_description", data.get("scene", ""))
            ),
        }
    except json.JSONDecodeError:
        pass

    narrative_match = re.search(
        r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"',
        cleaned,
        flags=re.DOTALL,
    )
    scene_match = re.search(
        r'"scene(?:_description)?"\s*:\s*"((?:[^"\\]|\\.)*)"',
        cleaned,
        flags=re.DOTALL,
    )

    if narrative_match:
        return {
            "narrative": narrative_match.group(1),
            "scene_description": scene_match.group(1) if scene_match else "",
        }

    return {"narrative": "", "scene_description": ""}
