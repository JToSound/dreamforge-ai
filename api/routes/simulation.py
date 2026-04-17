from __future__ import annotations

from fastapi import APIRouter

from api import main as api_main
from api.schemas import DreamNightSchema, DreamSimulationRequest, serialize_dream_night

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SimulationEngine:
    """Legacy-compatible shim retained for route-level compatibility tests."""

    def __init__(self, config: api_main.SimulationConfig):
        self.config = config

    def build_night(self):
        return None


@router.post(
    "/night",
    response_model=DreamNightSchema,
    deprecated=True,
    summary="Legacy simulation endpoint compatibility adapter",
)
async def simulate_night(body: DreamSimulationRequest) -> DreamNightSchema:
    """Compatibility adapter to the active api.main SimulationConfig contract."""
    config = api_main.SimulationConfig(
        duration_hours=body.duration_hours,
        dt_minutes=body.dt_minutes,
        ssri_strength=body.ssri_strength,
        stress_level=body.stress_level,
        sleep_start_hour=23.0,
        melatonin=False,
        cannabis=False,
        prior_day_events=[],
        emotional_state="neutral",
        style_preset="scientific",
        prompt_profile="A",
        use_llm=bool(body.llm_enabled),
        llm_segments_only=bool(body.llm_important_only),
    )
    legacy_engine = SimulationEngine(config)
    legacy_night = legacy_engine.build_night()
    if legacy_night is not None:
        legacy_payload = serialize_dream_night(legacy_night)
        merged_config = dict(legacy_payload.config or {})
        merged_config.setdefault("duration_hours", body.duration_hours)
        merged_config.setdefault("dt_minutes", body.dt_minutes)
        merged_config.setdefault("stress_level", body.stress_level)
        return DreamNightSchema(
            id=legacy_payload.id,
            segments=legacy_payload.segments,
            config=merged_config,
            notes=legacy_payload.notes,
            metadata=legacy_payload.metadata,
        )

    live_payload = (await api_main.simulate_night(config)).model_dump()
    return DreamNightSchema.model_validate(
        {
            "id": live_payload.get("id", ""),
            "segments": live_payload.get("segments", []),
            "config": live_payload.get("config", {}),
            "metadata": {"source": "api.main"},
        }
    )
