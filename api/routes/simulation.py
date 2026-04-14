from __future__ import annotations

from fastapi import APIRouter

from api.schemas import DreamNightSchema, DreamSimulationRequest, serialize_dream_night
from core.simulation.engine import SimulationEngine
from core.agents.orchestrator import OrchestratorConfig
from core.utils.pharmacology import PharmacologyProfile


router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/night", response_model=DreamNightSchema)
async def simulate_night(body: DreamSimulationRequest) -> DreamNightSchema:
    pharm = PharmacologyProfile(
        ssri_strength=body.ssri_strength,
        stress_level=body.stress_level,
    )
    config = OrchestratorConfig(
        night_duration_hours=body.duration_hours,
        dt_minutes=body.dt_minutes,
        pharmacology=pharm,
        llm_enabled=body.llm_enabled,
        llm_provider=body.llm_provider,
        llm_model=body.llm_model,
        llm_important_only=body.llm_important_only,
        # Forward API key from request (overrides server env var when provided)
        llm_api_key=body.llm_api_key or None,
    )
    engine = SimulationEngine(config=config)
    engine.simulate_night()  # timeseries stored on orchestrator
    night = engine.build_night()  # embeds sleep/neuro/memory into metadata
    night.config = {
        "duration_hours": body.duration_hours,
        "dt_minutes": body.dt_minutes,
        "ssri_strength": body.ssri_strength,
        "stress_level": body.stress_level,
        "llm_enabled": body.llm_enabled,
        "llm_provider": body.llm_provider,
        "llm_model": body.llm_model,
        "llm_important_only": body.llm_important_only,
    }
    return serialize_dream_night(night)
