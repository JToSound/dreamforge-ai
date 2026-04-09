from __future__ import annotations

from fastapi import APIRouter

from api.schemas import DreamNightSchema, DreamSimulationRequest, serialize_dream_night
from core.simulation.engine import SimulationEngine
from core.agents.orchestrator import OrchestratorConfig
from core.utils.pharmacology import PharmacologyProfile


router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/night", response_model=DreamNightSchema)
async def simulate_night(body: DreamSimulationRequest) -> DreamNightSchema:
    pharm = PharmacologyProfile(ssri_strength=body.ssri_strength, stress_level=body.stress_level)
    config = OrchestratorConfig(
        night_duration_hours=body.duration_hours,
        dt_minutes=body.dt_minutes,
        pharmacology=pharm,
    )
    engine = SimulationEngine(config=config)
    engine.simulate_night()
    night = engine.build_night()
    night.config = {
        "duration_hours": body.duration_hours,
        "dt_minutes": body.dt_minutes,
        "ssri_strength": body.ssri_strength,
        "stress_level": body.stress_level,
    }

    return serialize_dream_night(night)
