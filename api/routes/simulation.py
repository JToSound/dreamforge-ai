from __future__ import annotations

from fastapi import APIRouter

from api.schemas import DreamNightSchema, DreamSimulationRequest, serialize_dream_night
from core.simulation.engine import SimulationEngine
from core.agents.orchestrator import OrchestratorConfig
from core.agents.phenomenology_reporter import PhenomenologyReporter


router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/night", response_model=DreamNightSchema)
async def simulate_night(body: DreamSimulationRequest) -> DreamNightSchema:
    config = OrchestratorConfig(
        night_duration_hours=body.duration_hours,
        dt_minutes=body.dt_minutes,
    )
    engine = SimulationEngine(config=config)
    segments = engine.simulate_night()

    reporter = engine.orchestrator.phenom_agent
    night = reporter.build_night()
    night.config = {"duration_hours": body.duration_hours, "dt_minutes": body.dt_minutes}

    return serialize_dream_night(night)
