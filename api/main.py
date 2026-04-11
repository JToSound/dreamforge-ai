from __future__ import annotations

from fastapi import FastAPI

from api.routes import simulation, journal
from api.routes.simulation import simulate_night as _simulate_night
from api.schemas import DreamNightSchema, DreamSimulationRequest


app = FastAPI(title="DreamForge AI API", version="0.2.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/simulate-night", response_model=DreamNightSchema)
async def simulate_night_root(body: DreamSimulationRequest) -> DreamNightSchema:
    """Alias endpoint for running a night simulation at /simulate-night.

    This simply forwards the request to the canonical /api/simulation/night handler
    so that UIs can call a concise top-level path.
    """

    return await _simulate_night(body)


app.include_router(simulation.router, prefix="/api")
app.include_router(journal.router, prefix="/api")
