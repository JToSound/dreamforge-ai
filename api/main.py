from __future__ import annotations

from fastapi import FastAPI

from api.routes import simulation, journal

from api.routes.llm_settings import router as llm_router
app.include_router(llm_router)

app = FastAPI(title="DreamForge AI API", version="0.2.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(simulation.router, prefix="/api")
app.include_router(journal.router, prefix="/api")
