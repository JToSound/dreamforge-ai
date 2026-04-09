from __future__ import annotations

from fastapi import FastAPI

from api.routes import simulation


app = FastAPI(title="DreamForge AI API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(simulation.router, prefix="/api")
