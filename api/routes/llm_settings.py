"""
GET  /api/llm/health    — check LM Studio connectivity
POST /api/llm/settings  — hot-reload LLM config without restart
GET  /api/llm/settings  — return current config
"""
from fastapi import APIRouter
from pydantic import BaseModel
from core.llm_client import LLMClient, LLMConfig, _default_client
import core.llm_client as llm_module

router = APIRouter(prefix="/api/llm", tags=["llm"])

class LLMSettingsUpdate(BaseModel):
    base_url:    str   = "http://host.docker.internal:1234/v1"
    model:       str   = "qwen/qwen3.5-9b"
    api_key:     str   = "lm-studio"
    max_tokens:  int   = 512
    temperature: float = 0.85

@router.get("/health")
async def llm_health():
    client = llm_module.get_llm_client()
    result = await client.check_health()
    return result

@router.get("/settings")
async def get_settings():
    client = llm_module.get_llm_client()
    cfg = client.config
    return {
        "provider":    cfg.provider,
        "base_url":    cfg.base_url,
        "model":       cfg.model,
        "max_tokens":  cfg.max_tokens,
        "temperature": cfg.temperature,
    }

@router.post("/settings")
async def update_settings(update: LLMSettingsUpdate):
    """Hot-reload: swap out the global LLM client without container restart."""
    new_cfg = LLMConfig(
        base_url    = update.base_url,
        model       = update.model,
        api_key     = update.api_key,
        max_tokens  = update.max_tokens,
        temperature = update.temperature,
    )
    if llm_module._default_client:
        await llm_module._default_client.aclose()
    llm_module._default_client = LLMClient(new_cfg)
    health = await llm_module._default_client.check_health()
    return {"updated": True, "health": health}