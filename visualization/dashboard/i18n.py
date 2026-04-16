from __future__ import annotations

from typing import Dict


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "DreamForge AI — Live Simulation",
        "app_tagline": "The first open-source AI system that thinks while it sleeps.",
        "sidebar_title": "DreamForge AI",
        "sidebar_llm_config": "LLM Configuration",
        "run_simulation": "Run Simulation",
        "compare_center": "Compare & Report Center",
        "compare_baseline": "Baseline run",
        "compare_candidate": "Candidate run",
        "compare_action": "Generate Comparison",
    },
    "zh-HK": {
        "app_title": "DreamForge AI — 即時模擬",
        "app_tagline": "首個會「睡眠思考」嘅開源 AI 系統。",
        "sidebar_title": "DreamForge AI",
        "sidebar_llm_config": "LLM 設定",
        "run_simulation": "執行模擬",
        "compare_center": "比較與報告中心",
        "compare_baseline": "基準 run",
        "compare_candidate": "對比 run",
        "compare_action": "產生比較報告",
    },
}


def tr(locale: str, key: str) -> str:
    table = TRANSLATIONS.get(locale) or TRANSLATIONS["en"]
    return table.get(key) or TRANSLATIONS["en"].get(key, key)
