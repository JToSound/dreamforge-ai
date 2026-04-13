import sys
import pathlib
from statistics import mean

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.orchestrator import OrchestratorAgent
from scipy.stats import pearsonr
import numpy as np


def run_validation():
    orch = OrchestratorAgent()
    # Use a modest LLM cadence to generate many dream segments for analysis
    result = orch.run_night(duration_hours=8.0, dt_minutes=0.5, llm_every_n_segments=6)
    segments = result.get("dream_segments") or result.get("segments") or []

    all_biz = [float(s.get("bizarreness_score", s.get("bizarreness", 0.0))) for s in segments]
    all_luc = [float(s.get("lucidity_probability", 0.0)) for s in segments]

    if len(all_biz) < 2:
        print("Not enough segments to compute correlation.")
        return

    r, p = pearsonr(all_biz, all_luc)
    print(f"Pearson r(biz, lucidity) = {r:.3f}, p={p:.3g}")

    lucid_times = [s.get("time_hours") for s in segments if float(s.get("lucidity_probability", 0.0)) > 0.35]
    if len(lucid_times) == 0:
        print("No high-lucidity segments found; mean time undefined.")
        avg_lucid_time = None
    else:
        avg_lucid_time = mean([float(t) for t in lucid_times])
        print(f"Mean lucid time (hours): {avg_lucid_time:.3f} over {len(lucid_times)} segments")

    # Acceptance criteria
    assert abs(r) < 0.55, f"Correlation too high: {r}"
    if avg_lucid_time is not None:
        assert avg_lucid_time > 4.0, f"Lucid moments not clustered late-night: {avg_lucid_time}"

    print("Validation passed: lucidity model meets acceptance criteria.")


if __name__ == '__main__':
    run_validation()
