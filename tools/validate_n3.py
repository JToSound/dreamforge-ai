import sys
import pathlib
from collections import Counter

# Ensure project root is on sys.path for direct script execution
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.models.sleep_cycle import SleepCycleModel, SleepStage

def run_validation():
    m = SleepCycleModel()
    states, stages = m.simulate_night(duration_hours=8.0, dt_minutes=0.5)
    total = len(stages)
    cnt = Counter(stages)
    fractions = {stage.value: cnt.get(stage, 0) / total for stage in SleepStage}
    print("Total steps:", total)
    for k, v in fractions.items():
        print(f"{k}: {v:.3f} ({cnt.get(SleepStage(k),0)})")

    # Simple assertions (as per acceptance criteria)
    n3_frac = fractions.get("N3", 0.0)
    n1_frac = fractions.get("N1", 0.0)
    rem_frac = fractions.get("REM", 0.0)

    assert 0.12 <= n3_frac <= 0.25, f"N3 fraction out of range: {n3_frac}"
    assert n1_frac <= 0.12, f"N1 fraction too high: {n1_frac}"
    assert 0.18 <= rem_frac <= 0.28, f"REM fraction out of range: {rem_frac}"
    print("Validation passed: N3/N1/REM within target ranges.")

if __name__ == '__main__':
    run_validation()
