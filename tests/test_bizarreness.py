import numpy as np

from core.utils.bizarreness_scorer import compute_bizarreness
from core.models.sleep_cycle import SleepStage

# deterministic noise for unit tests
np.random.seed(0)


def test_bizarreness_rem_peak():
    # High ACh, low NE, high arousal in REM should produce high bizarreness
    score = compute_bizarreness(
        stage=SleepStage.REM,
        ach_level=0.9,
        ne_level=0.05,
        memory_arousal=0.8,
        cycle_index=3,
    )
    assert 0.7 <= score.total_score <= 0.95


def test_bizarreness_nrem_trough():
    # Moderate ACh/NE/arousal in N2 should yield low-moderate bizarreness
    score = compute_bizarreness(
        stage=SleepStage.N2,
        ach_level=0.45,
        ne_level=0.3,
        memory_arousal=0.2,
        cycle_index=1,
    )
    assert 0.05 <= score.total_score <= 0.45
