from core.models.neurochemistry import NeurochemistryModel, NeurochemistryParameters
from core.models.sleep_cycle import SleepStage


def test_ach_saturation():
    params = NeurochemistryParameters(
        ach_max=0.8, ach_saturating=True, r_ach_rem=2.0, k_clear_ach=1.0
    )
    model = NeurochemistryModel(params=params)
    state = model.initial_state(time_hours=0.0, ach=0.7)

    # constant REM stage function
    def stage_fn(t):
        return SleepStage.REM

    traj, _ = model.integrate(state, t_end=1.0, stage_fn=stage_fn, max_step=0.1)
    final = traj[-1]
    # final ACh should not exceed ach_max (allow small numeric tolerance)
    assert final.ach <= params.ach_max + 1e-6


def test_cortisol_asymmetry():
    # Use sigmoid-based cortisol rise centered at hour 6.0 with moderate steepness
    params = NeurochemistryParameters(
        cortisol_rise_time=6.0, cortisol_k_rise=6.0, cortisol_k_fall=1.0
    )
    model = NeurochemistryModel(params=params)
    p = params.cortisol_rise_time

    # sample points nearer the inflection to capture asymmetric sigmoid rise/fall
    v_left_far = model._cortisol_drive(p - 0.5)
    v_left_near = model._cortisol_drive(p - 0.1)
    v_right_near = model._cortisol_drive(p + 0.1)
    v_right_far = model._cortisol_drive(p + 0.5)

    # rise should be steeper immediately before midpoint than the fall immediately after
    rise_delta = abs(v_left_near - v_left_far)
    fall_delta = abs(v_right_far - v_right_near)
    assert rise_delta > fall_delta
