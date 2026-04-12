**Simulation Analysis — ce9c09ce-b94b-4efd-8f01-d2adc9686fba**

- **Source JSON:** uploaded file (original path provided by user)
- **Analysis output:** [uploaded_sim_analysis.json](uploaded_sim_analysis.json)
- **Analysis script:** [tools/analyze_uploaded_sim.py](tools/analyze_uploaded_sim.py)

**1) Quick summary**
- Segments: 960
- Stage distribution: N2 55.10%, REM 22.81%, N1 17.29%, N3 4.79%

**2) Neurochemistry (means per stage)**
- ACh: REM ≈ 0.900, N1 ≈ 0.502, N2 ≈ 0.449, N3 ≈ 0.299
- 5-HT (serotonin): REM ≈ 0.050, N1 ≈ 0.299, N2 ≈ 0.279, N3 ≈ 0.202
- NE: REM ≈ 0.050, N1 ≈ 0.300, N2 ≈ 0.280, N3 ≈ 0.197
- Cortisol: REM ≈ 0.502, N1 ≈ 0.399, N2 ≈ 0.379, N3 ≈ 0.348

**3) Bizarreness & emotions**
- Bizarreness means: REM ≈ 0.904, N2 ≈ 0.605, N1 ≈ 0.616, N3 ≈ 0.564
- Dominant emotion counts: melancholic 150, anxious 338, joyful 100, neutral 100, curious 94, fearful 131, serene 47

**4) Cortisol circadian shape (simple test)**
- First-half mean cortisol ≈ 0.39895
- Second-half mean cortisol ≈ 0.41846
- Delta (second - first) ≈ +0.01951 (small upward shift in second half)

**5) Observations**
- ACh is very high during REM relative to other stages (expected qualitatively), but magnitude (~0.90) may be larger than intended; this amplifies REM-driven bizarreness.
- Serotonin and NE are strongly suppressed in REM (≈0.05), consistent with REM physiology, but confirm these floor values are intended.
- Cortisol shows a small upward trend toward later night — not the large dawn spike one might expect; consider changing cortisol drive to an asymmetric function that rises more sharply near wake time.
- Bizarreness is much higher in REM (expected). N1/N2 also show moderate bizarreness — consider tying bizarreness more tightly to ACh × recent_replay_strength to increase effect size and variance.
- Segment count and stage proportions look plausible for an 8-hour simulation, though N3 appears low (≈4.8%). Verify sleep cycle parameters if more N3 is desired.

**6) Recommended quick fixes (P0/P1)**
- P0: Verify that the engine uses a time→stage lookup for neuro ODE integration (already implemented). Add unit test asserting ACh_mean_REM > ACh_mean_N2 > ACh_mean_N3.
- P1: Scale down REM ACh production coefficient (or add nonlinear saturation) to bring ACh means into intended bounds.
- P1: Replace cortisol cosine with an asymmetric dawn-rise profile (e.g., logistic ramp or skewed cosine) to produce a stronger morning cortisol rise.
- P1: Compute bizarreness = clamp(alpha * ACh + beta * replay_strength + noise, 0, 1) and tune alpha/beta so REM dominates but N2/N1 are lower.
- P1: If `active_memory_ids` are frequently empty, ensure replay sampling triggers `apply_replay_effect()` and that replay selection probability is sufficient.

**7) Next steps (suggested)**



**New diagnostics & plots added**
- ACh-by-stage boxplot: `reports/plots/ce9c09ce-b94b-4efd-8f01-d2adc9686fba_ach_by_stage.png`
- ACh vs Biz scatter: `reports/plots/ce9c09ce-b94b-4efd-8f01-d2adc9686fba_ach_vs_biz.png`
- Bizarreness histogram: `reports/plots/ce9c09ce-b94b-4efd-8f01-d2adc9686fba_biz_hist.png`
- Diagnostics JSON (ACh↔Biz correlations): `reports/diagnostics/diagnostics_ce9c09ce-b94b-4efd-8f01-d2adc9686fba.json`

Key diagnostic result: overall Pearson correlation between ACh and bizarreness ≈ 0.918 (strong positive). Per-stage correlations are weaker and in some stages negative — this indicates ACh is a dominant driver overall but stage-level relationships vary; see `reports/diagnostics/...json` for details.
**Notes & artifacts**
- Analysis JSON: [uploaded_sim_analysis.json](uploaded_sim_analysis.json)
- Script used: [tools/analyze_uploaded_sim.py](tools/analyze_uploaded_sim.py)

**8) Code changes I applied**
- `core/models/neurochemistry.py`: added ACh saturation (`ach_max`, `ach_saturating`) and asymmetric cortisol rise/fall (`cortisol_rise_sigma`, `cortisol_fall_sigma`), and applied saturating ACh production.
- `core/simulation/engine.py`: added configurable bizarreness parameters and deterministic post-processing to compute `bizarreness_score` = clamp(alpha*ACh + beta*replay_strength + noise).
- `api/main.py`: adjusted fallback bizarreness formula to weight ACh primarily and reduce monoamine-only sensitivity.
- `tools/plot_sim.py`: new plotting helper to create hypnogram and bizarreness/chemistry plots from a simulation JSON.
- `tests/test_neuro_and_cortisol.py`: unit tests for ACh saturation and cortisol asymmetry (both passing locally).

**Run locally**
1. Run unit tests:

```bash
python -m pytest -q
```

2. Generate plots for a simulation JSON:

```bash
python tools/plot_sim.py /path/to/dreamforge-sim-...json
```

Generated plots for the uploaded JSON are in `reports/plots/`.

If you'd like, I can prepare a PR branch with these changes and a short changelog.

**Follow-up patch (tuning & plots)**

- Tuned REM ACh handling: added `ach_rem_scale` (default 0.85) so REM ACh production can be downscaled without changing other stage params. Location: `core/models/neurochemistry.py`.
- Increased default `cortisol_amplitude` to 0.7 and sharpened `cortisol_rise_sigma` to 0.9 to produce a stronger, earlier dawn rise. Location: `core/models/neurochemistry.py`.
- Added cortisol time-series plot and bizarreness histogram to `tools/plot_sim.py` and saved outputs under `reports/plots/`.
- Quick tuning knobs: edit `NeurochemistryParameters.ach_rem_scale`, `cortisol_amplitude`, and `cortisol_rise_sigma` to adjust behaviour.

Command to regenerate plots after tuning:

```bash
python tools/plot_sim.py /path/to/your-simulation.json
```

If you want, I can open a branch `feat/tune-chemistry-plots` and push these changes for review."

