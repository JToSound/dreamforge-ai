# Audit Before Fixes

## Commands Executed

### 1) `ruff check . --output-format=concise` (first 80 lines)
```text
All checks passed!
```

### 2) `mypy core/ --strict --ignore-missing-imports` (first 80 lines)
```text
core\models\sleep_cycle.py:236: error: Missing type parameters for generic type "dict"  [type-arg]
core\models\sleep_cycle.py:237: error: Missing type parameters for generic type "dict"  [type-arg]
core\models\sleep_cycle.py:618: error: Missing type parameters for generic type "dict"  [type-arg]
core\models\memory_graph.py:134: error: Missing type parameters for generic type "dict"  [type-arg]
core\models\memory_graph.py:399: error: Missing type parameters for generic type "dict"  [type-arg]
core\simulation\runner.py:40: error: Need type annotation for "mem_events"  [var-annotated]
core\simulation\runner.py:87: error: Argument "stage" ... incompatible type "Any | str"; expected "SleepStage"  [arg-type]
core\simulation\runner.py:119: error: Argument "stage" ... incompatible type "Any | str"; expected "SleepStage"  [arg-type]
...
(many additional strict typing errors across core/)
```

### 3) `pytest --tb=short -q` (last 40 lines)
```text
36 passed, 1 skipped, 3 warnings in 9.13s
```

### 4) Code search equivalents for requested `grep` commands
`rg` is not available as an executable in this PowerShell shell, so repository-native code search was run with the built-in `rg` tool.

#### 4a) `memory_activations` in `core/**/*.py`
```text
core/simulation/engine.py:27:    record_memory_activations: bool = False
core/simulation/engine.py:143:            if config.record_memory_activations and (
```

#### 4b) `neurochemistry` + csv/export/write/to_csv in `core/**/*.py`
```text
No direct CSV export/write path found in core simulation modules for neurochemistry ticks.
```

#### 4c) `n3_threshold|tau_sleep` in `core/**/*.py`
```text
core/models/sleep_cycle.py:81: tau_sleep: float = Field(...)
core/models/sleep_cycle.py:329: s_new = s_prev * math.exp(-dt_hours / p.tau_sleep)
```

#### 4d) `cortisol` + `sigmoid|gaussian|peak` in `core/**/*.py`
```text
core/models/neurochemistry.py:154: def _cortisol_drive(self, time_hours: float) -> float:
core/models/neurochemistry.py:155: """Asymmetric sigmoid cortisol profile with a fixed peak time.
core/models/neurochemistry.py:161: guarantees the maximum occurs at `cortisol_rise_time`
...
```

---

## Findings

1. **Lint status**
   - Ruff passes at repository level.

2. **Type-check status (strict)**
   - `mypy --strict` currently fails with numerous pre-existing strict typing issues across core modules.
   - Hotspots include: `core/simulation/runner.py`, `core/models/*`, `core/utils/llm_adapters.py`, `core/agents/*`.

3. **Test status**
   - Baseline tests pass (`36 passed, 1 skipped`), indicating runtime behavior is currently stable for covered paths.

4. **Memory activations serialization gap**
   - `memory_activations` is referenced as a config toggle in `core/simulation/engine.py` but not broadly surfaced as a guaranteed top-level result invariant across outputs.

5. **Neurochemistry CSV export gap**
   - No direct/export-focused neurochemistry CSV writing path is discoverable in `core` via `neurochemistry + csv/export/write/to_csv` search pattern.
   - This aligns with the reported dashboard blank neurochemistry panel symptom.

6. **N3 physiology parameters**
   - `tau_sleep` exists in `core/models/sleep_cycle.py`.
   - The code currently uses `n3_s_threshold` naming (not `n3_threshold`), so targeted calibration checks should use the actual symbol present.

7. **Cortisol model**
   - Cortisol is currently modeled via an asymmetric sigmoid-style driver in `core/models/neurochemistry.py`.
   - Any further changes should preserve peak/nadir physiological constraints while maintaining existing test compatibility.
