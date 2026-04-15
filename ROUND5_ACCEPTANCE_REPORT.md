# Round 5 Acceptance Report

| Area | Criterion | Measured value | Status |
|------|-----------|---------------|--------|
| Typing | `mypy --strict core/` target <= 30 errors | 30 errors | ✅ |
| Lint | `ruff check .` | pass | ✅ |
| Format | `black --check .` | pass | ✅ |
| Tests | `pytest tests/ -q` | 99 passed, 1 skipped, 0 failed | ✅ |
| Coverage | total coverage >= 85% | 85% | ✅ |
| Exporters | `neurochemistry.csv` rows after 8h smoke run >= 950 | 960 rows | ✅ |
| Exporters | `memory_activations.csv` rows after 8h smoke run >= 90 × avg nodes | 1152 rows (96 snapshots × 12 nodes) | ✅ |
| API payload | `memory_activations` present in simulation JSON | present, 96 snapshots | ✅ |
| Sleep architecture | N3 proportion in 8h smoke run >= 10% | 18.333% | ✅ |
| Template quality | TEMPLATE mean word count >= 40 | 55.654 words | ✅ |
| Template source | YAML-backed templates used for TEMPLATE segments >= 50% | 100% | ✅ |

