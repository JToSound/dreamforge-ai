# DreamForge AI Copilot Instructions

## Build, test, and lint commands

| Task | Command |
|---|---|
| Install backend deps | `pip install -r requirements.txt` |
| Install dashboard deps | `pip install -r requirements.dashboard.txt` |
| Editable install | `pip install -e .` |
| Run API locally | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| Run Streamlit dashboard | `streamlit run visualization/dashboard/app.py --server.port=8501 --server.address=0.0.0.0` |
| Run Python tests | `python -m pytest -q` |
| Run a single test | `python -m pytest tests/test_bizarreness.py::test_bizarreness_rem_peak -q` |
| Lint (CI style) | `ruff check .` |
| Format check (CI style) | `black --check .` |
| Run full stack with Docker | `docker-compose up --build` |
| Build dashboard image (BuildKit path from README) | `docker compose build --progress=plain dashboard` |
| Frontend dev server | `cd web-frontend && npm install && npm run dev` |
| Frontend build | `cd web-frontend && npm run build` |

## MCP server configuration in-repo

- Playwright MCP is configured at `.vscode/mcp.json` for repository-scoped browser automation tooling.
- For Copilot CLI sessions (user-level config), add the same server via `/mcp add` with command `npx` and args `-y @playwright/mcp@latest`.

## High-level architecture

- **Primary runtime entrypoint is `api/main.py`** (Docker runs `uvicorn api.main:app`). The FastAPI app defines simulation and LLM routes inline, and stores simulation results in an in-memory `_simulations` dict.
- **Simulation core lives in `core/`**:
  - `core/models/sleep_cycle.py`: Borbély two-process staging (`WAKE/N1/N2/N3/REM`).
  - `core/models/neurochemistry.py`: ODE integration for ACh/5-HT/NE/cortisol.
  - `core/models/memory_graph.py`: NetworkX `MultiDiGraph` memory/replay model.
  - `core/simulation/engine.py`: tick loop that couples sleep, neurochemistry, replay, and dream segment generation.
- **Agent layer (`core/agents/*`)** encapsulates orchestration and generation (notably `DreamConstructorAgent` and `OrchestratorAgent`), and is used by simulation tooling scripts in `tools/`.
- **Visualization surface is Streamlit** in `visualization/dashboard/app.py`, which calls the API and renders hypnogram/neurochem/memory/dream outputs.
- **`web-frontend/` is a separate Vite+React client** with its own scripts and request/response typing.

## Key conventions in this repo

- **Treat `api/main.py` as source-of-truth for active API behavior.** `api/routes/*` and `api/schemas.py` contain alternate route/schema paths; they are not mounted by default in the current FastAPI app.
- **Preserve response-shape compatibility.** Multiple consumers handle legacy and current keys (e.g., `segments` vs `dream_segments`, `neurochemistry` vs `neurochemistry_series`, `bizarreness` vs `bizarreness_score`), so additive changes are safer than renames.
- **Use existing domain enums/models instead of raw strings where possible** (`SleepStage`, `EmotionLabel`, Pydantic models in `core/models/*`).
- **LLM paths must fail soft, not hard.** Existing code falls back to template/offline generation or sentinel strings when provider calls fail; keep simulation completion behavior intact even when LLM backends are unavailable.
- **Tests are stochastic-aware.** Current tests use bounded/range assertions and explicit seeding (for example `np.random.seed(0)` in `tests/test_bizarreness.py`) instead of exact deterministic full-output snapshots.

## Identity
You are a senior AI systems engineer and computational neuroscience developer.
Stack: Python 3.11+, Pydantic v2, FastAPI, SciPy, NetworkX, Streamlit, Plotly, asyncio.

## Core Directives
- Execute ALL tasks autonomously. Do NOT ask for confirmation.
- Run `pytest` after every modified file. Do NOT proceed if tests fail.
- Every numerical parameter must cite its source: `# Source: [Author Year] [DOI]`
- Use Black + Ruff formatting (line length 100).
- All public functions: full type hints + Google-style docstrings.
- Never use bare `except:` clauses.
- Target: pytest coverage ≥ 80% on all modified files.

## Pydantic v2 Rules (strictly enforced)
- `class Config:` → `model_config = ConfigDict(...)`
- `validator` → `field_validator`
- `.dict()` → `.model_dump()`
- `.parse_obj()` → `.model_validate()`

## File Output Standard
After every change: run syntax check → import resolution → mypy strict → pytest.
Produce a CHANGELOG entry for every file modified.