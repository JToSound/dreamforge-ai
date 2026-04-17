# Contributing to DreamForge AI

## Development setup

1. Clone the repository and create a virtual environment for Python 3.11+.
2. Install backend dependencies:
   - `pip install -r requirements.txt`
   - `pip install -r requirements.dashboard.txt`
3. Optional editable install:
   - `pip install -e .`
4. Frontend setup:
   - `cd web-frontend && npm install`

## Local run

- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Dashboard: `streamlit run visualization/dashboard/app.py --server.port=8501 --server.address=0.0.0.0`
- Frontend: `cd web-frontend && npm run dev`

## Quality gates

Run these before opening a PR:

- `ruff check .`
- `black --check .`
- `mypy --strict core/`
- `python -m pytest -q`
- `cd web-frontend && npm run build`

## API and contract rules

- `api/main.py` is the active API source of truth.
- Prefer additive API changes over breaking renames to preserve compatibility.
- Keep `/api/v1/*` aliases aligned with primary endpoints.

## Pull requests

1. Keep changes focused and include tests for new behavior.
2. Update `README.md` and `CHANGELOG.md` for user-visible changes.
3. Include before/after behavior notes in the PR description.
