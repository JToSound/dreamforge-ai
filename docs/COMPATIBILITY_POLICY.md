# DreamForge Compatibility Policy

This policy defines what compatibility guarantees DreamForge provides for API consumers and operators.

## 1. Compatibility scope

DreamForge compatibility commitments apply to:

1. Public HTTP API routes documented in `README.md` and `/docs`.
2. Public response fields in `api/main.py` Pydantic response models.
3. Benchmark pack output schema from `scripts/benchmark_pack.py`.
4. Artifact manifest schema in `artifacts/manifest.json`.

Internal helper functions and private fields (`_internal*`, implementation-only keys) may change without notice.

## 2. Versioning rules

1. `api_contract` is currently `v1` and is exposed by `/api/version`.
2. Breaking response-shape changes must ship behind a new contract version (for example `v2`) or keep additive compatibility in `v1`.
3. New response fields are additive by default and should not remove or rename existing fields in the same major contract.

## 3. Async job schema

Async job payloads include a dedicated `schema_version` field (current value `v2`).

1. `schema_version` changes when required fields are removed/renamed or when semantic meaning changes.
2. Additive fields do not require a schema bump.
3. Clients should ignore unknown fields for forward compatibility.

## 4. Dependency/runtime support window

1. Python baseline: `>=3.11` (see `pyproject.toml` and CI matrix).
2. Runtime behavior may differ by LLM provider; fail-soft fallback semantics are part of compatibility.
3. Artifact integrity and presence checks are surfaced via `/api/artifacts/health`.

## 5. Change process

Every compatibility-impacting change must include:

1. `CHANGELOG.md` entry.
2. Regression tests for old and new payload expectations.
3. Deprecation notice (if applicable) following `docs/DEPRECATION_POLICY.md`.
