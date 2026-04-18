# DreamForge Deprecation Policy

This policy defines how DreamForge deprecates API surfaces safely.

## 1. Deprecation principles

1. Prefer additive changes over destructive changes.
2. Keep old fields/routes available during a published deprecation window.
3. Emit clear migration guidance before removal.

## 2. Required deprecation lifecycle

1. **Announce**
   - Mark route/field as deprecated in docs and changelog.
   - Include replacement API and migration example.
2. **Dual support**
   - Maintain old + new behavior in parallel.
   - Add tests proving both paths still work.
3. **Removal**
   - Remove old behavior only after the deprecation window and changelog notice.

## 3. API contract behavior

1. Breaking removals in `v1` are not allowed without introducing a new contract version.
2. During dual support, return both legacy and new keys when practical.
3. For async job payloads, preserve `schema_version` and maintain backward-safe semantics.

## 4. Communication requirements

Every deprecation must include:

1. Changelog entry with:
   - affected routes/fields,
   - first deprecated release,
   - planned removal release.
2. Documentation updates in `README.md` and relevant docs under `docs/`.
3. Issue template references for reporting migration blockers.
