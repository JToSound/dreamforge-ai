# Round 4 Changelog

## `core/agents/dream_constructor_agent.py`
- **Fix 2 / Fix 4**
- Enforced `/no_think` at request construction level, added JSON-parse validation, and added one retry path for malformed JSON responses before fallback.
- Added per-call narrative token budget (`narrative_max_tokens=600`) while preserving existing global config compatibility.
- Added richer neurochemical descriptor injection in prompt context via `nchem_to_descriptors(...)`.
- Added longer fallback narrative expansion to improve template quality floor.

## `core/llm_client.py`
- **Fix 2**
- Extended `LLMConfig` with `no_think`, `json_mode`, and `timeout_seconds`.
- Added conditional `response_format={"type":"json_object"}` payload support and moved `/no_think` enforcement to system message construction.

## `api/main.py`
- **Fix 3 / Fix 4 / Fix 5**
- Replaced random-only sampling with prioritized event-driven trigger routing:
  `REM_ONSET`, `LUCIDITY_THRESHOLD`, `BIZARRENESS_PEAK`, `MEMORY_SALIENCE`, `N3_ONSET`, fallback `API_SAMPLED`.
- Added trigger cooldown (8 ticks) and call cap tied to duration.
- Added richer memory graph dynamics (semantic node seeding, decay/boost updates, active-memory assignment, 10-tick snapshots).
- Ensured `memory_graph.activation_snapshots` is exported and aligned with `memory_activations`.
- Upgraded template narrative generation to longer-form outputs and de-duplicated consecutive template lines.
- Reordered response assembly so memory-derived `active_memory_ids` are reflected in returned segments.

## `core/models/sleep_cycle.py`
- **Fix 1**
- Added `HomeostaticState` with independent `sws_debt` and `rem_debt`.
- Added REM debt accumulation/discharge controls in cycle schedule construction.
- Added N2 ceiling rebalance guard that redistributes excess N2 to REM/N3 (0.7/0.3).
- Kept template architecture stable while layering homeostatic rebalancing logic.

## `core/simulation/engine.py`
- **Fix 5**
- Added active-memory assignment for non-replay segments by selecting currently high-activation nodes.

## `core/simulation/narrative_cache.py`
- **Fix 4**
- Added consecutive-duplicate avoidance in template selection.

## `core/utils/neurochemistry_descriptors.py` (new)
- **Fix 4**
- Added `nchem_to_descriptors(...)` helper for qualitative mapping of ACh/5-HT/NE/cortisol.

## `tests/test_dreamforge_fixes.py`
- **Fix 2 / Fix 3 / Fix 4**
- Added tests for:
  - trigger diversity and `API_SAMPLED` rate bound,
  - retry + fallback behavior in `DreamConstructorAgent._call_llm`,
  - template narrative quality floor (mean word count and no consecutive duplicates).

## `tests/test_fixes_v5.py`
- **Fix 1 / Fix 4 / Fix 5**
- Added coverage for:
  - cycle REM trend checks and 3-seed sleep architecture bounds,
  - neurochemical descriptor mapping utility,
  - memory-activation snapshot export behavior.
