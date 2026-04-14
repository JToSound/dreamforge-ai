# DreamForge Fix Skill
## Scope
Four targeted bug fixes derived from simulation session ec094636.
## Key invariants
- result dict MUST contain keys: segments, neurochemistry_ticks, memory_activations, memory_graph
- neurochemistry CSV MUST have columns: time_hours, ach, serotonin, ne, cortisol (960 rows)
- N3 stage proportion target: 13–23% of total ticks
- Cortisol must peak between 06:00–08:00 using asymmetric sigmoid
