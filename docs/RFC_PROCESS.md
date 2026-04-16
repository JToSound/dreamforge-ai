# DreamForge RFC Process

## Purpose
Use RFCs for non-trivial changes that affect API contracts, simulation behavior, model strategy, or product editions.

## Lifecycle
1. **Draft**
   - Open a GitHub issue with label `rfc`.
   - Include problem, proposal, alternatives, risks, rollout, and rollback.
2. **Review**
   - Minimum one maintainer + one domain reviewer (simulation, API, or UX).
   - Resolve open questions before implementation.
3. **Decision**
   - Status: `accepted`, `rejected`, or `deferred`.
4. **Implementation**
   - PR must link RFC issue and describe validation evidence.
5. **Post-release**
   - Add changelog note and record any follow-up actions.

## Required RFC Template Fields
- Problem statement
- Scope (in/out)
- API/data contract impact
- Migration and compatibility plan
- Operational impact (metrics, alerts, runbooks)
- Security/privacy implications
