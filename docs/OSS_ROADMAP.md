# DreamForge OSS Roadmap

## Product Baseline (current)
- Product-grade simulation API with metrics, error taxonomy, and async job submission.
- Narrative quality scoring and report/compare API endpoints.
- Dashboard exports and analytics charts with provenance tags.

## Next Milestones
1. **Reliability hardening**
   - Prometheus + Grafana dashboards in deployment docs.
   - Incident runbooks for provider errors, timeout bursts, and export failures.
2. **Narrative quality scale**
   - Golden-seed benchmark suite and quality threshold gates in CI.
   - Prompt/profile A/B framework.
3. **Analytics expansion**
   - Compare center parity in API + dashboard.
   - Report bundles (JSON + PNG/SVG + method appendix).
4. **Product/security**
   - RBAC roles, API key scopes, audit query endpoint.
   - Policy docs and release discipline.

## Public Contribution Themes
- `good first issue`: docs, chart polish, export UX
- `help wanted`: prompt/model ops, async queue backends, multi-run analytics
- `research`: lucidity calibration and memory-grounding evaluation
