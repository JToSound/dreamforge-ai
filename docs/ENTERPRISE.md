# DreamForge Enterprise Surface

## Product intent
DreamForge keeps the simulation core open-source while layering enterprise controls for regulated, team-based, and SLA-backed usage.

## Enterprise scope
- Role-based access and scoped tokens for operational control.
- Audit event query surface for compliance and incident investigations.
- SLA and support pathways via dedicated enterprise onboarding links.

## API surface
- `GET /api/enterprise`
  - Returns active editions, value propositions, and configured conversion links.
- `GET /api/audit/events`
  - Returns recent audit entries.
  - Requires `audit:read` scope when auth is enabled.

## Conversion configuration
Set the following environment variables in deployment:
- `ENTERPRISE_WAITLIST_URL`
- `PRO_TRIAL_URL`
- `ENTERPRISE_SLA_URL`

## Edition posture
- **Community**: simulation API + dashboard baseline.
- **Pro**: collaboration, advanced analytics workflows.
- **Enterprise**: security controls, auditability, SLA-backed operations.
