# DreamForge AI — Agent Behavior Contract

## Execution Mode

Fully automated. Sequential phase execution. No human confirmation required.

## Tool Usage

- Use filesystem MCP for all file reads/writes
- Use git MCP to commit after each phase with descriptive messages
- Use /plan before any phase with >3 file changes
- Use /fleet for independent parallel tasks (e.g., tests + docs)

## Error Handling

- Log ALL exceptions with: type, message, file:line
- Never swallow exceptions silently
- If a fix breaks a test, fix the test AND the code

## Success Gate (per phase)

Must pass before proceeding to next phase:

- All syntax checks pass
- All imports resolve
- All existing tests still pass
- New tests added for changed code
