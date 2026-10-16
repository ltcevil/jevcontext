# Context Optimizer CLI - Task Plan

## Goal
Build a standalone CLI tool (`context-optimizer`) acting as an intelligent context optimizer that:
1. Operates from the current working directory (`os.getcwd()`), iterating recursively over all relevant files and subdirectories.
2. Accepts and processes user input / query / task specification.
3. Uses the Google Antigravity Python SDK (`google-antigravity`), specifically leveraging:
   - Conversation handling (`Conversation`)
   - Features (e.g. streaming, tool integration, system instructions)
   - Hooks and Policies (lifecycle hooks, safety policies, validation)
   - Triggers (periodic/event triggers or trigger interface so another agent can trigger it)
4. Uses its OWN isolated environment variables / configuration (NOT polluted by or polluting global env).
5. Exposes trigger interfaces so other agents can trigger it (e.g., via CLI subcommands, programmatic API, MCP tool definition, JSON event / JEV).
6. Integrates `jev` based on typesafe.ai JSON Event Specification / Use Case Map (`https://docs.typesafe.ai/concepts/use-case-map`).

## Status Summary
- **Current Phase**: Complete
- **Progress**: 100% complete
- **Last Updated**: 2026-10-16

## Phases

### Phase 1: Requirements Research & Reference Gathering
- [x] Inspect Antigravity SDK references (`references/architecture.md`, `references/agent_configuration.md`, `references/safety_policies.md`, `examples/getting_started/periodic_trigger.md`, `examples/getting_started/hooks.md`, etc.).
- [x] Research typesafe.ai Use Case Map and `jev` (JSON Event / typesafe schema / event stream specs).
- [x] Inspect Python runtime and dependencies rule.
- [x] Record findings in `findings.md`.
- *Status*: `complete`

### Phase 2: Architecture & Design Specification
- [x] Design the CLI tool interface, flags, isolated env loader (`.env` in tool dir or custom config file, custom prefix, isolated `dict`/environ overlay).
- [x] Design directory scanner with ignore filters (`.git`, `node_modules`, `venv`, binaries, size limits).
- [x] Design context optimization pipeline (summarization, relevance scoring, token compression, AST/symbol extraction, file ranking).
- [x] Design agent integration: Antigravity SDK agent with custom hooks, safety policies, conversation, triggers.
- [x] Design `jev` (JSON Event) integration for structured agent-to-agent triggering, event emission, and use-case mapping.
- *Status*: `complete`

### Phase 3: Implementation
- [x] Setup project structure in `tools/context-optimizer/` (`requirements.txt`, virtualenv, entrypoint script, module package).
- [x] Implement isolated environment configuration manager (`config.py`).
- [x] Implement recursive file iterator and smart content filter / AST/chunk parser (`scanner.py`).
- [x] Implement `jev` System One primitives and decision engine (`jev.py`).
- [x] Implement Antigravity SDK agent wrapper with conversation, hooks, policies, and triggers (`agent.py`).
- [x] Implement context optimizer core engine (`optimizer.py`).
- [x] Implement agent trigger listeners & JEV protocol (`triggers.py`).
- [x] Implement MCP tool definition (`mcp_tool.py`).
- [x] Implement CLI entrypoint (`cli.py`, `__main__.py`, executable wrapper).
- *Status*: `complete`

### Phase 4: Testing & Verification
- [x] Create unit tests and integration tests for scanner, env isolation, JEV protocol, and agent triggers (`tests/test_optimizer.py`).
- [x] Test CLI directly on test directories and the current repository.
- [x] Verify isolated environment behavior without global env leakage.
- [x] Verify agent invocation / triggering works cleanly.
- *Status*: `complete`

### Phase 5: Documentation & Integration
- [x] Document usage, configuration, environment variables, JEV integration, and agent invocation examples in `README.md`.
- [x] Update `tools/README.md` to reference `context-optimizer`.
- *Status*: `complete`

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `cannot import name 'types' from 'google.antigravity.hooks'` | 1 | Imported `types` directly from `google.antigravity` (`google.antigravity.types`) |
| `line_count` test failure (`count("\n") + 1` overcounted trailing newline) | 1 | Switched to `len(raw_text.splitlines())` for precise line counting |
| `400 Unknown model: jev` from live TypeSafe API | 1 | Normalized model name default to `jev-latest`, parsed continuous float `score` from Jev `ScoreAnswer`, and added `cto` symlink in `.local/bin` |
