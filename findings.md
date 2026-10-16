# Context Optimizer - Architecture & Design Specification

## Overview
The `context-optimizer` is a modular, high-performance CLI tool and agent-triggerable service designed to distill, score, and optimize repository context for AI coding agents.

## Architecture Diagram
```
+--------------------------------------------------------------------------+
|                            context-optimizer CLI                         |
|  (CLI args, Stdin JEV events, Trigger inbox, Programmatic API, MCP Tool) |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                     Isolated Configuration Manager                       |
|   - Loads from .env.context_optimizer or --env-file                      |
|   - Strict isolation: ZERO leakage into or out of os.environ             |
|   - Manages GEMINI_API_KEY, TYPESAFE_API_KEY, budgets, thresholds        |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                     Recursive Directory Scanner (CWD)                    |
|   - Starts at cwd (os.getcwd()) or --root                                |
|   - Respects .gitignore, ignores binaries, lockfiles, node_modules, etc. |
|   - Extracts metadata, line counts, language, AST symbol outlines        |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                Jev (System One) Semantic Decision Engine                 |
|   - Based on typesafe.ai System One primitives:                          |
|     * Noul: Relevance filter P(relevant) >= threshold                    |
|     * Score: Priority rating [irrelevant, low, medium, high, critical]   |
|     * Choice: Role classification (entrypoint, logic, config, doc, test) |
|     * Noul: Guardrail & prompt injection verification                    |
|   - Dual Engine: Live TypeSafe API if key provided; autonomous           |
|     Antigravity / heuristic engine fallback with identical Jev schema    |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                Google Antigravity SDK Orchestrator                       |
|   - LocalAgentConfig with isolated API keys                              |
|   - Conversation session lifecycle & Turn management                     |
|   - Hooks: @on_session_start, @pre_turn, @post_turn, @on_compaction      |
|   - Declarative Policies: policy.workspace_only(cwd), policy.deny(...)   |
|   - Triggers: on_file_change, periodic every(), custom TriggerContext    |
|   - Features: token streaming, thought streaming, tool call streaming    |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                    Context Compactor & Serializer                        |
|   - Token budget allocator                                               |
|   - Slices critical code, summarizes medium, maps low-priority files     |
|   - Outputs Markdown, JSON, or typed JEV (JSON Event) stream             |
+--------------------------------------------------------------------------+
```

## Module Structure in `tools/context-optimizer/`
```text
tools/context-optimizer/
├── task_plan.md
├── findings.md
├── progress.md
├── requirements.txt
├── README.md
├── context_optimizer/
│   ├── __init__.py
│   ├── config.py           # Isolated environment variables & settings
│   ├── scanner.py          # Recursive directory crawler & AST symbol extractor
│   ├── jev.py              # TypeSafe Jev System One primitives & decision engine
│   ├── agent.py            # Antigravity SDK Agent, Conversation, Hooks, Policies, Triggers
│   ├── optimizer.py        # Pipeline orchestrator & context compaction algorithms
│   ├── triggers.py         # Inter-agent trigger listeners (file inbox, IPC, stdin)
│   ├── mcp_tool.py         # MCP tool definition for agent integration
│   └── cli.py              # CLI entry point (argparse / rich output)
└── bin/
    └── context-optimizer   # Executable CLI wrapper
```

## JEV (JSON Event) Protocol Schema
Input event:
```json
{
  "event_id": "evt-12345",
  "event_type": "context.optimize.request",
  "timestamp": "2026-10-16T11:45:00Z",
  "sender": "agent-orchestrator",
  "payload": {
    "query": "find authentication logic and middleware",
    "root_dir": ".",
    "token_budget": 8000,
    "strategy": "balanced"
  }
}
```

Output event:
```json
{
  "event_id": "evt-12345-response",
  "in_reply_to": "evt-12345",
  "event_type": "context.optimize.response",
  "timestamp": "2026-10-16T11:45:02Z",
  "status": "success",
  "payload": {
    "summary": "...",
    "estimated_tokens": 3450,
    "selected_files": [...],
    "context_bundle": "...",
    "jev_decisions": {
      "guardrail_passed": true,
      "relevance_scores": {...},
      "roles": {...}
    }
  }
}
```
