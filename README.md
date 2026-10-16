# Context Optimizer (`context-optimizer`)

An intelligent, token-budgeted codebase context optimizer CLI tool designed for AI agents, built with the **Google Antigravity Python SDK** and **TypeSafe Jev System One** semantic decision engine.

Operates directly on the current working directory (`os.getcwd()`), recursively indexes all files and subdirectories, parses AST symbol structures, enforces strict environment variable isolation, and provides multiple trigger interfaces so other agents can seamlessly invoke it.

---

## Key Features

1. **Working Directory Crawling & AST Outline Extraction**
   - Starts at `cwd` (or `--root <path>`).
   - Recursively scans files and subdirectories while respecting `.gitignore` and ignore filters (`.git`, `node_modules`, `__pycache__`, `.venv`, binaries, lockfiles, images).
   - Extracts character counts, line counts, estimated token usage, and structural symbols (classes, functions, methods, docstrings, imports) using AST for Python and regex for TypeScript/JavaScript/Go/Rust.

2. **Strict Environment Variable Isolation**
   - Uses its **own isolated environment configuration** (`~/.config/context-optimizer/env` or `.env.context_optimizer` or `--env-file`).
   - **Zero leakage**: Does NOT pollute `os.environ` or read ambient global variables by default, guaranteeing credentials (`GEMINI_API_KEY`, `TYPESAFE_API_KEY`) remain isolated.

3. **TypeSafe Jev (System One) Integration**
   - Implements the System One primitives from [TypeSafe Use Case Map](https://docs.typesafe.ai/concepts/use-case-map):
     - **Detection / Guardrail**: `Noul` verifies prompts against injection, jailbreaks, and credential leaks.
     - **Search & Retrieval**: `Noul` evaluates binary hypothesis $P(\text{relevant})$.
     - **Scoring**: `Score` rates file relevance on descriptive levels (`["irrelevant", "mention", "relevant", "critical"]`).
     - **Role Classification**: `Choice` categorizes file architectural role (`["entrypoint", "core_logic", "data_model", "utility", "config", "test", "docs"]`).
     - **Routing & Inclusion Strategy**: `Choice` determines window representation (`["full_content", "outline_and_excerpts", "manifest_entry", "exclude"]`).
   - **Dual-Engine Architecture**: Connects to the live TypeSafe Jev API when `TYPESAFE_API_KEY` is configured, with a built-in calibrated fallback engine producing identical structured outputs when running offline or without external credentials.

4. **Google Antigravity SDK Integration**
   - **Advanced Conversation**: Full stateful session tracking with `Conversation.create(strategy)`, `conversation.chat()`, `conversation.history`, `conversation.turn_count`, and `conversation.receive_steps()`.
   - **Features**: Real-time token streaming (`async for token in response:`), reasoning thoughts streaming (`async for thought in response.thoughts:`), and tool call streaming (`async for call in response.tool_calls:`).
   - **Lifecycle Hooks**: Intercepts `@hooks.on_session_start`, `@hooks.on_session_end`, `@hooks.pre_turn`, `@hooks.post_turn`, `@hooks.on_compaction`, and `@hooks.pre_tool_call_decide`.
   - **Declarative Safety Policies**: Enforces `policy.workspace_only([cwd])`, `policy.deny("run_command")`, and `policy.allow("view_file")`.
   - **Background Triggers**: Integrates `google.antigravity.triggers` (`every`, `on_file_change`, and `TriggerContext.send`).

5. **Inter-Agent Trigger Interfaces**
   - **CLI Trigger Subcommand**: `context-optimizer trigger --payload '...'`
   - **Standard Input / Pipe (JEV Protocol)**: `cat request.json | context-optimizer trigger --stdin`
   - **File Inbox Watcher Daemon**: `context-optimizer listen` (watches `.context_optimizer/inbox` and outputs to `.context_optimizer/outbox`)
   - **Programmatic Python API**: `from context_optimizer import optimize_context`
   - **Model Context Protocol (MCP)**: Exposes tool definition schema via `context-optimizer mcp` and handler `handle_mcp_call()`.

---

## Installation & Setup

```bash
cd tools/context-optimizer

# Virtual environment and dependencies are automatically managed:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Executable binary:
```bash
bin/context-optimizer --help
```

---

## Configuration & Environment Isolation

Configure credentials without polluting your shell's global environment:

```bash
# Set isolated Gemini API key
bin/context-optimizer config set GEMINI_API_KEY="AIzaSy..."

# Set isolated TypeSafe API key
bin/context-optimizer config set TYPESAFE_API_KEY="ts_live_..."

# View current isolated configuration
bin/context-optimizer config show

# Show path to isolated environment file
bin/context-optimizer config path
# ~/.config/context-optimizer/env
```

---

## CLI Usage

### 1. Optimize Context (Interactive)

Run in any repository or directory to extract minimal, high-density context:

```bash
# Scan current working directory for authentication logic
bin/context-optimizer optimize "find authentication and session logic"

# Scan a specific directory with a custom token budget
bin/context-optimizer optimize "database models and migrations" --root /path/to/repo --budget 4000

# Force-include critical files
bin/context-optimizer optimize "api routes" --force-include src/app.py config.json

# Save output to Markdown file
bin/context-optimizer optimize "error handling" -o context_bundle.md

# Output in JSON or JEV (JSON Event) format
bin/context-optimizer optimize "billing checkout" --format jev
```

### 2. Trigger from Another Agent (JEV Protocol)

Agents can pipe JSON events to `context-optimizer trigger`:

```bash
echo '{
  "id": "agent-turn-42",
  "data": {
    "query": "find configuration manager and environment loader",
    "token_budget": 6000
  }
}' | bin/context-optimizer trigger --stdin
```

Sample output:
```json
{
  "specversion": "1.0",
  "id": "jev-ctx-1792144264133",
  "in_reply_to": "agent-turn-42",
  "type": "ai.typesafe.context.optimized",
  "source": "context-optimizer-cli",
  "time": "2026-10-16T09:53:04Z",
  "datacontenttype": "application/json",
  "data": {
    "query": "find configuration manager and environment loader",
    "guardrail": {
      "passed": true,
      "confidence": 0.95,
      "message": "Pass"
    },
    "stats": {
      "total_scanned": 10,
      "total_included": 3,
      "estimated_tokens": 2840,
      "max_budget": 6000,
      "duration_seconds": 0.045
    },
    "selected_files": [
      {
        "path": "context_optimizer/config.py",
        "role": "config",
        "score": "critical",
        "probability": 0.95,
        "strategy": "full",
        "tokens": 1761
      }
    ],
    "context_markdown": "# Optimized Context for: `find configuration manager`\n..."
  }
}
```

### 3. Agent Inbox Listener Daemon

Run a background daemon watching for dropped request files from peer agents:

```bash
bin/context-optimizer listen --interval 2
```
- Peer agents drop JSON request files into `.context_optimizer/inbox/`.
- The daemon processes them, writes responses to `.context_optimizer/outbox/`, and emits Antigravity trigger notifications.

### 4. MCP Tool Schema

To register `context-optimizer` in any MCP host or Claude/Antigravity configuration:

```bash
bin/context-optimizer mcp
```

---

## Python API Usage

Other Python agents can import and run the optimizer directly:

```python
import asyncio
from context_optimizer import ContextOptimizer, optimize_context

# Simple one-liner
result = optimize_context(
    query="find auth middleware and user model",
    root_dir=".",
    token_budget=5000
)

print(f"Included {result.total_files_included} files, ~{result.estimated_tokens} tokens")
print(result.context_markdown)

# Advanced usage with Antigravity Agent & Conversation
from context_optimizer.agent import AntigravityAgentManager
from context_optimizer.config import IsolatedConfig

config = IsolatedConfig.load()
manager = AntigravityAgentManager(config=config)

async def stream_agent_thoughts():
    response_text = await manager.chat_stream(
        prompt="Synthesize a minimal API boundary from the optimized files.",
        on_thought=lambda thought: print(f"Agent thought: {thought}", flush=True),
        on_token=lambda token: print(token, end="", flush=True)
    )

asyncio.run(stream_agent_thoughts())
```

---

## Running Tests

```bash
cd tools/context-optimizer
PYTHONPATH=. .venv/bin/pytest tests/test_optimizer.py -v
```
All 8 unit and integration tests verify:
- Isolated environment prevents global environment leaks.
- Recursive crawler honors ignore filters and extracts symbols.
- Jev System One guardrails, Noul probabilities, and Score classifications.
- Strict token budget compaction.
- Inter-agent JEV event execution and inbox file watcher.
- Antigravity agent configuration, hooks, and safety policies.
- MCP tool schema and handler execution.
