"""Model Context Protocol (MCP) Tool Integration for Context Optimizer.

Provides an MCP-compliant tool schema and call handler, enabling any MCP-capable
agent (Antigravity, Claude Code, Cursor, etc.) to invoke the optimizer directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from .config import IsolatedConfig
from .optimizer import ContextOptimizer

MCP_TOOL_DEFINITION = {
    "name": "optimize_context",
    "description": (
        "Iterates over codebase files starting from cwd or a specified root directory, "
        "evaluates relevance using TypeSafe Jev System One semantic decisions (Score, Choice, Noul), "
        "and distills a high-density, token-budgeted context bundle for AI tasks."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The target instruction, user query, or topic to optimize context for."
            },
            "root_dir": {
                "type": "string",
                "description": "Optional root directory path to scan. Defaults to the current working directory."
            },
            "token_budget": {
                "type": "integer",
                "description": "Optional token ceiling for the context bundle. Defaults to 8000."
            },
            "force_include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of relative file paths to force into the context bundle."
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "json", "jev"],
                "description": "Output format: 'markdown' (default), 'json', or 'jev' (JSON Event)."
            }
        },
        "required": ["query"]
    }
}


def handle_mcp_call(arguments: Dict[str, Any], config: Optional[IsolatedConfig] = None) -> Dict[str, Any]:
    """Execute an MCP tool call."""
    query = arguments.get("query", "")
    root_dir = arguments.get("root_dir")
    token_budget = arguments.get("token_budget")
    force_include = arguments.get("force_include")
    out_format = arguments.get("format", "markdown")

    cfg = config or IsolatedConfig.load()
    optimizer = ContextOptimizer(config=cfg, root_dir=root_dir)
    res = optimizer.optimize(query=query, token_budget=token_budget, force_include=force_include)

    if out_format == "jev":
        content = json.dumps(res.to_jev_event(), indent=2)
    elif out_format == "json":
        content = json.dumps(res.to_dict(), indent=2)
    else:
        content = res.context_markdown

    return {
        "content": [
            {
                "type": "text",
                "text": content
            }
        ]
    }
