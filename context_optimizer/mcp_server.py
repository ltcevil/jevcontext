"""Model Context Protocol (MCP) Server for Context Optimizer.

Exposes `optimize_context` as a standard MCP tool over stdio transport.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

# Route all logs strictly to stderr so stdout remains 100% pure JSON-RPC
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for noisy in ["httpx", "httpcore", "httpx2", "typesafe_sdk", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from mcp.server.mcpserver import MCPServer
from .config import IsolatedConfig
from .mcp_tool import handle_mcp_call

server = MCPServer("context-optimizer")


@server.tool(
    name="optimize_context",
    description=(
        "Iterates over codebase files starting from cwd or a specified root directory, "
        "evaluates relevance using TypeSafe Jev System One semantic decisions (Score, Choice, Noul), "
        "and distills a high-density, token-budgeted context bundle for AI tasks."
    ),
)
def optimize_context(
    query: str,
    root_dir: Optional[str] = None,
    token_budget: Optional[int] = None,
    force_include: Optional[List[str]] = None,
    format: str = "markdown",
) -> str:
    """Iterates over codebase files and distills a high-density context bundle."""
    args = {
        "query": query,
        "root_dir": root_dir,
        "token_budget": token_budget,
        "force_include": force_include,
        "format": format,
    }
    result = handle_mcp_call(args, config=_CONFIG)
    if result.get("content") and len(result["content"]) > 0:
        return result["content"][0]["text"]
    return ""


# Loaded once in main() from an explicit --env-file, so the server never depends on the
# directory it was launched from (IsolatedConfig.load() otherwise resolves
# ./.env.context_optimizer / ./.optimizer.env against the cwd). None = that legacy lookup.
_CONFIG: Optional[IsolatedConfig] = None


def main(env_file: Optional[str] = None):
    """Run the MCP server over stdio transport.

    --env-file PATH pins the isolated config file (recommended for proxies/launchd).
    """
    import argparse

    global _CONFIG
    target_env = env_file
    if not target_env:
        parser = argparse.ArgumentParser(prog="context_optimizer.mcp_server")
        parser.add_argument(
            "--env-file", help="isolated env file to load (default: cwd lookup)"
        )
        args, _unknown = parser.parse_known_args()
        target_env = args.env_file

    if target_env:
        path = os.path.expanduser(target_env)
        if not os.path.isfile(path):
            print(f"context-optimizer: env file not found: {path}", file=sys.stderr)
            sys.exit(2)
        _CONFIG = IsolatedConfig.load(env_file_path=path)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
