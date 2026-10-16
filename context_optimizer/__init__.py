"""Context Optimizer Package.

Intelligent codebase context optimizer powered by Google Antigravity SDK
and TypeSafe Jev System One semantic decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from .config import IsolatedConfig
from .scanner import DirectoryScanner, ScannedFile, SymbolInfo
from .jev import JevContextEngine, FileJevDecision
from .agent import AntigravityAgentManager
from .optimizer import ContextOptimizer, OptimizationResult, OptimizedFileBundle
from .triggers import JevRequestEvent, JevTriggerPayload, TriggerInboxWatcher, execute_trigger_request
from .mcp_tool import MCP_TOOL_DEFINITION, handle_mcp_call


def optimize_context(
    query: str,
    root_dir: Optional[str | Path] = None,
    token_budget: Optional[int] = None,
    force_include: Optional[List[str]] = None,
    env_file_path: Optional[str | Path] = None
) -> OptimizationResult:
    """Convenience functional API for other agents to optimize context programmatically."""
    config = IsolatedConfig.load(env_file_path=env_file_path)
    optimizer = ContextOptimizer(config=config, root_dir=root_dir)
    return optimizer.optimize(query=query, token_budget=token_budget, force_include=force_include)


__all__ = [
    "ContextOptimizer",
    "OptimizationResult",
    "OptimizedFileBundle",
    "IsolatedConfig",
    "DirectoryScanner",
    "ScannedFile",
    "SymbolInfo",
    "JevContextEngine",
    "FileJevDecision",
    "AntigravityAgentManager",
    "JevRequestEvent",
    "JevTriggerPayload",
    "TriggerInboxWatcher",
    "execute_trigger_request",
    "MCP_TOOL_DEFINITION",
    "handle_mcp_call",
    "optimize_context"
]
