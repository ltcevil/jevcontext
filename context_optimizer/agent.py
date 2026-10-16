"""Google Antigravity SDK Agent and Conversation Integration.

Implements advanced agent features, conversation lifecycle, hooks, declarative
policies, and triggers using the google-antigravity Python SDK.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from google.antigravity import (
    Agent,
    AgentBehavior,
    CapabilitiesConfig,
    GeminiAPIEndpoint,
    LocalAgentConfig,
    ModelTarget,
    ModelType,
    VertexEndpoint,
    types as hook_types,
)
from google.antigravity.connections.local import LocalConnectionStrategy
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.hooks import hooks, policy
from google.antigravity.tools.tool_runner import ToolRunner
from google.antigravity.triggers import TriggerContext, every, on_file_change

from .config import IsolatedConfig

logger = logging.getLogger("context_optimizer.agent")


class AntigravityAgentManager:
    """Manages the Antigravity Agent lifecycle, Conversation, Hooks, Policies, and Triggers."""

    def __init__(self, config: IsolatedConfig, workspace_dir: Optional[str | Path] = None):
        self.config = config
        self.workspace_dir = Path(workspace_dir or Path.cwd()).resolve()
        self.session_events: List[str] = []
        self._conversation: Optional[Conversation] = None
        self._agent: Optional[Agent] = None

    def build_system_instructions(self) -> str:
        """Construct high-density system instructions tailored for context optimization."""
        return (
            "You are an expert Context Optimization Agent and Codebase Architect. "
            "Your task is to analyze candidate code files, semantic scores, and AST outlines "
            "to distill minimal, high-density, and non-redundant context bundles. "
            "Eliminate dead code, unnecessary boilerplates, and out-of-scope files. "
            "Focus strictly on interfaces, essential logic, state definitions, and data flow."
        )

    def build_policies(self) -> List[Any]:
        """Construct declarative safety policies following SDK best practices."""
        # 1. Strictly restrict file operations to the current workspace
        workspace_policy = policy.workspace_only([str(self.workspace_dir)])
        
        # 2. Deny arbitrary shell commands for safety
        deny_shell = policy.deny("run_command")
        
        # 3. Allow read-only file inspections
        allow_view = policy.allow("view_file")

        return [workspace_policy, deny_shell, allow_view]

    def build_hooks(self) -> List[Any]:
        """Register lifecycle hooks for auditing and turn control."""
        mgr = self

        @hooks.on_session_start
        async def on_start():
            mgr.session_events.append("session_start")
            logger.info("Antigravity Agent session started.")

        @hooks.on_session_end
        async def on_end():
            mgr.session_events.append("session_end")
            logger.info("Antigravity Agent session ended.")

        @hooks.pre_turn
        async def pre_turn(data: str) -> hook_types.HookResult:
            mgr.session_events.append(f"pre_turn:{len(data)}")
            logger.debug(f"Intercepted prompt for pre_turn audit ({len(data)} chars).")
            return hook_types.HookResult(allow=True)

        @hooks.post_turn
        async def post_turn(data: str):
            mgr.session_events.append(f"post_turn:{len(data)}")
            logger.debug(f"Turn completed ({len(data)} chars response).")

        @hooks.on_compaction
        async def on_compaction(data):
            mgr.session_events.append("compaction_event")
            logger.info("Context compaction event intercepted in conversation.")

        @hooks.pre_tool_call_decide
        async def pre_tool(data: hook_types.ToolCall) -> hook_types.HookResult:
            logger.debug(f"Pre-tool call validation: {data.name}")
            return hook_types.HookResult(allow=True)

        return [on_start, on_end, pre_turn, post_turn, on_compaction, pre_tool]

    def build_agent_config(
        self,
        custom_triggers: Optional[List[Any]] = None,
        behavior: AgentBehavior = AgentBehavior.AUTONOMOUS
    ) -> LocalAgentConfig:
        """Build strongly-typed LocalAgentConfig with isolated environment credentials."""
        kwargs: Dict[str, Any] = {
            "system_instructions": self.build_system_instructions(),
            "policies": self.build_policies(),
            "hooks": self.build_hooks(),
            "workspaces": [str(self.workspace_dir)],
            "capabilities": CapabilitiesConfig(
                agent_behavior=behavior,
            )
        }

        # Isolate credentials: pass explicitly if provided, avoiding global env reads
        if self.config.use_vertex:
            kwargs["vertex"] = True
            if self.config.gcp_project:
                kwargs["project"] = self.config.gcp_project
            if self.config.gcp_location:
                kwargs["location"] = self.config.gcp_location
        elif self.config.gemini_api_key:
            kwargs["api_key"] = self.config.gemini_api_key

        model_name = self.config.gemini_model or "gemini-2.5-flash"
        kwargs["model"] = model_name

        if custom_triggers:
            kwargs["triggers"] = custom_triggers

        return LocalAgentConfig(**kwargs)

    @asynccontextmanager
    async def open_conversation(self) -> AsyncIterator[Conversation]:
        """Context manager to create, manage, and cleanly exit a Conversation session."""
        tool_runner = ToolRunner()
        model_name = self.config.gemini_model or "gemini-2.5-flash"

        if self.config.use_vertex:
            endpoint = VertexEndpoint(
                project=self.config.gcp_project,
                location=self.config.gcp_location or "us-central1",
            )
        else:
            endpoint = GeminiAPIEndpoint(
                api_key=self.config.gemini_api_key or ""
            )

        model_target = ModelTarget(
            name=model_name,
            types=[ModelType.TEXT],
            endpoint=endpoint,
        )

        strategy = LocalConnectionStrategy(
            tool_runner=tool_runner,
            models=[model_target],
            system_instructions=self.build_system_instructions(),
            workspaces=[str(self.workspace_dir)],
            policies=self.build_policies(),
        )
        async with Conversation.create(strategy) as conversation:
            self._conversation = conversation
            yield conversation

    async def create_conversation(self) -> Conversation:
        """Create and initialize a stateful Conversation session with LocalConnectionStrategy."""
        ctx = self.open_conversation()
        conv = await ctx.__aenter__()
        self._conversation_cm = ctx
        self._conversation = conv
        return conv

    async def close_conversation(self) -> None:
        """Cleanly close the active Conversation session."""
        if hasattr(self, "_conversation_cm") and self._conversation_cm:
            await self._conversation_cm.__aexit__(None, None, None)
            self._conversation_cm = None
            self._conversation = None

    async def chat_stream(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_thought: Optional[Callable[[str], None]] = None
    ) -> str:
        """Chat with streaming response tokens and internal model reasoning (thoughts)."""
        config = self.build_agent_config()
        accumulated_text: List[str] = []

        async with Agent(config) as agent:
            self._agent = agent
            response = await agent.chat(prompt)

            # 1. Stream sugared thoughts (reasoning process)
            if on_thought and hasattr(response, "thoughts"):
                async for thought in response.thoughts:
                    on_thought(thought)

            # 2. Stream response tokens
            async for token in response:
                accumulated_text.append(token)
                if on_token:
                    on_token(token)

        return "".join(accumulated_text)

    async def run_conversation_turn(self, prompt: str) -> Dict[str, Any]:
        """Execute a low-level turn using Conversation, accumulating step history."""
        if not self._conversation:
            await self.create_conversation()

        assert self._conversation is not None
        response = await self._conversation.chat(prompt)
        text = await response.text()

        return {
            "response": text,
            "turns": self._conversation.turn_count,
            "steps": len(self._conversation.history),
            "last_response": str(self._conversation.last_response)
        }
