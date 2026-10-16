"""Inter-Agent Trigger Integration and JEV Event Protocol.

Allows other agents to trigger context optimization via:
1. Programmatic Python API
2. CLI JSON / JEV event arguments or standard input
3. File inbox trigger watcher (.context_optimizer/inbox)
4. Antigravity SDK native triggers (on_file_change, every, TriggerContext)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from pydantic import BaseModel, Field

from google.antigravity.triggers import TriggerContext, every, on_file_change

from .config import IsolatedConfig
from .optimizer import ContextOptimizer, OptimizationResult

logger = logging.getLogger("context_optimizer.triggers")


class JevTriggerPayload(BaseModel):
    """Payload inside a context optimization request event."""
    query: str = Field(..., description="Target search query or instruction for context optimization")
    root_dir: Optional[str] = Field(default=None, description="Root directory to scan (defaults to cwd)")
    token_budget: Optional[int] = Field(default=None, description="Max token budget for the context bundle")
    force_include: Optional[list[str]] = Field(default=None, description="Specific relative file paths to force include")


class JevRequestEvent(BaseModel):
    """TypeSafe CloudEvents / JEV-compliant trigger request."""
    specversion: str = Field(default="1.0")
    id: str = Field(default_factory=lambda: f"req-{int(time.time() * 1000)}")
    type: str = Field(default="ai.typesafe.context.optimize.request")
    source: str = Field(default="agent")
    time: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    datacontenttype: str = Field(default="application/json")
    data: JevTriggerPayload


def execute_trigger_request(
    request_data: Union[str, Dict[str, Any], JevRequestEvent],
    config: Optional[IsolatedConfig] = None
) -> Dict[str, Any]:
    """Execute a context optimization request from an incoming JEV trigger event."""
    cfg = config or IsolatedConfig.load()

    if isinstance(request_data, str):
        try:
            parsed = json.loads(request_data)
        except json.JSONDecodeError as e:
            # If not JSON, treat raw string as the query directly
            parsed = {"query": request_data}
    elif isinstance(request_data, JevRequestEvent):
        parsed = request_data.model_dump()
    else:
        parsed = request_data

    # Extract event metadata if wrapped in a CloudEvent / JEV envelope
    event_id = parsed.get("id", f"req-{int(time.time() * 1000)}")
    payload_dict = parsed.get("data", parsed)
    if "data" not in parsed and "query" in parsed:
        payload_dict = parsed

    payload = JevTriggerPayload(**payload_dict)

    optimizer = ContextOptimizer(config=cfg, root_dir=payload.root_dir)
    result = optimizer.optimize(
        query=payload.query,
        token_budget=payload.token_budget,
        force_include=payload.force_include
    )

    return result.to_jev_event(in_reply_to=event_id)


class TriggerInboxWatcher:
    """Watches a local filesystem inbox where other agents drop trigger files."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        config: Optional[IsolatedConfig] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.inbox_dir = self.base_dir / ".context_optimizer" / "inbox"
        self.outbox_dir = self.base_dir / ".context_optimizer" / "outbox"
        self.config = config or IsolatedConfig.load()
        self.on_result = on_result

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

    def process_pending_files(self) -> int:
        """Scan inbox directory for pending request files and process them."""
        processed_count = 0
        for req_file in sorted(self.inbox_dir.glob("*.json")):
            try:
                content = req_file.read_text(encoding="utf-8")
                response_event = execute_trigger_request(content, config=self.config)

                # Write response to outbox
                resp_id = response_event.get("id", f"resp-{int(time.time() * 1000)}")
                out_file = self.outbox_dir / f"{resp_id}.json"
                out_file.write_text(json.dumps(response_event, indent=2), encoding="utf-8")

                if self.on_result:
                    self.on_result(response_event)

                # Remove processed request
                req_file.unlink(missing_ok=True)
                processed_count += 1
                logger.info(f"Processed agent trigger file {req_file.name} -> {out_file.name}")
            except Exception as e:
                logger.error(f"Failed to process trigger file {req_file}: {e}")

        return processed_count

    def create_antigravity_trigger(self) -> Any:
        """Create an Antigravity SDK on_file_change trigger watching the inbox directory."""
        watcher = self

        async def handle_changes(ctx: TriggerContext, changes):
            for change in changes:
                logger.info(f"Antigravity Trigger detected inbox change: {change.path} ({change.kind})")
            processed = watcher.process_pending_files()
            if processed > 0:
                await ctx.send(f"Context Optimizer processed {processed} agent trigger requests.")

        return on_file_change(str(self.inbox_dir), handle_changes)

    def create_periodic_trigger(self, interval_seconds: int = 10) -> Any:
        """Create an Antigravity SDK periodic trigger checking the inbox."""
        watcher = self

        async def check_inbox(ctx: TriggerContext):
            processed = watcher.process_pending_files()
            if processed > 0:
                await ctx.send(f"Periodic check: processed {processed} context optimization requests.")

        return every(interval_seconds, check_inbox)
