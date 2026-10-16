"""Context Optimizer Core Engine.

Integrates directory crawling, Jev System One decision models, and token budget
compaction to generate distilled, high-density context packages for AI agents.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import IsolatedConfig
from .jev import FileJevDecision, JevContextEngine
from .scanner import DirectoryScanner, ScannedFile

logger = logging.getLogger("context_optimizer.optimizer")


@dataclass
class OptimizedFileBundle:
    """Formatted file entry included in the optimized context."""
    relpath: str
    decision: FileJevDecision
    inclusion_type: str  # "full", "outline", "excerpt", "manifest"
    token_cost: int
    rendered_text: str


@dataclass
class OptimizationResult:
    """Complete result of a context optimization pass."""
    query: str
    root_dir: str
    guardrail_passed: bool
    guardrail_confidence: float
    guardrail_message: str
    total_files_scanned: int
    total_files_included: int
    estimated_tokens: int
    max_tokens_budget: int
    execution_time_seconds: float
    selected_files: List[OptimizedFileBundle] = field(default_factory=list)
    manifest_files: List[str] = field(default_factory=list)
    context_markdown: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "query": self.query,
            "root_dir": self.root_dir,
            "guardrail": {
                "passed": self.guardrail_passed,
                "confidence": self.guardrail_confidence,
                "message": self.guardrail_message
            },
            "stats": {
                "total_scanned": self.total_files_scanned,
                "total_included": self.total_files_included,
                "estimated_tokens": self.estimated_tokens,
                "max_budget": self.max_tokens_budget,
                "duration_seconds": self.execution_time_seconds
            },
            "selected_files": [
                {
                    "path": b.relpath,
                    "role": b.decision.role,
                    "score": b.decision.score_label,
                    "probability": b.decision.relevance_probability,
                    "strategy": b.inclusion_type,
                    "tokens": b.token_cost
                }
                for b in self.selected_files
            ],
            "manifest_files": self.manifest_files,
            "context_markdown": self.context_markdown
        }

    def to_jev_event(self, event_id: Optional[str] = None, in_reply_to: Optional[str] = None) -> Dict[str, Any]:
        """Export as a typed JSON Event (JEV) matching the typesafe.ai event standard."""
        evt_id = event_id or f"jev-ctx-{int(time.time() * 1000)}"
        return {
            "specversion": "1.0",
            "id": evt_id,
            "in_reply_to": in_reply_to,
            "type": "ai.typesafe.context.optimized",
            "source": "context-optimizer-cli",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datacontenttype": "application/json",
            "data": self.to_dict()
        }


class ContextOptimizer:
    """Core Context Optimizer orchestrator."""

    def __init__(self, config: Optional[IsolatedConfig] = None, root_dir: Optional[str | Path] = None):
        self.config = config or IsolatedConfig.load()
        self.root_dir = Path(root_dir or Path.cwd()).resolve()
        self.scanner = DirectoryScanner(root_dir=self.root_dir, exclude_patterns=self.config.exclude_patterns)
        self.jev_engine = JevContextEngine(self.config)

    def optimize(
        self,
        query: str,
        token_budget: Optional[int] = None,
        force_include: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Execute the end-to-end context optimization pipeline."""
        start_time = time.time()
        budget = token_budget or self.config.max_context_tokens

        # Step 1: Jev Guardrail Verification
        safe, conf, msg = self.jev_engine.check_guardrails(query)
        if not safe:
            duration = round(time.time() - start_time, 3)
            return OptimizationResult(
                query=query,
                root_dir=str(self.root_dir),
                guardrail_passed=False,
                guardrail_confidence=conf,
                guardrail_message=msg,
                total_files_scanned=0,
                total_files_included=0,
                estimated_tokens=0,
                max_tokens_budget=budget,
                execution_time_seconds=duration,
                context_markdown=f"> [!WARNING] Guardrail rejection: {msg}"
            )

        # Step 2: Recursive File Scanning
        scanned_files = self.scanner.scan(load_content=True)

        # Step 3: Semantic Decision Pass with Jev (Noul, Score, Choice)
        decisions: List[Tuple[ScannedFile, FileJevDecision]] = []
        force_set = set(force_include or [])

        for f in scanned_files:
            dec = self.jev_engine.evaluate_file(f, query)
            # Force include override if user requested specific files
            if f.relpath in force_set:
                dec.is_relevant = True
                dec.score_label = "critical"
                dec.score_level_index = 3
                dec.inclusion_strategy = "full_content"
            decisions.append((f, dec))

        # Sort candidate files by relevance probability descending, then by score index
        decisions.sort(
            key=lambda item: (item[1].score_level_index, item[1].relevance_probability, -item[0].line_count),
            reverse=True
        )

        # Step 4: Token Budget Allocation & Content Compaction
        # Reserve ~20% of budget for headers, metadata, and directory manifest
        content_budget = max(200, int(budget * 0.8))
        current_tokens = 0
        selected_bundles: List[OptimizedFileBundle] = []
        manifest_files: List[str] = []

        for f, dec in decisions:
            if not dec.is_relevant and f.relpath not in force_set:
                manifest_files.append(f.relpath)
                continue

            strategy = dec.inclusion_strategy
            if dec.score_level_index >= 3 and strategy in {"manifest_entry", "exclude"}:
                strategy = "full_content" if f.line_count <= 400 else "outline_and_excerpts"
            elif dec.score_level_index >= 2 and strategy in {"manifest_entry", "exclude"}:
                strategy = "outline_and_excerpts"

            # Full content inclusion for critical or compact relevant files
            if strategy == "full_content" and f.content:
                cost = f.estimated_tokens
                if current_tokens + cost <= content_budget:
                    rendered = f"### File: `{f.relpath}` ({dec.role}, {f.line_count} lines)\n```{f.language}\n{f.content.strip()}\n```\n"
                    selected_bundles.append(OptimizedFileBundle(
                        relpath=f.relpath,
                        decision=dec,
                        inclusion_type="full",
                        token_cost=cost,
                        rendered_text=rendered
                    ))
                    current_tokens += cost
                    continue
                else:
                    # Downgrade to outline if full content exceeds remaining budget
                    strategy = "outline_and_excerpts"

            # Outline and excerpts for medium-sized or relevant files
            if strategy in {"full_content", "outline_and_excerpts"}:
                outline_text = f.get_summary_outline()
                excerpt_text = f.excerpt.strip()
                rendered = (
                    f"### File: `{f.relpath}` [Outline & Excerpt] ({dec.role})\n"
                    f"**Structure**:\n```\n{outline_text}\n```\n"
                    f"**Key Excerpt**:\n```{f.language}\n{excerpt_text}\n```\n"
                )
                cost = max(30, len(rendered) // 4)
                if current_tokens + cost <= content_budget:
                    selected_bundles.append(OptimizedFileBundle(
                        relpath=f.relpath,
                        decision=dec,
                        inclusion_type="outline",
                        token_cost=cost,
                        rendered_text=rendered
                    ))
                    current_tokens += cost
                    continue

            # Fallback to manifest entry if over budget or lower relevance
            manifest_files.append(f.relpath)

        # Step 5: Synthesize High-Density Context Markdown Bundle
        md_parts: List[str] = []
        md_parts.append(f"# Optimized Context for: `{query}`\n")
        md_parts.append(
            f"> **Root**: `{self.root_dir}` | **Scanned**: {len(scanned_files)} files | "
            f"**Included**: {len(selected_bundles)} files | **Estimated Tokens**: ~{current_tokens + 200}/{budget}\n"
        )

        # Architecture & Role Summary
        role_counts: Dict[str, int] = {}
        for b in selected_bundles:
            role_counts[b.decision.role] = role_counts.get(b.decision.role, 0) + 1
        roles_str = ", ".join(f"{k}: {v}" for k, v in sorted(role_counts.items()))
        md_parts.append(f"**Included Roles**: {roles_str}\n")

        # Key Files Details
        md_parts.append("## High-Priority Context & Source Slices\n")
        for b in selected_bundles:
            md_parts.append(b.rendered_text)

        # Directory Manifest
        if manifest_files:
            md_parts.append("## Repository Manifest (Other Scanned Files)\n")
            md_parts.append("<details><summary>Click to view remaining files</summary>\n\n")
            for mf in sorted(manifest_files)[:100]:
                md_parts.append(f"- `{mf}`\n")
            if len(manifest_files) > 100:
                md_parts.append(f"- ... ({len(manifest_files) - 100} more files)\n")
            md_parts.append("\n</details>\n")

        context_markdown = "\n".join(md_parts)
        total_estimated = max(1, len(context_markdown) // 4)
        duration = round(time.time() - start_time, 3)

        return OptimizationResult(
            query=query,
            root_dir=str(self.root_dir),
            guardrail_passed=True,
            guardrail_confidence=conf,
            guardrail_message="Pass",
            total_files_scanned=len(scanned_files),
            total_files_included=len(selected_bundles),
            estimated_tokens=total_estimated,
            max_tokens_budget=budget,
            execution_time_seconds=duration,
            selected_files=selected_bundles,
            manifest_files=manifest_files,
            context_markdown=context_markdown
        )
