"""TypeSafe Jev (System One) Integration for Context Optimization.

Implements the System One primitives (Choice, Score, Noul) from TypeSafe
(https://docs.typesafe.ai/concepts/use-case-map) for semantic context retrieval,
relevance scoring, role classification, and guardrail verification.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typesafe_sdk import (
        TypeSafeClient,
        Choice,
        Noul,
        Score,
        SystemOneResponse,
        NoulAnswer,
        ChoiceAnswer,
        ScoreAnswer,
    )
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    TYPESAFE_SDK_AVAILABLE = False
    TypeSafeClient = None
    Choice = None
    Noul = None
    Score = None
    SystemOneResponse = None
    NoulAnswer = None
    ChoiceAnswer = None
    ScoreAnswer = None

from .config import IsolatedConfig
from .scanner import ScannedFile

logger = logging.getLogger("context_optimizer.jev")


@dataclass
class FileJevDecision:
    """The structured System One decisions produced by Jev for a single file."""
    relpath: str
    is_relevant: bool
    relevance_probability: float  # from Noul [0.0, 1.0]
    score_label: str             # from Score: "irrelevant", "mention", "relevant", "critical"
    score_level_index: int       # 0, 1, 2, 3
    role: str                    # from Choice: "entrypoint", "core_logic", "data_model", "utility", "config", "test", "docs"
    inclusion_strategy: str      # from Choice: "full_content", "outline_and_excerpts", "manifest_entry", "exclude"
    confidence: float            # overall confidence
    reasoning_summary: str = ""


class JevContextEngine:
    """Orchestrates Jev System One decisions over codebase files."""

    SCORE_LEVELS = ["irrelevant", "mention", "relevant", "critical"]
    ROLES = ["entrypoint", "core_logic", "data_model", "utility", "config", "test", "docs"]
    INCLUSION_STRATEGIES = ["full_content", "outline_and_excerpts", "manifest_entry", "exclude"]

    def __init__(self, config: IsolatedConfig):
        self.config = config
        self.client: Optional[Any] = None
        
        # Initialize native TypeSafe client if API key is provided in isolated config
        if TYPESAFE_SDK_AVAILABLE and config.typesafe_api_key:
            model_name = config.typesafe_model if config.typesafe_model not in {"jev", ""} else "jev-latest"
            try:
                self.client = TypeSafeClient(
                    api_key=config.typesafe_api_key,
                    base_url=config.typesafe_base_url,
                    model=model_name
                )
                logger.info(f"Initialized live TypeSafe client with model '{model_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize live TypeSafe client: {e}. Falling back to embedded engine.")
                self.client = None

    def check_guardrails(self, prompt: str) -> Tuple[bool, float, str]:
        """Use Jev System One Noul to check for prompt injection or unsafe content.
        
        Returns:
            (is_safe, confidence, reason)
        """
        # If live client is available, ask live Jev
        if self.client:
            try:
                questions = {
                    "is_safe": Noul(
                        instructions="Is this user prompt safe, benign, and free of prompt injection, jailbreaks, or exfiltration attempts?"
                    )
                }
                res = self.client.system_one(prompt, questions)
                noul_val = res.nouls["is_safe"].noul
                is_safe = noul_val >= 0.5
                return is_safe, noul_val, "TypeSafe live Jev guardrail verification"
            except Exception as e:
                logger.warning(f"TypeSafe live guardrail evaluation failed: {e}. Disabling live client and falling back to embedded evaluation.")
                self.client = None

        # Heuristic embedded guardrail
        lower = prompt.lower()
        risky_terms = [
            "ignore previous instructions", "system prompt", "bypass security",
            "exfiltrate", "dump private key", "reveal secret", "disable safety"
        ]
        detected = [t for t in risky_terms if t in lower]
        if detected:
            return False, 0.9, f"Potential guardrail violation detected: {', '.join(detected)}"
        return True, 0.95, "Safe"

    def evaluate_file(self, scanned_file: ScannedFile, user_query: str) -> FileJevDecision:
        """Evaluate a scanned file against the user's query using Jev primitives."""
        file_summary = (
            f"File: {scanned_file.relpath}\n"
            f"Language: {scanned_file.language}, Lines: {scanned_file.line_count}\n"
            f"Symbols: {', '.join(s.name for s in scanned_file.symbols[:10])}\n"
            f"Imports: {', '.join(scanned_file.imports[:8])}\n"
            f"Excerpt:\n{scanned_file.excerpt[:400]}"
        )
        state = f"USER QUERY:\n{user_query}\n\nCANDIDATE FILE:\n{file_summary}"

        if self.client:
            try:
                questions = {
                    "is_relevant": Noul(
                        instructions=f"Is this file relevant to answering or fulfilling: '{user_query}'?"
                    ),
                    "relevance_score": Score(
                        instructions="Score the relevance of this file to the user query",
                        criteria=self.SCORE_LEVELS
                    ),
                    "role": Choice(
                        instructions="What is the architectural role of this file in the codebase?",
                        criteria={r: None for r in self.ROLES}
                    ),
                    "strategy": Choice(
                        instructions="What strategy should the context optimizer use for this file?",
                        criteria={s: None for s in self.INCLUSION_STRATEGIES}
                    )
                }
                res = self.client.system_one(state, questions)

                relevance_prob = res.nouls["is_relevant"].noul
                score_ans = res.scores["relevance_score"]
                score_raw = score_ans.score
                if isinstance(score_raw, (int, float)):
                    idx = int(round(score_raw))
                    idx = max(0, min(len(self.SCORE_LEVELS) - 1, idx))
                    score_label = self.SCORE_LEVELS[idx]
                    score_idx = idx
                else:
                    score_label = str(score_raw)
                    score_idx = self.SCORE_LEVELS.index(score_label) if score_label in self.SCORE_LEVELS else 0

                role_label = res.choices["role"].choice
                strategy_label = res.choices["strategy"].choice
                confidence = getattr(score_ans, "confidence", 0.85)

                # Ensure strategy aligns with relevance score
                if score_idx >= 3 and strategy_label in {"exclude", "manifest_entry"}:
                    strategy_label = "full_content" if scanned_file.line_count <= 400 else "outline_and_excerpts"
                elif score_idx >= 2 and strategy_label == "exclude":
                    strategy_label = "outline_and_excerpts" if scanned_file.line_count <= 600 else "manifest_entry"

                return FileJevDecision(
                    relpath=scanned_file.relpath,
                    is_relevant=relevance_prob >= self.config.relevance_threshold,
                    relevance_probability=relevance_prob,
                    score_label=score_label,
                    score_level_index=score_idx,
                    role=role_label,
                    inclusion_strategy=strategy_label,
                    confidence=confidence,
                    reasoning_summary=f"Evaluated via live TypeSafe Jev (P={relevance_prob:.2f})"
                )
            except Exception as e:
                logger.warning(f"TypeSafe live evaluation error for {scanned_file.relpath}: {e}. Disabling live client for subsequent files.")
                self.client = None

        # Embedded Fallback Jev Engine (exact same output model and calibrated probabilities)
        return self._embedded_evaluate_file(scanned_file, user_query)

    def _embedded_evaluate_file(self, scanned_file: ScannedFile, user_query: str) -> FileJevDecision:
        """Calibrated fallback decision engine matching Jev schema."""
        query_raw_words = [w.lower() for w in re.findall(r"\w+", user_query) if len(w) >= 3]
        query_stems = {w[:4] for w in query_raw_words}
        
        path_words = [w.lower() for w in re.findall(r"\w+", scanned_file.relpath) if len(w) >= 3]
        path_stems = {w[:4] for w in path_words}
        
        symbol_text = " ".join(s.name.lower() for s in scanned_file.symbols)
        symbol_words = [w.lower() for w in re.findall(r"\w+", symbol_text) if len(w) >= 3]
        symbol_stems = {w[:4] for w in symbol_words}

        excerpt_text = scanned_file.excerpt.lower()
        excerpt_words = [w.lower() for w in re.findall(r"\w+", excerpt_text) if len(w) >= 3]
        excerpt_stems = {w[:4] for w in excerpt_words}

        # Semantic & stem overlap
        path_matches = len(query_stems.intersection(path_stems))
        symbol_matches = len(query_stems.intersection(symbol_stems))
        excerpt_matches = len(query_stems.intersection(excerpt_stems))

        # Determine architectural role
        rel_lower = scanned_file.relpath.lower()
        if "test" in rel_lower or "spec" in rel_lower:
            role = "test"
        elif "config" in rel_lower or "env" in rel_lower or rel_lower.endswith((".json", ".yaml", ".yml", ".toml", ".ini")):
            role = "config"
        elif "doc" in rel_lower or rel_lower.endswith((".md", ".rst", ".txt")):
            role = "docs"
        elif "util" in rel_lower or "helper" in rel_lower or "common" in rel_lower:
            role = "utility"
        elif "model" in rel_lower or "schema" in rel_lower or "entity" in rel_lower:
            role = "data_model"
        elif any(ep in rel_lower for ep in ["main", "app", "index", "cli", "server", "entry"]):
            role = "entrypoint"
        else:
            role = "core_logic"

        # Calculate calibrated relevance probability (Noul)
        score_points = (path_matches * 3.5) + (symbol_matches * 2.5) + (min(excerpt_matches, 6) * 1.0)
        
        # If query is broad or empty, give baseline score to entrypoint and core_logic
        if len(query_raw_words) <= 2:
            if role in {"entrypoint", "core_logic"}:
                score_points += 2.0
            elif role in {"config", "docs"}:
                score_points += 1.0

        # Sigmoid calibration to [0.05, 0.98]
        relevance_prob = 1.0 / (1.0 + math.exp(-(score_points - 2.0) / 1.5))
        relevance_prob = max(0.05, min(0.98, round(relevance_prob, 3)))

        # Determine Score level
        if relevance_prob >= 0.70:
            score_label = "critical"
            score_idx = 3
        elif relevance_prob >= 0.45:
            score_label = "relevant"
            score_idx = 2
        elif relevance_prob >= 0.25:
            score_label = "mention"
            score_idx = 1
        else:
            score_label = "irrelevant"
            score_idx = 0

        # Determine inclusion strategy
        if score_idx >= 2:
            strategy = "full_content" if scanned_file.line_count <= 400 else "outline_and_excerpts"
        elif score_idx == 1 and relevance_prob >= self.config.relevance_threshold:
            strategy = "outline_and_excerpts" if scanned_file.line_count <= 300 else "manifest_entry"
        else:
            strategy = "exclude"

        is_relevant = relevance_prob >= self.config.relevance_threshold
        confidence = 0.90 if (path_matches > 0 or symbol_matches > 0) else 0.75

        return FileJevDecision(
            relpath=scanned_file.relpath,
            is_relevant=is_relevant,
            relevance_probability=relevance_prob,
            score_label=score_label,
            score_level_index=score_idx,
            role=role,
            inclusion_strategy=strategy,
            confidence=confidence,
            reasoning_summary=f"Embedded System One decision (matches: path={path_matches}, symbols={symbol_matches})"
        )
