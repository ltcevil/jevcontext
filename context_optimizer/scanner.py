"""Directory Scanner and AST/Symbol Extractor.

Iterates over all files within the current working directory and subdirectories,
extracts structural outlines, symbol trees, and file contents while honoring
ignore rules and binary safety checks.
"""

from __future__ import annotations

import ast
import fnmatch
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# Common text file extensions
KNOWN_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".txt",
    ".sh",
    ".bash",
    ".zsh",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".xml",
    ".svg",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
}

# Common binary extensions
KNOWN_BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".a",
    ".o",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".mp3",
    ".mp4",
    ".wav",
    ".bin",
    ".exe",
    ".dat",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".wasm",
}


@dataclass
class SymbolInfo:
    """Represents a code symbol (class, function, method, interface)."""

    name: str
    kind: str  # "class", "function", "method", "interface", "export"
    line_number: int
    signature: str = ""
    docstring: Optional[str] = None


@dataclass
class ScannedFile:
    """Represents a scanned source or documentation file."""

    path: Path
    relpath: str
    size_bytes: int
    line_count: int
    char_count: int
    estimated_tokens: int
    language: str
    is_code: bool
    symbols: List[SymbolInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    excerpt: str = ""
    content: Optional[str] = None

    def get_summary_outline(self) -> str:
        """Return a compact outline representation of the file."""
        lines = [
            f"{self.relpath} ({self.line_count} lines, ~{self.estimated_tokens} tokens)"
        ]
        if self.symbols:
            for sym in self.symbols[:15]:  # Limit top 15 symbols to save tokens
                lines.append(f"  - {sym.kind} {sym.signature or sym.name}")
            if len(self.symbols) > 15:
                lines.append(f"  - ... ({len(self.symbols) - 15} more symbols)")
        return "\n".join(lines)


SENSITIVE_NAME_GLOBS = (
    ".env",
    ".env.*",
    "*.env",
    ".optimizer.env",
    ".env.context_optimizer",
    ".credentials",
    ".credentials.json",
    ".ssh",
    ".aws",
    ".gnupg",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*.token",
    "*.session",
    "*.session-journal",
    "secret.txt",
    "hide.txt",
    "secrets.*",
    "*.kdbx",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
)


class DirectoryScanner:
    """Crawler that iterates over all files in cwd or specified root."""

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size_bytes: int = 500_000,  # 500 KB limit for full read
    ):
        self.root_dir = Path(root_dir or os.getcwd()).resolve()
        self.exclude_patterns = exclude_patterns or [
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".DS_Store",
            "*.pyc",
            "*.lock",
            "package-lock.json",
            "dist",
            "build",
            "*.egg-info",
            ".idea",
            ".vscode",
            "*.min.js",
            "*.min.css",
            "*.map",
            ".context_optimizer",
            ".history",
            "docs",
        ]
        self.max_file_size_bytes = max_file_size_bytes
        # Inside a git work tree git itself decides what is ignored: it understands
        # negation ("!userbot/**"), nested .gitignore files and info/exclude, which the
        # fnmatch fallback below does not (a whitelist-style ".gitignore" starting with
        # "*" would otherwise exclude every file).
        self.git_files, self.git_dirs = self._git_visible_paths()
        self.gitignore_patterns = (
            [] if self.git_files is not None else self._load_gitignore()
        )

    def _git_visible_paths(self):
        """Tracked + untracked-not-ignored files under root_dir (relative), or (None, None)."""
        import subprocess

        try:
            out = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.root_dir),
                    "-c",
                    "core.quotepath=off",
                    "ls-files",
                    "-z",
                    "-co",
                    "--exclude-standard",
                ],
                capture_output=True,
                timeout=30,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None, None
        files: Set[str] = set()
        dirs: Set[str] = set()
        for rel in out.decode("utf-8", "surrogateescape").split("\0"):
            if not rel:
                continue
            files.add(rel)
            parent = os.path.dirname(rel)
            while parent and parent not in dirs:
                dirs.add(parent)
                parent = os.path.dirname(parent)
        return files, dirs

    def _load_gitignore(self) -> List[str]:
        """Load patterns from root .gitignore if present."""
        gitignore_path = self.root_dir / ".gitignore"
        patterns: List[str] = []
        if gitignore_path.is_file():
            try:
                for line in gitignore_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except Exception:
                pass
        return patterns

    def should_exclude(self, rel_path: str, is_dir: bool = False) -> bool:
        """Check if a path matches any exclude or gitignore pattern."""
        norm_path = rel_path.replace(os.sep, "/")
        parts = Path(norm_path).parts
        # Credential-looking names are never read, whatever git says: their content would
        # otherwise be sent on to TypeSafe and Gemini.
        if any(
            fnmatch.fnmatch(part.lower(), g)
            for part in parts
            for g in SENSITIVE_NAME_GLOBS
        ):
            return True
        for part in parts:
            if part in {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".context_optimizer",
                ".history",
                ".DS_Store",
            }:
                return True

        # Check explicit exclude patterns
        has_negation = any(pat.startswith("!") for pat in self.exclude_patterns)
        has_matching_root = has_negation and any(
            (self.root_dir / pat[1:].lstrip("./").split("/")[0]).exists()
            for pat in self.exclude_patterns
            if pat.startswith("!")
        )
        if has_negation and has_matching_root:
            excluded_by_pattern = False
            for pat in self.exclude_patterns:
                negated = pat.startswith("!")
                actual_pat = pat[1:] if negated else pat
                clean_pat = actual_pat.lstrip("./").rstrip("/")

                matched = False
                if (
                    fnmatch.fnmatch(norm_path, clean_pat)
                    or fnmatch.fnmatch(norm_path, f"{clean_pat}/*")
                    or fnmatch.fnmatch(norm_path, f"{clean_pat}/**")
                ):
                    matched = True
                elif fnmatch.fnmatch(os.path.basename(norm_path), clean_pat):
                    matched = True
                else:
                    for part in parts:
                        if fnmatch.fnmatch(part, clean_pat):
                            matched = True
                            break

                if matched:
                    excluded_by_pattern = not negated

            if excluded_by_pattern:
                return True
        else:
            for pat in self.exclude_patterns:
                if pat.startswith("!"):
                    continue
                clean_pat = pat.lstrip("./").rstrip("/")
                if clean_pat == "*":
                    continue
                if fnmatch.fnmatch(norm_path, clean_pat) or fnmatch.fnmatch(
                    os.path.basename(norm_path), clean_pat
                ):
                    return True
                for part in parts:
                    if fnmatch.fnmatch(part, clean_pat):
                        return True

        if self.git_files is not None:
            return rel_path not in (self.git_dirs if is_dir else self.git_files)

        # Check gitignore patterns (fallback outside git) with negation support
        ignored = False
        for pat in self.gitignore_patterns:
            negated = pat.startswith("!")
            actual_pat = pat[1:] if negated else pat
            clean_pat = actual_pat.rstrip("/")
            if clean_pat.startswith("./"):
                clean_pat = clean_pat[2:]

            matched = False
            if fnmatch.fnmatch(norm_path, clean_pat) or fnmatch.fnmatch(
                os.path.basename(norm_path), clean_pat
            ):
                matched = True
            else:
                for part in parts:
                    if fnmatch.fnmatch(part, clean_pat):
                        matched = True
                        break

            if matched:
                ignored = not negated

        return ignored

    def is_binary_file(self, file_path: Path) -> bool:
        """Heuristic check to determine if a file is binary."""
        ext = file_path.suffix.lower()
        if ext in KNOWN_BINARY_EXTENSIONS:
            return True
        if ext in KNOWN_TEXT_EXTENSIONS:
            return False

        mime, _ = mimetypes.guess_type(str(file_path))
        if (
            mime
            and not mime.startswith("text/")
            and mime not in {"application/json", "application/javascript"}
        ):
            return True

        # Check first 1024 bytes for null byte
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return True
        except Exception:
            return True

        return False

    def scan(self, load_content: bool = True) -> List[ScannedFile]:
        """Recursively scan files and extract symbols and metadata."""
        scanned_files: List[ScannedFile] = []

        for root, dirs, files in os.walk(self.root_dir):
            rel_root = os.path.relpath(root, self.root_dir)
            if rel_root == ".":
                rel_root = ""

            # Prune excluded directories in-place to avoid descending into them
            dirs[:] = [
                d
                for d in dirs
                if not self.should_exclude(
                    os.path.join(rel_root, d) if rel_root else d, is_dir=True
                )
            ]

            for file_name in sorted(files):
                rel_path = os.path.join(rel_root, file_name) if rel_root else file_name
                if self.should_exclude(rel_path):
                    continue

                full_path = Path(root) / file_name
                if not full_path.is_file():
                    continue

                if self.is_binary_file(full_path):
                    continue

                try:
                    stat = full_path.stat()
                    size = stat.st_size
                except OSError:
                    continue

                # Read text safely
                content: Optional[str] = None
                line_count = 0
                char_count = 0
                symbols: List[SymbolInfo] = []
                imports: List[str] = []

                try:
                    raw_text = full_path.read_text(encoding="utf-8", errors="replace")
                    char_count = len(raw_text)
                    line_count = len(raw_text.splitlines())

                    if load_content and size <= self.max_file_size_bytes:
                        content = raw_text

                    ext = full_path.suffix.lower()
                    is_code = ext in {
                        ".py",
                        ".ts",
                        ".js",
                        ".tsx",
                        ".jsx",
                        ".go",
                        ".rs",
                        ".java",
                        ".c",
                        ".cpp",
                    }
                    language = ext.lstrip(".") or "txt"

                    # Extract symbols and imports based on language
                    if ext == ".py":
                        symbols, imports = self._extract_python_symbols(raw_text)
                    elif ext in {".ts", ".js", ".tsx", ".jsx"}:
                        symbols, imports = self._extract_js_ts_symbols(raw_text)

                    # Extract short excerpt (first 10 non-empty lines)
                    lines = [
                        line.strip() for line in raw_text.splitlines() if line.strip()
                    ]
                    excerpt = "\n".join(lines[:10])

                    estimated_tokens = max(1, char_count // 4)

                    scanned_files.append(
                        ScannedFile(
                            path=full_path,
                            relpath=rel_path,
                            size_bytes=size,
                            line_count=line_count,
                            char_count=char_count,
                            estimated_tokens=estimated_tokens,
                            language=language,
                            is_code=is_code,
                            symbols=symbols,
                            imports=imports,
                            excerpt=excerpt,
                            content=content,
                        )
                    )

                except Exception:
                    continue

        return scanned_files

    def _extract_python_symbols(self, code: str) -> tuple[List[SymbolInfo], List[str]]:
        """Extract classes, functions, and imports using Python AST."""
        symbols: List[SymbolInfo] = []
        imports: List[str] = []

        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for alias in node.names:
                        imports.append(f"{mod}.{alias.name}")
                elif isinstance(node, ast.ClassDef):
                    bases = (
                        [ast.unparse(b) for b in node.bases]
                        if hasattr(ast, "unparse")
                        else []
                    )
                    base_str = f"({', '.join(bases)})" if bases else ""
                    doc = ast.get_docstring(node)
                    symbols.append(
                        SymbolInfo(
                            name=node.name,
                            kind="class",
                            line_number=node.lineno,
                            signature=f"class {node.name}{base_str}:",
                            docstring=doc.split("\n")[0] if doc else None,
                        )
                    )
                    # Methods inside class
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            prefix = (
                                "async def"
                                if isinstance(sub, ast.AsyncFunctionDef)
                                else "def"
                            )
                            symbols.append(
                                SymbolInfo(
                                    name=f"{node.name}.{sub.name}",
                                    kind="method",
                                    line_number=sub.lineno,
                                    signature=f"  {prefix} {sub.name}(...)",
                                )
                            )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = (
                        "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    )
                    doc = ast.get_docstring(node)
                    symbols.append(
                        SymbolInfo(
                            name=node.name,
                            kind="function",
                            line_number=node.lineno,
                            signature=f"{prefix} {node.name}(...)",
                            docstring=doc.split("\n")[0] if doc else None,
                        )
                    )
        except SyntaxError:
            # Fallback to regex if file has Python syntax error
            for i, line in enumerate(code.splitlines(), start=1):
                match = re.match(r"^\s*(class|def|async def)\s+([a-zA-Z0-9_]+)", line)
                if match:
                    k, name = match.group(1), match.group(2)
                    kind = "class" if k == "class" else "function"
                    symbols.append(
                        SymbolInfo(
                            name=name, kind=kind, line_number=i, signature=line.strip()
                        )
                    )

        return symbols, imports

    def _extract_js_ts_symbols(self, code: str) -> tuple[List[SymbolInfo], List[str]]:
        """Extract JavaScript and TypeScript symbols and imports using regex."""
        symbols: List[SymbolInfo] = []
        imports: List[str] = []

        import_pattern = re.compile(
            r"^\s*import\s+.*?\s+from\s+['\"](.*?)['\"]", re.MULTILINE
        )
        for m in import_pattern.finditer(code):
            imports.append(m.group(1))

        # Functions, classes, interfaces
        sym_pattern = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(function|class|interface|type|const|let|var)\s+([a-zA-Z0-9_]+)",
            re.MULTILINE,
        )
        for m in sym_pattern.finditer(code):
            kind, name = m.group(1), m.group(2)
            line_no = code[: m.start()].count("\n") + 1
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind=kind,
                    line_number=line_no,
                    signature=f"{kind} {name}",
                )
            )

        return symbols, imports
