"""Command Line Interface for Context Optimizer.

Provides subcommands for interactive context distillation, machine-readable JEV event
triggers, inbox listener daemon, MCP tool export, and isolated config management.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import IsolatedConfig, DEFAULT_ENV_FILE
from .mcp_tool import MCP_TOOL_DEFINITION
from .optimizer import ContextOptimizer
from .scanner import DirectoryScanner
from .agent import AntigravityAgentManager
from .triggers import TriggerInboxWatcher, execute_trigger_request

console = Console(stderr=True)  # Logs and status go to stderr
stdout_console = Console()  # Payload output goes to stdout


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ["httpx", "httpcore", "httpx2", "typesafe_sdk", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_optimize(args: argparse.Namespace):
    """Execute context optimization for a user query."""
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    if not query.strip():
        console.print("[bold red]Error:[/] Query cannot be empty.")
        sys.exit(1)

    root_dir = Path(args.root or os.getcwd()).resolve()
    overrides = {}
    if getattr(args, "vertex", False):
        overrides["use_vertex"] = True
    if getattr(args, "project", None):
        overrides["gcp_project"] = args.project
    if getattr(args, "location", None):
        overrides["gcp_location"] = args.location
    if getattr(args, "model", None):
        overrides["gemini_model"] = args.model
    if getattr(args, "exclude_patterns", None):
        patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]
        if patterns:
            overrides["exclude_patterns"] = patterns
    if getattr(args, "min_score", None) is not None:
        overrides["relevance_threshold"] = args.min_score

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None,
    )

    auth_mode = (
        f"Vertex AI (ADC, project: {config.gcp_project or 'auto'})"
        if config.use_vertex
        else "Gemini API Key"
    )
    console.print(f"[bold cyan]🔍 Scanning directory:[/] {root_dir}")
    console.print(f"[bold cyan]🎯 Optimization query:[/] {query}")
    console.print(
        f"[dim]Auth: {auth_mode} | Env isolation: {'enabled' if config.isolate_env else 'disabled'} | Budget: {args.budget or config.max_context_tokens} tokens[/dim]"
    )

    optimizer = ContextOptimizer(config=config, root_dir=root_dir)
    res = optimizer.optimize(
        query=query, token_budget=args.budget, force_include=args.force_include
    )

    if not res.guardrail_passed:
        console.print(
            f"[bold red]❌ Guardrail check rejected query:[/] {res.guardrail_message}"
        )
        sys.exit(2)

    # Display Rich summary table to stderr
    table = Table(title="Jev System One Decisions & Optimization Summary")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    table.add_column("Jev Score", style="yellow")
    table.add_column("P(Relevant)", style="green")
    table.add_column("Strategy", style="blue")
    table.add_column("Tokens", style="white", justify="right")

    for bundle in res.selected_files:
        table.add_row(
            bundle.relpath,
            bundle.decision.role,
            bundle.decision.score_label,
            f"{bundle.decision.relevance_probability:.2f}",
            bundle.inclusion_type,
            str(bundle.token_cost),
        )

    console.print(table)
    console.print(
        f"[bold green]✔ Optimization complete:[/] Included {res.total_files_included}/{res.total_files_scanned} files "
        f"(~{res.estimated_tokens} tokens) in {res.execution_time_seconds}s"
    )

    # Format output
    output_text = ""
    if args.format == "jev":
        output_text = json.dumps(res.to_jev_event(), indent=2)
    elif args.format == "json":
        output_text = json.dumps(res.to_dict(), indent=2)
    else:
        output_text = res.context_markdown

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Saved output to:[/] {out_path}")
    elif args.format in ("json", "jev") or getattr(args, "context_only", False):
        try:
            sys.stdout.write(output_text)
            if not output_text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        except BrokenPipeError:
            pass
    else:
        # Default interactive behavior: stream the Gemini answer over the distilled context
        console.print(f"\n[bold green]Answer ({config.gemini_model}):[/]\n")
        context_prompt = f"{res.context_markdown}\n\n---\nUSER TASK:\n{query}"
        agent_mgr = AntigravityAgentManager(config=config, workspace_dir=root_dir)

        def on_token(token: str):
            sys.stdout.write(token)
            sys.stdout.flush()

        import asyncio

        try:
            asyncio.run(agent_mgr.chat_stream(context_prompt, on_token=on_token))
            sys.stdout.write("\n")
        except Exception as e:
            console.print(f"\n[bold red]Gemini error:[/] {e}")


def cmd_trigger(args: argparse.Namespace):
    """Trigger context optimization via a JEV event (from another agent)."""
    raw_payload = ""
    if args.stdin or not sys.stdin.isatty():
        raw_payload = sys.stdin.read()
    elif args.payload:
        raw_payload = args.payload
    elif args.query:
        raw_payload = json.dumps({"query": " ".join(args.query)})
    else:
        console.print(
            "[bold red]Error:[/] Must provide --payload, --query, or send via stdin."
        )
        sys.exit(1)

    overrides = {}
    if getattr(args, "vertex", False):
        overrides["use_vertex"] = True
    if getattr(args, "project", None):
        overrides["gcp_project"] = args.project
    if getattr(args, "location", None):
        overrides["gcp_location"] = args.location
    if getattr(args, "model", None):
        overrides["gemini_model"] = args.model
    if getattr(args, "exclude_patterns", None):
        patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]
        if patterns:
            overrides["exclude_patterns"] = patterns
    if getattr(args, "min_score", None) is not None:
        overrides["relevance_threshold"] = args.min_score

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None,
    )

    try:
        response_event = execute_trigger_request(raw_payload, config=config)
        output_json = json.dumps(response_event, indent=2)
        sys.stdout.write(output_json + "\n")
        sys.stdout.flush()
    except Exception as e:
        console.print(f"[bold red]Trigger error:[/] {e}")
        sys.exit(1)


def cmd_listen(args: argparse.Namespace):
    """Run inbox watcher daemon for inter-agent triggers."""
    base_dir = Path(args.dir or os.getcwd()).resolve()
    config = IsolatedConfig.load(
        env_file_path=args.env_file, allow_global_env=args.allow_global_env
    )

    watcher = TriggerInboxWatcher(
        base_dir=base_dir,
        config=config,
        on_result=lambda res: console.print(
            f"[green]✔ Processed trigger:[/] {res.get('id')} ({res.get('type')})"
        ),
    )

    console.print(
        Panel(
            f"[bold green]Context Optimizer Agent Listener active[/]\n"
            f"Watching inbox: [cyan]{watcher.inbox_dir}[/]\n"
            f"Writing outbox: [cyan]{watcher.outbox_dir}[/]\n"
            f"Interval: {args.interval}s | Press Ctrl+C to stop",
            title="Agent Trigger Listener",
        )
    )

    try:
        import time

        while True:
            watcher.process_pending_files()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Listener stopped.[/]")


def cmd_config(args: argparse.Namespace):
    """Inspect or manage the isolated configuration store."""
    if args.config_action == "path":
        print(DEFAULT_ENV_FILE)
        return

    if args.config_action == "set":
        if not args.key_value or "=" not in args.key_value:
            console.print(
                "[bold red]Error:[/] Format must be KEY=VALUE (e.g. GEMINI_API_KEY=xyz)"
            )
            sys.exit(1)
        k, v = args.key_value.split("=", 1)
        target = Path(args.env_file).resolve() if args.env_file else DEFAULT_ENV_FILE
        saved_path = IsolatedConfig.save_variable(
            k.strip(), v.strip(), target_path=target
        )
        console.print(f"[green]Saved {k.strip()} to isolated store:[/] {saved_path}")
        return

    # Default: show
    cfg = IsolatedConfig.load(env_file_path=args.env_file)
    table = Table(title="Isolated Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Source", style="dim")

    for k, v in cfg.model_dump().items():
        val_str = str(v)
        if "key" in k.lower() and v:
            val_str = v[:4] + "..." + v[-4:] if len(v) > 8 else "***"
        table.add_row(k, val_str, "isolated_store")

    console.print(table)


def cmd_mcp(args: argparse.Namespace):
    """Output the MCP tool schema for agent discovery."""
    print(json.dumps(MCP_TOOL_DEFINITION, indent=2))


def cmd_scan(args: argparse.Namespace):
    """Scan and audit repository files, symbols, token estimates, and exclusions."""
    root_dir = Path(args.root or os.getcwd()).resolve()
    overrides = {}
    if getattr(args, "exclude_patterns", None):
        patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]
        if patterns:
            overrides["exclude_patterns"] = patterns

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None,
    )

    scanner = DirectoryScanner(
        root_dir=root_dir, exclude_patterns=config.exclude_patterns
    )
    scanned = scanner.scan(load_content=args.symbols)

    if args.ext:
        ext = args.ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        scanned = [f for f in scanned if f.path.suffix.lower() == ext]

    if getattr(args, "json", False):
        data = {
            "root_dir": str(root_dir),
            "total_files": len(scanned),
            "total_tokens": sum(f.estimated_tokens for f in scanned),
            "total_lines": sum(f.line_count for f in scanned),
            "files": [
                {
                    "path": f.relpath,
                    "lines": f.line_count,
                    "tokens": f.estimated_tokens,
                    "language": f.language,
                    "symbols": [s.name for s in f.symbols],
                }
                for f in scanned
            ],
        }
        try:
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        return

    table = Table(title=f"Codebase Scan: {root_dir}")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Lang", style="magenta")
    table.add_column("Lines", style="yellow", justify="right")
    table.add_column("Est. Tokens", style="green", justify="right")
    table.add_column("Symbols", style="blue", justify="right")

    limit = args.limit or 50
    for f in scanned[:limit]:
        table.add_row(
            f.relpath,
            f.language,
            str(f.line_count),
            str(f.estimated_tokens),
            str(len(f.symbols)),
        )

    console.print(table)
    if len(scanned) > limit:
        console.print(
            f"[dim]... and {len(scanned) - limit} more files (use --limit to show more)[/dim]"
        )

    total_tokens = sum(f.estimated_tokens for f in scanned)
    total_lines = sum(f.line_count for f in scanned)
    console.print(
        f"[bold green]Summary:[/] {len(scanned)} files | {total_lines} lines | ~{total_tokens} estimated tokens"
    )


def cmd_agent(args: argparse.Namespace):
    """Run an Antigravity AI Agent with distilled codebase context."""
    prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    if not prompt.strip():
        console.print("[bold red]Error:[/] Agent prompt cannot be empty.")
        sys.exit(1)

    root_dir = Path(args.root or os.getcwd()).resolve()
    overrides = {}
    if getattr(args, "vertex", False):
        overrides["use_vertex"] = True
    if getattr(args, "project", None):
        overrides["gcp_project"] = args.project
    if getattr(args, "location", None):
        overrides["gcp_location"] = args.location
    if getattr(args, "model", None):
        overrides["gemini_model"] = args.model
    if getattr(args, "exclude_patterns", None):
        patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]
        if patterns:
            overrides["exclude_patterns"] = patterns

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None,
    )

    console.print("[bold cyan]🤖 Distilling context for agent...[/]")
    optimizer = ContextOptimizer(config=config, root_dir=root_dir)
    res = optimizer.optimize(query=prompt, token_budget=args.budget or 4000)

    context_prompt = f"{res.context_markdown}\n\n---\nUSER TASK:\n{prompt}"

    agent_mgr = AntigravityAgentManager(config=config, workspace_dir=root_dir)
    console.print(f"[bold green]🚀 Agent ({config.gemini_model}) executing...[/]\n")

    def on_token(token: str):
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_thought(thought: str):
        if getattr(args, "thoughts", False):
            console.print(f"[dim italic]💭 {thought}[/dim italic]")

    import asyncio

    try:
        asyncio.run(
            agent_mgr.chat_stream(
                context_prompt,
                on_token=on_token,
                on_thought=on_thought,
            )
        )
        sys.stdout.write("\n")
    except Exception as e:
        console.print(f"\n[bold red]Agent error:[/] {e}")
        sys.exit(1)


CLI_MAIN_EPILOG = """\
examples (copy, paste, adjust):

  1. OPTIMIZE — distill high-density context for questions or coding tasks:
    context-optimizer optimize "how does callback router dispatch work?"
    context-optimizer optimize "why does the ingest fail?" --root /opt/jarina --budget 12000
    context-optimizer optimize "audit secrets handling" --format json -o context.json
    context-optimizer optimize "fix Pinecone batch timeout" --force-include userbot/plugins/pine.py
    context-optimizer optimize "summarize architecture" --vertex --project jarinagem --model gemini-3.8-flash
    context-optimizer optimize "refactor scanner" --exclude-patterns "*,!tests,!scripts,!userbot" --min-score 0.7

  2. TRIGGER — trigger context optimization programmatically via JEV event or JSON:
    context-optimizer trigger "find memory leak in worker"
    context-optimizer trigger --payload '{"query": "diagnose 500 error", "token_budget": 6000}'
    echo '{"query": "trace call path", "force_include": ["main.py"]}' | context-optimizer trigger --stdin

  3. AGENT — interactive Antigravity agent answering over distilled context:
    context-optimizer agent "explain the authentication flow and propose unit tests"
    context-optimizer agent "refactor the database query layer" --root /opt/jarina --stream
    context-optimizer agent "review changes in tests" --exclude-patterns "*,!tests" --thoughts

  4. SCAN — inspect repository files, AST symbols, token estimates & exclusions:
    context-optimizer scan
    context-optimizer scan --root /opt/jarina --symbols
    context-optimizer scan --root /opt/jarina --exclude-patterns "*,!tests,!scripts,!userbot"
    context-optimizer scan --ext .py --top-k 20 --json

  5. MCP & MCP-SERVER — inspect schema or run stdio server for Claude & Agent IDEs:
    context-optimizer mcp                                           # Output JSON schema for MCP tools
    context-optimizer mcp-server                                    # Run stdio MCP server for agent IDEs
    context-optimizer mcp-server --env-file ~/.optimizer.env         # Run with explicit environment file

  6. LISTEN — run inbox listener daemon for background agent triggers:
    context-optimizer listen                                        # Listen on .context_optimizer/inbox.jsonl
    context-optimizer listen --dir /opt/jarina --interval 2

  7. CONFIG — inspect or update isolated environment configuration:
    context-optimizer config show
    context-optimizer config path
    context-optimizer config set GEMINI_MODEL=gemini-3.8-flash
    context-optimizer config set USE_VERTEX=true
    context-optimizer config set GCP_PROJECT=jarinagem
"""


def main():
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="context-optimizer",
        formatter_class=fmt,
        epilog=CLI_MAIN_EPILOG,
        description="Intelligent Context Optimizer for AI Agents using Google Antigravity SDK & TypeSafe Jev",
    )
    parser.add_argument(
        "--env-file", type=str, help="Path to isolated environment file"
    )
    parser.add_argument(
        "--allow-global-env",
        action="store_true",
        help="Opt-in to reading ambient global os.environ",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Enable Google Cloud Vertex AI via Application Default Credentials (ADC)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Google Cloud project ID for Vertex AI (auto-detected from ADC if omitted)",
    )
    parser.add_argument(
        "--location",
        type=str,
        help="Google Cloud region/location for Vertex AI (default: us-central1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override Google Gemini model identifier (e.g. gemini-3.8-flash)",
    )
    parser.add_argument(
        "--exclude-patterns",
        type=str,
        help="Override comma-separated exclude glob patterns",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    subparsers = parser.add_subparsers(dest="command")

    # 1. Optimize subcommand
    opt_parser = subparsers.add_parser(
        "optimize",
        formatter_class=fmt,
        help="Optimize context for a given query",
    )
    opt_parser.add_argument(
        "query", nargs="+", help="Query or task instruction to optimize context for"
    )
    opt_parser.add_argument("--root", type=str, help="Root directory (default: cwd)")
    opt_parser.add_argument("--budget", type=int, help="Max token budget")
    opt_parser.add_argument(
        "--force-include", nargs="*", help="File relative paths to force include"
    )
    opt_parser.add_argument(
        "--format",
        choices=["markdown", "json", "jev"],
        default="markdown",
        help="Output format",
    )
    opt_parser.add_argument(
        "--model", type=str, help="Override Gemini model identifier"
    )
    opt_parser.add_argument(
        "--exclude-patterns", type=str, help="Comma-separated exclude glob patterns"
    )
    opt_parser.add_argument(
        "--min-score", type=float, help="Minimum relevance threshold (0.0 to 1.0)"
    )
    opt_parser.add_argument(
        "--context-only",
        action="store_true",
        help="Only output distilled context files without calling Gemini",
    )
    opt_parser.add_argument(
        "--project",
        type=str,
        help="Google Cloud project ID for Vertex AI",
    )
    opt_parser.add_argument(
        "--location",
        type=str,
        help="Google Cloud region/location for Vertex AI",
    )
    opt_parser.add_argument(
        "--vertex",
        action="store_true",
        help="Enable Google Cloud Vertex AI via ADC",
    )
    opt_parser.add_argument(
        "-o", "--output", type=str, help="Write output to file instead of stdout"
    )

    # 2. Trigger subcommand (for other agents)
    trig_parser = subparsers.add_parser(
        "trigger",
        formatter_class=fmt,
        help="Trigger optimization programmatically via JEV event",
    )
    trig_parser.add_argument("--payload", type=str, help="JSON / JEV event string")
    trig_parser.add_argument(
        "--stdin", action="store_true", help="Read JEV event from stdin"
    )
    trig_parser.add_argument(
        "query", nargs="*", help="Direct query string if payload not specified"
    )

    # 3. Listen daemon
    listen_parser = subparsers.add_parser(
        "listen",
        formatter_class=fmt,
        help="Run inbox listener daemon for agent triggers",
    )
    listen_parser.add_argument(
        "--dir", type=str, help="Base directory containing .context_optimizer folder"
    )
    listen_parser.add_argument(
        "--interval", type=int, default=3, help="Polling interval in seconds"
    )

    # 4. Config manager
    cfg_parser = subparsers.add_parser(
        "config",
        formatter_class=fmt,
        help="Manage isolated environment variables",
    )
    cfg_parser.add_argument(
        "config_action", choices=["show", "set", "path"], nargs="?", default="show"
    )
    cfg_parser.add_argument("key_value", nargs="?", help="KEY=VALUE to set")

    # 5. MCP tool schema
    subparsers.add_parser(
        "mcp",
        formatter_class=fmt,
        help="Output MCP tool definition schema",
    )

    # 6. MCP stdio server
    subparsers.add_parser(
        "mcp-server",
        formatter_class=fmt,
        help="Run MCP server over stdio transport",
    )

    # 7. Scan codebase files & AST symbols
    scan_parser = subparsers.add_parser(
        "scan",
        formatter_class=fmt,
        help="Scan and audit repository files, symbols, token counts, and exclusions",
    )
    scan_parser.add_argument("--root", type=str, help="Root directory (default: cwd)")
    scan_parser.add_argument(
        "--symbols", action="store_true", help="Extract and display AST symbols"
    )
    scan_parser.add_argument(
        "--ext", type=str, help="Filter by file extension (e.g. .py, .ts)"
    )
    scan_parser.add_argument(
        "--limit",
        "--top-k",
        dest="limit",
        type=int,
        default=50,
        help="Max files to display in table",
    )
    scan_parser.add_argument(
        "--exclude-patterns", type=str, help="Comma-separated exclude glob patterns"
    )
    scan_parser.add_argument(
        "--json", action="store_true", help="Output raw JSON scan summary"
    )

    # 8. Antigravity AI Agent integration
    agent_parser = subparsers.add_parser(
        "agent",
        formatter_class=fmt,
        help="Run Antigravity AI Agent with distilled codebase context",
    )
    agent_parser.add_argument(
        "prompt", nargs="+", help="Task instruction or prompt for the AI agent"
    )
    agent_parser.add_argument("--root", type=str, help="Root directory (default: cwd)")
    agent_parser.add_argument(
        "--budget", type=int, help="Token budget for context distillation"
    )
    agent_parser.add_argument(
        "--model", type=str, help="Override Gemini model identifier"
    )
    agent_parser.add_argument(
        "--project", type=str, help="Google Cloud project ID for Vertex AI"
    )
    agent_parser.add_argument(
        "--location", type=str, help="Google Cloud region/location for Vertex AI"
    )
    agent_parser.add_argument(
        "--vertex", action="store_true", help="Enable Google Cloud Vertex AI via ADC"
    )
    agent_parser.add_argument(
        "--exclude-patterns", type=str, help="Comma-separated exclude glob patterns"
    )
    agent_parser.add_argument(
        "--thoughts", action="store_true", help="Display internal reasoning stream"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        # If user directly invokes without subcommand: check if query passed
        parser.print_help(sys.stderr)
        sys.exit(0)

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "trigger":
        cmd_trigger(args)
    elif args.command == "listen":
        cmd_listen(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "mcp-server":
        from .mcp_server import main as run_mcp_server

        run_mcp_server(env_file=args.env_file)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "agent":
        cmd_agent(args)


if __name__ == "__main__":
    main()
