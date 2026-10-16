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
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import IsolatedConfig, DEFAULT_ENV_FILE
from .mcp_tool import MCP_TOOL_DEFINITION
from .optimizer import ContextOptimizer
from .triggers import TriggerInboxWatcher, execute_trigger_request

console = Console(stderr=True)  # Logs and status go to stderr
stdout_console = Console()      # Payload output goes to stdout


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
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

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None
    )

    auth_mode = f"Vertex AI (ADC, project: {config.gcp_project or 'auto'})" if config.use_vertex else "Gemini API Key"
    console.print(f"[bold cyan]🔍 Scanning directory:[/] {root_dir}")
    console.print(f"[bold cyan]🎯 Optimization query:[/] {query}")
    console.print(f"[dim]Auth: {auth_mode} | Env isolation: {'enabled' if config.isolate_env else 'disabled'} | Budget: {args.budget or config.max_context_tokens} tokens[/dim]")

    optimizer = ContextOptimizer(config=config, root_dir=root_dir)
    res = optimizer.optimize(
        query=query,
        token_budget=args.budget,
        force_include=args.force_include
    )

    if not res.guardrail_passed:
        console.print(f"[bold red]❌ Guardrail check rejected query:[/] {res.guardrail_message}")
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
            str(bundle.token_cost)
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
    else:
        # Print raw content to stdout
        sys.stdout.write(output_text)
        if not output_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()


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
        console.print("[bold red]Error:[/] Must provide --payload, --query, or send via stdin.")
        sys.exit(1)

    overrides = {}
    if getattr(args, "vertex", False):
        overrides["use_vertex"] = True
    if getattr(args, "project", None):
        overrides["gcp_project"] = args.project
    if getattr(args, "location", None):
        overrides["gcp_location"] = args.location

    config = IsolatedConfig.load(
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env,
        overrides=overrides or None
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
        env_file_path=args.env_file,
        allow_global_env=args.allow_global_env
    )

    watcher = TriggerInboxWatcher(
        base_dir=base_dir,
        config=config,
        on_result=lambda res: console.print(f"[green]✔ Processed trigger:[/] {res.get('id')} ({res.get('type')})")
    )

    console.print(Panel(
        f"[bold green]Context Optimizer Agent Listener active[/]\n"
        f"Watching inbox: [cyan]{watcher.inbox_dir}[/]\n"
        f"Writing outbox: [cyan]{watcher.outbox_dir}[/]\n"
        f"Interval: {args.interval}s | Press Ctrl+C to stop",
        title="Agent Trigger Listener"
    ))

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
            console.print("[bold red]Error:[/] Format must be KEY=VALUE (e.g. GEMINI_API_KEY=xyz)")
            sys.exit(1)
        k, v = args.key_value.split("=", 1)
        target = Path(args.env_file).resolve() if args.env_file else DEFAULT_ENV_FILE
        saved_path = IsolatedConfig.save_variable(k.strip(), v.strip(), target_path=target)
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


def main():
    parser = argparse.ArgumentParser(
        prog="context-optimizer",
        description="Intelligent Context Optimizer for AI Agents using Google Antigravity SDK & TypeSafe Jev"
    )
    parser.add_argument("--env-file", type=str, help="Path to isolated environment file")
    parser.add_argument("--allow-global-env", action="store_true", help="Opt-in to reading ambient global os.environ")
    parser.add_argument("--vertex", action="store_true", help="Enable Google Cloud Vertex AI via Application Default Credentials (ADC)")
    parser.add_argument("--project", type=str, help="Google Cloud project ID for Vertex AI (auto-detected from ADC if omitted)")
    parser.add_argument("--location", type=str, help="Google Cloud region/location for Vertex AI (default: us-central1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # 1. Optimize subcommand
    opt_parser = subparsers.add_parser("optimize", help="Optimize context for a given query")
    opt_parser.add_argument("query", nargs="+", help="Query or task instruction to optimize context for")
    opt_parser.add_argument("--root", type=str, help="Root directory (default: cwd)")
    opt_parser.add_argument("--budget", type=int, help="Max token budget")
    opt_parser.add_argument("--force-include", nargs="*", help="File relative paths to force include")
    opt_parser.add_argument("--format", choices=["markdown", "json", "jev"], default="markdown", help="Output format")
    opt_parser.add_argument("-o", "--output", type=str, help="Write output to file instead of stdout")

    # 2. Trigger subcommand (for other agents)
    trig_parser = subparsers.add_parser("trigger", help="Trigger optimization programmatically via JEV event")
    trig_parser.add_argument("--payload", type=str, help="JSON / JEV event string")
    trig_parser.add_argument("--stdin", action="store_true", help="Read JEV event from stdin")
    trig_parser.add_argument("query", nargs="*", help="Direct query string if payload not specified")

    # 3. Listen daemon
    listen_parser = subparsers.add_parser("listen", help="Run inbox listener daemon for agent triggers")
    listen_parser.add_argument("--dir", type=str, help="Base directory containing .context_optimizer folder")
    listen_parser.add_argument("--interval", type=int, default=3, help="Polling interval in seconds")

    # 4. Config manager
    cfg_parser = subparsers.add_parser("config", help="Manage isolated environment variables")
    cfg_parser.add_argument("config_action", choices=["show", "set", "path"], nargs="?", default="show")
    cfg_parser.add_argument("key_value", nargs="?", help="KEY=VALUE to set")

    # 5. MCP tool schema
    subparsers.add_parser("mcp", help="Output MCP tool definition schema")

    # 6. MCP stdio server
    subparsers.add_parser("mcp-server", help="Run MCP server over stdio transport")

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
        run_mcp_server()


if __name__ == "__main__":
    main()
