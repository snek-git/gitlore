"""Typer CLI for gitlore."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from gitlore.build import build_index
from gitlore.config import DEFAULT_CONFIG_TEMPLATE, GitloreConfig
from gitlore.export import load_export_bundle, write_exports
from gitlore.mcp_server import run_stdio_server
from gitlore.query import build_planning_brief, render_planning_brief

# Load .env from CWD, then fall back to ~/.config/gitlore/.env
load_dotenv()
load_dotenv(Path.home() / ".config" / "gitlore" / ".env")

app = typer.Typer(
    name="gitlore",
    help="Build and retrieve planning-time tribal knowledge for humans and coding agents.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def build(
    repo: Annotated[
        Path, typer.Option("--repo", "-r", help="Path to git repository")
    ] = Path("."),
    since: Annotated[
        str | None, typer.Option("--since", "-s", help="Lookback period (e.g. 6m, 12m)")
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to gitlore.toml")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Skip cached GitHub/LLM reads")
    ] = False,
) -> None:
    """Build the local `.gitlore/index.db` knowledge index."""
    config = GitloreConfig.load(config_path)
    config.repo_path = str(repo.resolve())

    if since:
        months = _parse_since(since)
        if months is not None:
            config.build.since_months = months

    metadata = build_index(config, use_cache=not no_cache, console=console)
    coverage = metadata.source_coverage
    console.print(
        "\n[bold green]Built index[/bold green] "
        f"with {metadata.card_count} advice cards from {metadata.total_commits_analyzed} commits."
    )
    console.print(
        "[dim]"
        f"Sources: git={coverage.git} github={coverage.github} docs={coverage.docs} "
        f"classified_reviews={coverage.classified_reviews} semantic={coverage.semantic}"
        "[/dim]"
    )


@app.command(name="advise")
def advise(
    task: Annotated[str, typer.Option("--task", help="Task description for retrieval")],
    repo: Annotated[
        Path, typer.Option("--repo", "-r", help="Path to git repository")
    ] = Path("."),
    files: Annotated[
        list[str] | None,
        typer.Option("--files", help="Repo-relative files to focus retrieval on"),
    ] = None,
    diff_path: Annotated[
        Path | None,
        typer.Option("--diff", help="Path to a diff/patch file for review tasks"),
    ] = None,
    tentative_plan: Annotated[
        str,
        typer.Option("--plan", help="Tentative plan text to evaluate during retrieval"),
    ] = "",
    question: Annotated[
        str,
        typer.Option("--question", help="Optional planning question for retrieval"),
    ] = "",
    format_name: Annotated[
        str | None,
        typer.Option("--format", help="summary or json"),
    ] = None,
    max_notes: Annotated[
        int | None,
        typer.Option("--max-notes", help="Maximum number of planning notes"),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to gitlore.toml")
    ] = None,
) -> None:
    """Retrieve a small planning brief from the local index."""
    config = GitloreConfig.load(config_path)
    config.repo_path = str(repo.resolve())

    brief = build_planning_brief(
        config,
        task=task,
        files=files or [],
        diff_path=str(diff_path) if diff_path else None,
        tentative_plan=tentative_plan,
        question=question,
        max_notes=max_notes or 5,
    )
    rendered = render_planning_brief(
        brief,
        format_name=format_name or config.query.default_format,
    )
    if (format_name or config.query.default_format) == "json":
        console.file.write(rendered + "\n")
    else:
        console.print(rendered)


@app.command()
def export(
    repo: Annotated[
        Path, typer.Option("--repo", "-r", help="Path to git repository")
    ] = Path("."),
    formats: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Output formats (comma-separated)"),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to gitlore.toml")
    ] = None,
) -> None:
    """Render stable guidance cards from the current index into export artifacts."""
    config = GitloreConfig.load(config_path)
    config.repo_path = str(repo.resolve())
    if formats:
        config.export.formats = [item.strip() for item in formats.split(",") if item.strip()]

    bundle = load_export_bundle(config)
    written = write_exports(bundle, config)
    console.print(f"\n[bold green]Exported[/bold green] {len(written)} files:")
    for path in written:
        console.print(f"  {path}")


@app.command(name="mcp")
def mcp_command(
    repo: Annotated[
        Path, typer.Option("--repo", "-r", help="Path to git repository")
    ] = Path("."),
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to gitlore.toml")
    ] = None,
) -> None:
    """Run the gitlore MCP server over stdio."""
    config = GitloreConfig.load(config_path)
    config.repo_path = str(repo.resolve())
    run_stdio_server(config)


@app.command()
def init(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Where to create gitlore.toml")
    ] = Path("."),
) -> None:
    """Create a gitlore.toml config file."""
    target = path / "gitlore.toml"
    if target.exists():
        console.print(f"[yellow]{target} already exists.[/yellow]")
        raise typer.Exit(1)
    target.write_text(DEFAULT_CONFIG_TEMPLATE)
    console.print(f"[green]Created {target}[/green]")


@app.command()
def auth() -> None:
    """Show which optional GitHub and LLM credentials are currently available."""
    config = GitloreConfig.load()
    token = config.github.resolve_token()
    if token:
        masked = token[:4] + "..." + token[-4:]
        console.print(f"[green]GitHub token found:[/green] {masked}")
    else:
        console.print("[yellow]No GitHub token found.[/yellow]")
        console.print("GitHub enrichment is optional. Set GITHUB_TOKEN or use `gh auth login`.")

    if config.models.classifier or config.models.embedding or config.models.compressor:
        if Path(".env").exists():
            console.print("[green].env detected[/green] — build-time LLM analysis may be available.")
        else:
            console.print("[yellow]No local .env detected.[/yellow]")
            console.print("Build still works with deterministic leads only.")


def _parse_since(value: str) -> int | None:
    value = value.strip().lower()
    if value.endswith("m"):
        try:
            return int(value[:-1])
        except ValueError:
            return None
    return None
