"""Typer CLI for gitlore."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from gitlore.config import DEFAULT_CONFIG_TEMPLATE, GitloreConfig

load_dotenv()

app = typer.Typer(
    name="gitlore",
    help="Mine git history and PR reviews to generate AI coding assistant configs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def analyze(
    repo: Annotated[
        Path, typer.Option("--repo", "-r", help="Path to git repository")
    ] = Path("."),
    git_only: Annotated[
        bool, typer.Option("--git-only", help="Skip GitHub PR comment analysis")
    ] = False,
    since: Annotated[
        str | None, typer.Option("--since", "-s", help="Lookback period (e.g. 6m, 12m)")
    ] = None,
    formats: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Output formats (comma-separated)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview without writing files")
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to gitlore.toml")
    ] = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Log all LLM calls (input/output)")
    ] = False,
) -> None:
    """Analyze git history and generate AI coding assistant config files."""
    import logging

    # Always log to file
    log_path = repo / ".gitlore.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    gitlore_logger = logging.getLogger("gitlore")
    gitlore_logger.setLevel(logging.DEBUG)
    gitlore_logger.addHandler(file_handler)

    if debug:
        import litellm as _litellm

        _litellm.suppress_debug_info = True
        _litellm.set_verbose = False

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("\n%(name)s\n%(message)s\n"))
        gitlore_logger.addHandler(stream_handler)

    from gitlore.pipeline import run_pipeline

    config = GitloreConfig.load(config_path)
    config.repo_path = str(repo.resolve())

    if since:
        months = _parse_since(since)
        if months:
            config.analysis.since_months = months

    if formats:
        config.output.formats = [f.strip() for f in formats.split(",")]

    with console.status("[bold green]Analyzing repository..."):
        result = run_pipeline(config, git_only=git_only)

    if dry_run:
        console.print(f"\n[bold]Generated CLAUDE.md content:[/bold]\n")
        console.print(result.content)
        console.print("\n[dim]Dry run — no files written.[/dim]")
    else:
        from gitlore.pipeline import write_outputs

        written = write_outputs(result, config)
        console.print(f"\n[bold green]Done![/bold green] Wrote {len(written)} files:")
        for path in written:
            console.print(f"  {path}")


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
    """Configure GitHub authentication."""
    config = GitloreConfig.load()
    token = config.github.resolve_token()
    if token:
        masked = token[:4] + "..." + token[-4:]
        console.print(f"[green]GitHub token found:[/green] {masked}")
    else:
        console.print("[yellow]No GitHub token found.[/yellow]")
        console.print("Set GITHUB_TOKEN env var or install gh CLI and run `gh auth login`.")


def _parse_since(value: str) -> int | None:
    """Parse a human duration like '6m' or '12m' into months."""
    value = value.strip().lower()
    if value.endswith("m"):
        try:
            return int(value[:-1])
        except ValueError:
            return None
    return None
