from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import default_config
from .models import RunState, TaskRecord, TaskStatus
from .router import load_orchestrator
from .state import load_state, save_state
from .worktree import remove_worktree

app = typer.Typer(help="Codex orchestrator CLI")


def _repo_root() -> Path:
    return Path.cwd()


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


def _default_config_path(repo_root: Path) -> Path:
    return repo_root / ".orchestrator" / "orchestrator.yaml"


def _state_path(paths, run_id: str) -> Path:
    return paths.runs / run_id / "state.json"


@app.command()
def init(
    force: bool = typer.Option(False, help="Overwrite existing orchestrator files"),
) -> None:
    """Create .orchestrator layout with default config, schemas, and prompts."""
    repo_root = _repo_root()
    base_dir = repo_root / ".orchestrator"
    base_dir.mkdir(parents=True, exist_ok=True)
    config_path = _default_config_path(repo_root)

    if config_path.exists() and not force:
        typer.echo(f"Config already exists at {config_path}. Use --force to overwrite.")
    else:
        config = default_config()
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        typer.echo(f"Wrote default config to {config_path}")

    # Copy templates
    templates = _templates_dir()
    for sub in ["schemas", "prompts"]:
        for src in (templates / sub).glob("*"):
            dest = base_dir / sub / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    typer.echo(f"Templates copied to {base_dir}")


@app.command()
def run(
    goal: str = typer.Argument(..., help="High-level goal for the navigator"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier"),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    run_identifier = run_id or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_identifier)
    result = orchestrator.run(goal=goal, run_id=run_identifier, state_path=state_path)
    typer.echo(
        f"Run {run_identifier} completed with {len(result.tasks)} tasks recorded."
    )


@app.command()
def task(
    role: str = typer.Option(..., help="Role to use"),
    prompt: str = typer.Argument(..., help="Prompt for the worker"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier"),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    run_identifier = run_id or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state = RunState(run_id=run_identifier, goal=prompt)
    state_path = _state_path(paths, run_identifier)
    save_state(state_path, state)
    record = TaskRecord(task_id="T0001", role=role, prompt=prompt)
    orchestrator.run_task(run_identifier, state_path, state, record)
    typer.echo(
        f"Task {record.task_id} for role {role} finished with status {record.status.value}."
    )


@app.command()
def resume(
    run_id: str = typer.Option(..., help="Run identifier to resume"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    state = load_state(state_path)

    pending = [
        t
        for t in state.tasks
        if t.status in {TaskStatus.PENDING, TaskStatus.NEEDS_RETRY, TaskStatus.FAILED}
    ]
    if not pending:
        typer.echo("No pending tasks to resume.")
        return

    for task in pending:
        orchestrator.run_task(run_id, state_path, state, task)
    typer.echo(f"Resumed run {run_id}; {len(pending)} tasks executed.")


@app.command()
def report(
    run_id: str = typer.Option(..., help="Run identifier"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    _, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    report_path = paths.runs / run_id / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Run {run_id}", ""]
    for task in state.tasks:
        lines.append(f"## {task.task_id} — {task.role}")
        lines.append(f"- status: {task.status.value}")
        if task.error:
            lines.append(f"- error: {task.error}")
        if task.result and getattr(task.result, "summary", None):
            lines.append(f"- summary: {task.result.summary}")
        lines.append("")
    report_path.write_text("\n".join(lines))
    typer.echo(f"Wrote report to {report_path}")


@app.command()
def clean(
    run_id: str = typer.Option(..., help="Run identifier"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
    keep_branches: bool = typer.Option(False, help="Do not delete branches"),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    _, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    for task in state.tasks:
        if task.worktree_path:
            remove_worktree(
                task.worktree_path,
                repo_root,
                keep_branch=keep_branches,
                branch=task.branch,
            )
    typer.echo(f"Cleaned worktrees for run {run_id}")


@app.command()
def merge(
    run_id: str = typer.Option(..., help="Run identifier"),
    config: Path = typer.Option(None, help="Path to orchestrator YAML config"),
    allow_automerge: bool = typer.Option(
        False, help="Attempt git merge (fast-forward) for each task branch"
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    _, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    if not allow_automerge:
        typer.echo(
            "Automerge disabled; rerun with --allow-automerge to merge branches."
        )
        return
    if not shutil.which("git"):
        typer.echo("git not found on PATH")
        raise typer.Exit(code=1)

    import subprocess

    for task in state.tasks:
        if not task.branch:
            continue
        typer.echo(f"Merging branch {task.branch}...")
        result = subprocess.run(
            ["git", "merge", "--ff-only", task.branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(result.stdout + result.stderr)
            typer.echo(f"Merge failed for {task.branch}; resolve manually.")
            break
    typer.echo("Merge step complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
