from __future__ import annotations

import json
import shutil
import threading
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
import typer
import yaml

from .config import default_config
from .models import RunState, TaskRecord, TaskStatus
from .router import Orchestrator, load_orchestrator, task_dir
from .state import load_state, save_state
from .worktree import remove_worktree

app = typer.Typer(
    help="Codex orchestrator CLI",
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)


def _repo_root() -> Path:
    return Path.cwd()


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


def _default_config_path(repo_root: Path) -> Path:
    return repo_root / ".orchestrator" / "orchestrator.yaml"


def _state_path(paths, run_id: str) -> Path:
    return paths.runs / run_id / "state.json"


PENDING_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.NEEDS_RETRY,
    TaskStatus.FAILED,
}

SPEC_MAX_BYTES = 256 * 1024


def _load_spec(
    repo_root: Path, spec_file: Path | None
) -> tuple[Path | None, str | None]:
    if not spec_file:
        return None, None
    path = spec_file if spec_file.is_absolute() else repo_root / spec_file
    if not path.exists() or not path.is_file():
        typer.echo(f"Spec file not found or not a file: {path}")
        raise typer.Exit(code=1)
    size = path.stat().st_size
    if size == 0:
        typer.echo(f"Spec file is empty: {path}")
        raise typer.Exit(code=1)
    if size > SPEC_MAX_BYTES:
        typer.echo(f"Spec file exceeds {SPEC_MAX_BYTES} bytes: {path}")
        raise typer.Exit(code=1)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        typer.echo(f"Spec file must be UTF-8 text: {path}")
        raise typer.Exit(code=1)
    return path, text


def _discover_run_ids(paths, explicit: list[str], older_than: int | None) -> list[str]:
    run_base = paths.runs
    if not run_base.exists():
        return []
    candidates = explicit or [p.name for p in run_base.iterdir() if p.is_dir()]
    if older_than is None:
        return sorted(candidates)

    cutoff = datetime.now(UTC) - timedelta(days=older_than)
    filtered: list[str] = []
    for run_id in candidates:
        run_dir = run_base / run_id
        state_path = _state_path(paths, run_id)
        target = state_path if state_path.exists() else run_dir
        try:
            mtime = datetime.fromtimestamp(target.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            continue
        if mtime <= cutoff:
            filtered.append(run_id)
    return sorted(filtered)


def _cleanup_run_artifacts(
    orchestrator: Orchestrator,
    run_id: str,
    repo_root: Path,
    state: RunState | None,
    keep_branches: bool,
) -> list[str]:
    actions: list[str] = []
    if orchestrator.config.use_single_workspace:
        shared_spec = orchestrator.shared_workspace_spec(run_id)
        target_path = (
            next((Path(t.worktree_path) for t in state.tasks if t.worktree_path), None)
            if state
            else None
        ) or shared_spec.path
        target_branch = (
            next((t.branch for t in state.tasks if t.branch), None) if state else None
        ) or shared_spec.branch
        if target_path and target_path.exists() and target_path != repo_root:
            remove_worktree(
                target_path,
                repo_root,
                keep_branch=keep_branches,
                branch=target_branch,
            )
            actions.append(f"removed worktree {target_path}")
        return actions

    if state is None:
        return actions

    seen_paths: set[Path] = set()
    for task in state.tasks:
        if not task.worktree_path:
            continue
        worktree_path = Path(task.worktree_path)
        if worktree_path in seen_paths:
            continue
        seen_paths.add(worktree_path)
        if worktree_path.exists() and worktree_path != repo_root:
            remove_worktree(
                worktree_path,
                repo_root,
                keep_branch=keep_branches,
                branch=task.branch,
            )
            actions.append(f"removed worktree {worktree_path}")
    return actions


def _status_color(status: TaskStatus) -> str:
    if status == TaskStatus.COMPLETED:
        return "green"
    if status == TaskStatus.RUNNING:
        return "yellow"
    # pending/failed/needs_retry are treated as not started/blocked
    return "red"


def _shorten(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _last_activity(paths, run_id: str, task_id: str, fallback: str | None) -> str | None:
    events_path = task_dir(paths, run_id, task_id) / "events.jsonl"
    if not events_path.exists():
        return fallback
    try:
        lines = events_path.read_text().strip().splitlines()
    except OSError:
        return fallback
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text")
        item = obj.get("item") or {}
        if not text and isinstance(item, dict):
            if item.get("type") == "command_execution":
                text = item.get("command")
            elif item.get("type") == "mcp_tool_call":
                text = f"tool {item.get('tool')}"
            elif item.get("type") == "agent_message":
                text = item.get("text")
        if text:
            return _shorten(str(text))
    return fallback


def _render_status(
    paths, run_id: str, header: str | None = None, clear_first: bool = False
) -> None:
    if clear_first:
        typer.clear()
    state_path = _state_path(paths, run_id)
    if header:
        typer.echo(f"{header} ({datetime.now().isoformat(timespec='seconds')})")
    if not state_path.exists():
        typer.echo("No state file yet.")
        return
    try:
        state = load_state(state_path)
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"Could not read state: {exc}")
        return
    for task in state.tasks:
        color = _status_color(task.status)
        status_text = typer.style(task.status.value, fg=color)
        typer.echo(
            f"{task.task_id} [{task.role}] {status_text} - {_shorten(task.prompt)}"
        )
        detail = _last_activity(paths, run_id, task.task_id, task.prompt)
        if detail:
            typer.echo(f"    {detail}")


def _status_loop(
    paths, run_id: str, stop_event: threading.Event, interval: float
) -> None:
    while not stop_event.is_set():
        _render_status(paths, run_id, clear_first=True)
        stop_event.wait(interval)


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing orchestrator files"
    ),
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
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run identifier"),
    spec_file: Path | None = typer.Option(
        None, "--spec-file", help="Path to a spec file to include in prompts"
    ),
    status_interval: float = typer.Option(
        3.0,
        "--status-interval",
        help="Seconds between status updates while the run is active.",
        min=0.2,
    ),
    show_status: bool = typer.Option(
        True,
        "--status/--no-status",
        help="Show live task status while the run is active.",
    ),
    prune_on_complete: bool = typer.Option(
        True,
        "--prune-on-complete/--keep-worktree",
        help="Remove worktree/run artifacts after a completed run.",
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    run_identifier = run_id or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    spec_path, spec_text = _load_spec(repo_root, spec_file)
    orchestrator.spec_file = spec_path
    orchestrator.spec_text = spec_text
    state_path = _state_path(paths, run_identifier)
    stop_event: threading.Event | None = None
    status_thread: threading.Thread | None = None

    if show_status:
        stop_event = threading.Event()
        status_thread = threading.Thread(
            target=_status_loop,
            args=(paths, run_identifier, stop_event, status_interval),
            daemon=True,
        )
        status_thread.start()

    try:
        result = orchestrator.run(
            goal=goal, run_id=run_identifier, state_path=state_path
        )
    finally:
        if stop_event:
            stop_event.set()
        if status_thread:
            status_thread.join(timeout=2)

    _render_status(paths, run_identifier, header="Final status")
    typer.echo(
        f"Run {run_identifier} completed with {len(result.tasks)} tasks recorded."
    )
    if prune_on_complete:
        actions = _cleanup_run_artifacts(
            orchestrator=orchestrator,
            run_id=run_identifier,
            repo_root=repo_root,
            state=result,
            keep_branches=False,
        )
        run_dir = paths.runs / run_identifier
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            actions.append(f"removed run dir {run_dir}")
        if actions:
            typer.echo("Cleanup:")
            for line in actions:
                typer.echo(f"- {line}")


@app.command()
def task(
    role: str = typer.Option(..., "--role", help="Role to use"),
    prompt: str = typer.Argument(..., help="Prompt for the worker"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run identifier"),
    spec_file: Path | None = typer.Option(
        None, "--spec-file", help="Path to a spec file to include in prompts"
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    run_identifier = run_id or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    spec_path, spec_text = _load_spec(repo_root, spec_file)
    orchestrator.spec_file = spec_path
    orchestrator.spec_text = spec_text
    orchestrator.goal = prompt
    state = RunState(
        run_id=run_identifier, goal=prompt, spec_file=spec_path, spec_text=spec_text
    )
    state_path = _state_path(paths, run_identifier)
    save_state(state_path, state)
    record = TaskRecord(task_id="T0001", role=role, prompt=prompt)
    orchestrator.run_task(run_identifier, state_path, state, record)
    typer.echo(
        f"Task {record.task_id} for role {role} finished with status {record.status.value}."
    )


@app.command()
def validate(
    run_id: str = typer.Option(..., "--run-id", help="Run identifier to validate"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    spec_file: Path | None = typer.Option(
        None,
        "--spec-file",
        help="Override spec file path for context (optional).",
    ),
    pytest_args: str = typer.Option(
        "poetry run pytest",
        "--pytest-args",
        help="Command to run tests (default: poetry run pytest).",
    ),
    lint_cmd: str = typer.Option(
        "poetry run ruff check .",
        "--lint-cmd",
        help="Command to run linting (default: poetry run ruff check .).",
    ),
) -> None:
    """Run validation (lint/tests) against a completed run's workspace."""
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    stored_spec = state.spec_file
    spec_path, spec_text = (
        _load_spec(repo_root, spec_file)
        if spec_file
        else _load_spec(repo_root, stored_spec)  # type: ignore[arg-type]
    )
    orchestrator.spec_file = spec_path
    orchestrator.spec_text = spec_text

    worktree_path: Path | None = None
    if orchestrator.config.use_single_workspace:
        worktree_path = orchestrator.shared_workspace_spec(run_id).path
    else:
        for task in state.tasks:
            if task.worktree_path:
                worktree_path = Path(task.worktree_path)
                break
    if not worktree_path or not worktree_path.exists():
        typer.echo("Worktree not found; cannot validate.")
        raise typer.Exit(code=1)

    typer.echo(f"Validating run {run_id} in {worktree_path}")

    def _run_cmd(cmd: str, label: str) -> int:
        typer.echo(f"- {label}: {cmd}")
        proc = subprocess.run(cmd, shell=True, cwd=worktree_path)
        if proc.returncode != 0:
            typer.echo(f"{label} failed (exit {proc.returncode})")
        return proc.returncode

    failures = 0
    failures += 1 if _run_cmd(lint_cmd, "Lint") else 0
    failures += 1 if _run_cmd(pytest_args, "Tests") else 0

    if failures:
        typer.echo(f"Validation failed with {failures} error(s).")
        raise typer.Exit(code=1)
    typer.echo("Validation succeeded.")


@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id", help="Run identifier to resume"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    spec_file: Path | None = typer.Option(
        None,
        "--spec-file",
        help="Override spec file path to use when resuming tasks (defaults to stored state)",
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    stored_spec = state.spec_file
    spec_path, spec_text = (
        _load_spec(repo_root, spec_file)
        if spec_file
        else _load_spec(repo_root, stored_spec)  # type: ignore[arg-type]
    )
    if stored_spec and not spec_path:
        typer.echo(
            "Stored spec file missing. Provide --spec-file to resume with a valid spec."
        )
        raise typer.Exit(code=1)
    state.spec_file = spec_path
    state.spec_text = spec_text
    save_state(state_path, state)
    orchestrator.spec_file = spec_path
    orchestrator.spec_text = spec_text
    orchestrator.goal = state.goal

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
    run_id: str = typer.Option(..., "--run-id", help="Run identifier"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
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
    run_id: str = typer.Option(..., "--run-id", help="Run identifier"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    keep_branches: bool = typer.Option(
        False, "--keep-branches", help="Do not delete branches"
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    state_path = _state_path(paths, run_id)
    if not state_path.exists():
        typer.echo(f"State not found for run {run_id}: {state_path}")
        raise typer.Exit(code=1)
    state = load_state(state_path)
    if orchestrator.config.use_single_workspace:
        spec = orchestrator.shared_workspace_spec(run_id)
        target_path = next(
            (t.worktree_path for t in state.tasks if t.worktree_path), spec.path
        )
        target_branch = next((t.branch for t in state.tasks if t.branch), spec.branch)
        if not target_path or target_path == repo_root or not target_path.exists():
            typer.echo("No shared worktree to clean for this run.")
            return
        remove_worktree(
            target_path,
            repo_root,
            keep_branch=keep_branches,
            branch=target_branch,
        )
        typer.echo(f"Cleaned shared worktree at {target_path}")
        return
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
    run_id: str = typer.Option(..., "--run-id", help="Run identifier"),
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    allow_automerge: bool = typer.Option(
        False,
        "--allow-automerge",
        help="Attempt git merge (fast-forward) for each task branch",
    ),
) -> None:
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
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

    branches: list[str] = []
    if orchestrator.config.use_single_workspace:
        spec = orchestrator.shared_workspace_spec(run_id)
        branch = next((t.branch for t in state.tasks if t.branch), spec.branch)
        if branch:
            branches = [branch]
    else:
        seen: set[str] = set()
        for task in state.tasks:
            if not task.branch or task.branch in seen:
                continue
            seen.add(task.branch)
            branches.append(task.branch)

    if not branches:
        typer.echo("No branches recorded for merge.")
        return

    for branch in branches:
        typer.echo(f"Merging branch {branch}...")
        result = subprocess.run(
            ["git", "merge", "--ff-only", branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(result.stdout + result.stderr)
            typer.echo(f"Merge failed for {branch}; resolve manually.")
            break
    typer.echo("Merge step complete.")


@app.command()
def prune(
    config: Path | None = typer.Option(
        None, "--config", help="Path to orchestrator YAML config"
    ),
    spec_file: Path | None = typer.Option(
        None,
        "--spec-file",
        help="Override spec file path to use when resuming tasks (defaults to stored state)",
    ),
    run_id: list[str] = typer.Option(
        [],
        "--run-id",
        help="Run identifier(s) to prune; defaults to all runs.",
    ),
    older_than: int | None = typer.Option(
        None,
        "--older-than",
        help="Only prune runs whose state is older than N days.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply cleanup; default is dry-run.",
    ),
    resume_pending: bool = typer.Option(
        True,
        "--resume-pending/--no-resume",
        help="Resume pending/failed tasks before cleanup.",
    ),
    keep_branches: bool = typer.Option(
        False,
        "--keep-branches",
        help="Do not delete branches when removing worktrees.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Proceed with cleanup even if tasks fail to resume or remain pending.",
    ),
) -> None:
    """Sweep runs, optionally resume pending tasks, and remove artifacts."""
    repo_root = _repo_root()
    config_path = config or _default_config_path(repo_root)
    orchestrator, paths = load_orchestrator(config_path, repo_root)
    run_ids = _discover_run_ids(paths, run_id, older_than)
    if not run_ids:
        typer.echo("No runs found to prune.")
        return

    runs_scanned = 0
    runs_cleaned = 0
    tasks_resumed = 0
    failures = 0
    forced_runs = 0
    mode = "apply" if apply else "dry-run"
    typer.echo(f"Prune ({mode}): processing {len(run_ids)} run(s).")

    for rid in run_ids:
        runs_scanned += 1
        run_dir = paths.runs / rid
        state_path = _state_path(paths, rid)
        if not run_dir.exists():
            typer.echo(f"- {rid}: run directory not found; skipping.")
            failures += 1
            continue
        state = load_state(state_path) if state_path.exists() else None
        stored_spec = state.spec_file if state else None
        spec_path, spec_text = (
            _load_spec(repo_root, spec_file)
            if spec_file
            else _load_spec(repo_root, stored_spec)  # type: ignore[arg-type]
        )
        if stored_spec and not spec_path:
            failures += 1
            typer.echo(
                f"- {rid}: stored spec file missing; provide --spec-file to prune this run."
            )
            continue
        if state:
            state.spec_file = spec_path
            state.spec_text = spec_text
            save_state(state_path, state)
        orchestrator.spec_file = spec_path
        orchestrator.spec_text = spec_text
        orchestrator.goal = state.goal if state else None
        pending = (
            [t for t in state.tasks if t.status in PENDING_STATUSES] if state else []
        )

        run_failed = False
        resumed_count = 0
        if resume_pending and pending:
            for task in pending:
                updated = orchestrator.run_task(rid, state_path, state, task)  # type: ignore[arg-type]
                if updated.status != TaskStatus.COMPLETED:
                    run_failed = True
                else:
                    resumed_count += 1
            tasks_resumed += resumed_count
            remaining = (
                [t for t in state.tasks if t.status in PENDING_STATUSES]
                if state
                else []
            )
            if remaining:
                run_failed = True
        elif pending:
            run_failed = True

        forced_cleanup = run_failed and force

        if run_failed and not force:
            failures += 1
            typer.echo(
                f"- {rid}: pending/failed tasks remain; skipping cleanup (use --force to override)."
            )
            continue

        if not apply:
            typer.echo(
                f"- {rid}: pending={len(pending)}, resumed={resumed_count}; would clean run dir and worktrees."
            )
            continue

        actions = _cleanup_run_artifacts(
            orchestrator=orchestrator,
            run_id=rid,
            repo_root=repo_root,
            state=state,
            keep_branches=keep_branches,
        )
        if run_dir.exists():
            shutil.rmtree(run_dir)
            actions.append(f"removed run dir {run_dir}")

        runs_cleaned += 1
        if forced_cleanup:
            forced_runs += 1
        elif run_failed:
            failures += 1
        detail = "; ".join(actions) if actions else "no artifacts removed"
        typer.echo(f"- {rid}: cleaned ({detail})")

    summary = (
        f"Prune complete: scanned {runs_scanned}, cleaned {runs_cleaned}, "
        f"resumed {tasks_resumed}, failures {failures}"
    )
    if forced_runs:
        summary += f", forced {forced_runs}"
    summary += "."
    typer.echo(summary)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
