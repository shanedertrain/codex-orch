import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from typer.testing import CliRunner

from codex_orch.cli import app
from codex_orch.config import default_config
from codex_orch.models import TaskStatus

runner = CliRunner()


def _init_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "orch@example.com"], cwd=repo_root, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex Orch"], cwd=repo_root, check=True
    )
    (repo_root / "README.md").write_text("demo")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_config(repo_root: Path, use_single_workspace: bool = True) -> None:
    cfg = default_config()
    cfg["use_single_workspace"] = use_single_workspace
    cfg_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))


def _write_state(
    repo_root: Path,
    run_id: str,
    status: str,
    worktree_path: Path | None = None,
    branch: str | None = None,
) -> Path:
    run_dir = repo_root / ".orchestrator" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "goal": "demo",
        "tasks": [
            {
                "task_id": "T0001",
                "role": "implementer",
                "prompt": "demo",
                "status": status,
                "worktree_path": str(worktree_path) if worktree_path else None,
                "branch": branch,
                "attempts": 0,
            }
        ],
    }
    (run_dir / "state.json").write_text(json.dumps(state, indent=2))
    return run_dir


def test_prune_dry_run_keeps_runs(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_config(tmp_path)
    run_dir = _write_state(tmp_path, "run-1", status=TaskStatus.COMPLETED.value)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["prune"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert run_dir.exists()


def test_prune_apply_removes_shared_worktree_and_run_dir(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_config(tmp_path)
    run_id = "run-apply"
    shared_path = tmp_path / ".orchestrator" / "worktrees" / run_id
    branch = f"orch/workspace/{run_id}"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "-B", branch, str(shared_path), "HEAD"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    run_dir = _write_state(
        tmp_path,
        run_id,
        status=TaskStatus.COMPLETED.value,
        worktree_path=shared_path,
        branch=branch,
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["prune", "--apply", "--no-resume"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert not run_dir.exists()
    assert not shared_path.exists()
    branches = subprocess.run(
        ["git", "branch", "--list", branch],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branches.stdout.strip() == ""


def test_prune_honors_older_than_filter(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_config(tmp_path)
    old_run = _write_state(tmp_path, "old-run", status=TaskStatus.COMPLETED.value)
    new_run = _write_state(tmp_path, "new-run", status=TaskStatus.COMPLETED.value)
    two_days_ago = datetime.now() - timedelta(days=2)
    os.utime(old_run, (two_days_ago.timestamp(), two_days_ago.timestamp()))
    os.utime(
        old_run / "state.json", (two_days_ago.timestamp(), two_days_ago.timestamp())
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(
            app,
            ["prune", "--apply", "--no-resume", "--older-than", "1"],
        )
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert not old_run.exists()
    assert new_run.exists()


def test_prune_force_does_not_count_failure(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_config(tmp_path)
    run_dir = _write_state(
        tmp_path, "run-force", status=TaskStatus.PENDING.value, worktree_path=None
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["prune", "--apply", "--no-resume", "--force"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert "failures 0" in result.output
    assert "forced 1" in result.output
    assert not run_dir.exists()
