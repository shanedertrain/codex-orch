import subprocess
from pathlib import Path

import yaml

from codex_orch.config import default_config, load_config
from codex_orch.models import TaskRecord
from codex_orch.router import Orchestrator


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


def test_branch_naming_uses_template(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(default_config()))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    task = TaskRecord(task_id="T0001", role="implementer", prompt="demo")
    branch = orch._branch_name(run_id="run-123", task=task)
    assert "run-123" in branch


def test_prepare_shared_workspace(tmp_path: Path) -> None:
    repo_root = tmp_path
    _init_repo(repo_root)
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    config_data["use_single_workspace"] = True
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    task = TaskRecord(task_id="T0001", role="implementer", prompt="demo")
    spec = orch._prepare_worktree(run_id="run-123", task=task)
    assert task.worktree_path == resolved.worktrees / "run-123"
    assert spec.path.exists()
    assert task.branch == spec.branch
    assert "workspace-shared" in task.branch

    follow_up = TaskRecord(task_id="T0002", role="tester", prompt="check")
    spec_again = orch._prepare_worktree(run_id="run-123", task=follow_up)
    assert spec_again.path == spec.path
    assert follow_up.worktree_path == spec.path
