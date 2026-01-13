import json
from pathlib import Path

import pytest
import typer

from codex_orch.cli import _load_spec
from codex_orch.models import (
    CodexConfig,
    ConcurrencyConfig,
    GitConfig,
    LimitsConfig,
    OrchestratorConfig,
    PathsConfig,
    RoleConfig,
    RunState,
    TaskRecord,
)
from codex_orch.router import Orchestrator
from codex_orch.runner import CodexExecutionResult


def _init_dirs(repo_root: Path) -> None:
    paths = PathsConfig().resolve_in_repo(repo_root)
    paths.runs.mkdir(parents=True, exist_ok=True)
    paths.worktrees.mkdir(parents=True, exist_ok=True)


def test_load_spec_reads_text(tmp_path: Path) -> None:
    repo_root = tmp_path
    spec = repo_root / "spec.md"
    spec.write_text("hello", encoding="utf-8")

    path, text = _load_spec(repo_root, spec)

    assert path == spec
    assert text == "hello"


def test_load_spec_missing_file_exits(tmp_path: Path) -> None:
    repo_root = tmp_path
    with pytest.raises(typer.Exit):
        _load_spec(repo_root, repo_root / "missing.md")


def test_run_task_appends_spec_to_prompt(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    _init_dirs(repo_root)
    # Avoid git worktree creation by pre-creating the shared workspace path.
    shared_path = repo_root / ".orchestrator" / "worktrees" / "run-1"
    shared_path.mkdir(parents=True, exist_ok=True)

    roles = [
        RoleConfig(
            name="navigator",
            profile="navigator",
            sandbox="workspace-write",
            full_auto=True,
            output_schema=None,
            prompt_template=None,
        )
    ]
    config = OrchestratorConfig(
        roles=roles,
        limits=LimitsConfig(),
        paths=PathsConfig(),
        git=GitConfig(),
        codex=CodexConfig(),
        concurrency=ConcurrencyConfig(),
        use_single_workspace=True,
        role_aliases={},
    )
    paths = config.paths.resolve_in_repo(repo_root)
    orchestrator = Orchestrator(config=config, paths=paths, repo_root=repo_root)
    orchestrator.goal = "demo-goal"
    orchestrator.spec_file = repo_root / "spec.md"
    orchestrator.spec_text = "Important spec details."

    captured: dict[str, str] = {}

    def fake_run_codex(cmd, jsonl_log_path, output_path, env=None):
        captured["prompt"] = cmd[-1]
        output = {
            "summary": "done",
            "decisions": [],
            "changes": [],
            "commands_ran": [],
            "risks": [],
            "next_tasks": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output))
        return CodexExecutionResult(
            exit_code=0,
            output=output,
            output_path=output_path,
            jsonl_log=jsonl_log_path,
            raw_last_message=None,
        )

    monkeypatch.setattr("codex_orch.router.run_codex_process", fake_run_codex)

    state = RunState(
        run_id="run-1",
        goal="demo-goal",
        spec_file=orchestrator.spec_file,
        spec_text=orchestrator.spec_text,
    )
    state_path = paths.runs / "run-1" / "state.json"
    task = TaskRecord(task_id="T0001", role="navigator", prompt="Do work")

    orchestrator.run_task("run-1", state_path, state, task)

    assert "Specification:\nImportant spec details." in captured["prompt"]
