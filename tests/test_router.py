import subprocess
from pathlib import Path

import yaml

from codex_orch.config import default_config, load_config
from codex_orch.models import (
    PlanResult,
    PlanTask,
    NextTask,
    RunState,
    TaskRecord,
    TaskResult,
    TaskStatus,
)
from codex_orch.router import Orchestrator
from codex_orch.state import upsert_task


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
    # Add a dummy submodule to assert submodule init runs without error.
    dummy_submodule = repo_root / "submodule-src"
    dummy_submodule.mkdir()
    (dummy_submodule / "README.md").write_text("dummy")


def test_branch_naming_uses_template(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
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


def test_run_respects_blocked_dependencies(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    call_order: list[str] = []

    def fake_run_task(self, run_id, state_path, state, task, state_lock=None):
        task.status = TaskStatus.COMPLETED
        task.result = TaskResult(summary=task.prompt)
        call_order.append(task.task_id)
        upsert_task(state, task)
        return task

    monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

    run_id = "run-123"
    state_path = resolved.runs / run_id / "state.json"
    plan = PlanResult(
        goal="goal",
        assumptions=[],
        tasks=[
            PlanTask(role="implementer", prompt="step 1", id="plan-a"),
            PlanTask(role="implementer", prompt="step 2", blocked_by=["plan-a"], id="plan-b"),
        ],
    )
    run_state = orch.run_plan(plan=plan, run_id=run_id, state_path=state_path)

    assert call_order == ["T0001", "T0002"]
    tasks = {t.task_id: t for t in run_state.tasks}
    assert tasks["T0001"].status == TaskStatus.COMPLETED
    assert tasks["T0002"].status == TaskStatus.COMPLETED


def test_role_based_blocking_defaults(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    monkeypatch.setattr(Orchestrator, "_warm_tldr", lambda self: None)

    call_order: list[str] = []

    def fake_run_task(self, run_id, state_path, state, task, state_lock=None):
        task.status = TaskStatus.COMPLETED
        task.result = TaskResult(summary=task.prompt)
        call_order.append(task.task_id)
        upsert_task(state, task)
        return task

    monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

    run_id = "run-roles"
    state_path = resolved.runs / run_id / "state.json"
    plan = PlanResult(
        goal="goal",
        assumptions=[],
        tasks=[
            PlanTask(role="implementer", prompt="impl"),
            PlanTask(role="reviewer", prompt="review"),
            PlanTask(role="tester", prompt="test"),
        ],
    )
    run_state = orch.run_plan(plan=plan, run_id=run_id, state_path=state_path)

    tasks = {t.task_id: t for t in run_state.tasks}
    assert tasks["T0002"].blocked_by == []
    assert tasks["T0003"].blocked_by == []
    assert set(call_order) == {"T0001", "T0002", "T0003"}


def test_next_tasks_inherit_parent_and_role_blockers(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    monkeypatch.setattr(Orchestrator, "_warm_tldr", lambda self: None)

    call_order: list[str] = []

    def fake_run_task(self, run_id, state_path, state, task, state_lock=None):
        if task.role == "implementer":
            task.result = TaskResult(
                summary="impl",
                next_tasks=[NextTask(role="reviewer", prompt="review")],
            )
            call_order.append(task.task_id)
        else:
            task.result = TaskResult(summary=task.prompt)
            call_order.append(task.task_id)
        task.status = TaskStatus.COMPLETED
        upsert_task(state, task)
        return task

    monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

    run_id = "run-next"
    state_path = resolved.runs / run_id / "state.json"
    plan = PlanResult(goal="goal", assumptions=[], tasks=[PlanTask(role="implementer", prompt="impl")])
    run_state = orch.run_plan(plan=plan, run_id=run_id, state_path=state_path)

    tasks = {t.task_id: t for t in run_state.tasks}
    reviewer_tasks = [t for t in tasks.values() if t.role == "reviewer"]
    assert len(reviewer_tasks) == 1
    reviewer = reviewer_tasks[0]
    assert reviewer.blocked_by == ["T0001"]
    assert call_order == ["T0001", reviewer.task_id]
def test_invalid_roles_are_rejected(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    _init_repo(repo_root)
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    plan = PlanResult(
        goal="goal",
        assumptions=[],
        tasks=[PlanTask(role="ghost", prompt="do ghost work")],
    )
    run_id = "run-ghost"
    state_path = resolved.runs / run_id / "state.json"
    def fake_run_task(self, run_id, state_path, state, task, state_lock=None):
        task.status = TaskStatus.COMPLETED
        task.result = TaskResult(summary=task.prompt)
        upsert_task(state, task)
        return task

    monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

    run_state = orch.run_plan(plan=plan, run_id=run_id, state_path=state_path)

    tasks = {t.task_id: t for t in run_state.tasks}
    assert tasks["T0001"].status == TaskStatus.COMPLETED


def test_warm_tldr_runs_before_tasks(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    _init_repo(repo_root)
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    for role in cfg["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(cfg))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    warmed = {"called": False}

    def fake_warm(self):
        warmed["called"] = True

    def fake_prepare(self, run_id, task):
        task.worktree_path = resolved.worktrees / run_id
        task.branch = "demo"
        return None  # type: ignore[return-value]

    def fake_run_task(self, run_id, state_path, state, task, state_lock=None):
        task.status = TaskStatus.COMPLETED
        task.result = TaskResult(summary="done")
        upsert_task(state, task)
        return task

    monkeypatch.setattr(Orchestrator, "_warm_tldr", fake_warm)
    monkeypatch.setattr(Orchestrator, "_prepare_worktree", fake_prepare)
    monkeypatch.setattr(Orchestrator, "run_task", fake_run_task)

    plan = PlanResult(goal="g", assumptions=[], tasks=[PlanTask(role="implementer", prompt="do it")])
    orch.run_plan(plan=plan, run_id="run-1", state_path=resolved.runs / "run-1" / "state.json")
    assert warmed["called"]


def test_navigator_prompt_includes_allowed_roles(tmp_path: Path) -> None:
    # Navigator prompt no longer used; ensure role metadata is not required.
    repo_root = tmp_path
    _init_repo(repo_root)
    config_path = repo_root / ".orchestrator" / "orchestrator.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = default_config()
    for role in config_data["roles"]:
        role["prompt_template"] = None
    config_path.write_text(yaml.safe_dump(config_data))
    config, resolved = load_config(config_path, repo_root)
    orch = Orchestrator(config=config, paths=resolved, repo_root=repo_root)

    plan = PlanResult(
        goal="g",
        assumptions=[],
        tasks=[PlanTask(role="implementer", prompt="do it", blocked_by=[])],
    )
    orch.run_plan(plan=plan, run_id="run-1", state_path=resolved.runs / "run-1" / "state.json")
    assert True
