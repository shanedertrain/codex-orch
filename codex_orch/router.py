from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

from .config import load_config
from .models import (
    CodexInvocation,
    PlanResult,
    OrchestratorConfig,
    ResolvedPaths,
    RunState,
    TaskRecord,
    TaskResult,
    TaskStatus,
)
from .runner import build_codex_command, run_codex_process
from .state import save_state, upsert_task
from .worktree import WorktreeSpec, create_worktree


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def render_template(path: Path, **kwargs: Any) -> str:
    content = path.read_text()
    try:
        return content.format(**kwargs)
    except KeyError:
        # If format placeholders are missing, fallback to raw content
        return content


def task_dir(paths: ResolvedPaths, run_id: str, task_id: str) -> Path:
    return paths.runs / run_id / "tasks" / task_id


class Orchestrator:
    def __init__(
        self, config: OrchestratorConfig, paths: ResolvedPaths, repo_root: Path
    ):
        self.config = config
        self.paths = paths
        self.repo_root = repo_root
        self.goal: str | None = None
        self.spec_file: Path | None = None
        self.spec_text: str | None = None

    def _branch_name(self, run_id: str, task: TaskRecord) -> str:
        today = datetime.now(UTC).date().isoformat()
        return self.config.git.branch_template.format(
            date=today,
            run_id=run_id,
            task_id=task.task_id,
            role=task.role,
        )

    def shared_workspace_spec(self, run_id: str) -> WorktreeSpec:
        """Build a shared worktree spec for single-workspace runs."""
        today = datetime.now(UTC).date().isoformat()
        branch = self.config.git.branch_template.format(
            date=today,
            run_id=run_id,
            task_id="workspace",
            role="shared",
        )
        path = self.paths.worktrees / run_id
        return WorktreeSpec(path=path, branch=branch, base_ref=self.config.git.base_ref)

    def _prepare_worktree(self, run_id: str, task: TaskRecord) -> WorktreeSpec:
        if self.config.use_single_workspace:
            spec = self.shared_workspace_spec(run_id)
            worktree_path = task.worktree_path or spec.path
            branch = task.branch or spec.branch
            if worktree_path == spec.path and not worktree_path.exists():
                create_worktree(spec, self.repo_root)
        else:
            worktree_path = self.paths.worktrees / f"{task.task_id}-{task.role}"
            branch = self._branch_name(run_id, task)
            spec = WorktreeSpec(
                path=worktree_path, branch=branch, base_ref=self.config.git.base_ref
            )
            if not worktree_path.exists():
                create_worktree(spec, self.repo_root)
        task.worktree_path = worktree_path
        task.branch = branch
        return spec

    def _append_decisions(self, run_id: str, task: TaskRecord) -> None:
        # Only TaskResult (non-navigator) carries decisions; navigator PlanResult does not.
        if not task.result or not hasattr(task.result, "decisions"):
            return
        if not task.result.decisions:  # type: ignore[attr-defined]
            return
        log_path = self.paths.decisions
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# Run {run_id} — {task.task_id} ({task.role})\n")
            for decision in task.result.decisions:  # type: ignore[attr-defined]
                handle.write(f"- {decision}\n")
            handle.write("\n")

    def _validate_output(
        self, output: dict[str, Any] | None, schema_path: Path | None
    ) -> None:
        if output is None or schema_path is None:
            return
        schema = load_schema(schema_path)
        jsonschema.validate(output, schema)

    def _run_single_task(self, run_id: str, task: TaskRecord) -> TaskRecord:
        role_config = self.config.role_for(task.role)
        if not role_config:
            alias = self.config.role_aliases.get(task.role)
            if alias:
                task.role = alias
                role_config = self.config.role_for(alias)
        if not role_config:
            task.status = TaskStatus.FAILED
            task.error = f"Role not found in config: {task.role}"
            return task

        spec = self._prepare_worktree(run_id, task)
        paths = task_dir(self.paths, run_id, task.task_id)
        paths.mkdir(parents=True, exist_ok=True)
        output_path = paths / "final.json"
        jsonl_path = paths / "events.jsonl"

        schema_path = role_config.output_schema
        prompt_template = role_config.prompt_template
        rendered_prompt = task.prompt
        if prompt_template:
            rendered_prompt = render_template(
                prompt_template,
                goal=self.goal or task.prompt,
                prompt=task.prompt,
            )
        if self.spec_text:
            rendered_prompt = f"{rendered_prompt}\n\nSpecification:\n{self.spec_text}"

        cmd = build_codex_command(
            role=role_config,
            worktree_path=spec.path,
            schema_path=schema_path,
            output_path=output_path,
            prompt=rendered_prompt,
            codex=self.config.codex,
        )

        execution = run_codex_process(
            cmd, jsonl_log_path=jsonl_path, output_path=output_path
        )
        task.attempts += 1
        task.codex = task.codex
        task.status = (
            TaskStatus.COMPLETED if execution.exit_code == 0 else TaskStatus.FAILED
        )
        task.error = (
            None
            if task.status == TaskStatus.COMPLETED
            else "Codex exited with non-zero status"
        )

        if execution.output is not None:
            try:
                self._validate_output(execution.output, schema_path)
            except jsonschema.ValidationError as exc:
                task.status = TaskStatus.NEEDS_RETRY
                task.error = f"Schema validation failed: {exc.message}"
            else:
                if task.role == "navigator":
                    task.result = PlanResult.from_dict(execution.output)
                else:
                    task.result = TaskResult.from_dict(execution.output)
        else:
            task.status = TaskStatus.NEEDS_RETRY
            task.error = "No structured output produced"

        task.codex = CodexInvocation(
            cmd=cmd,
            exit_code=execution.exit_code,
            output_last_message_path=execution.output_path,
            jsonl_log_path=execution.jsonl_log,
        )
        return task

    def run_task(
        self, run_id: str, state_path: Path, state: RunState, task: TaskRecord
    ) -> TaskRecord:
        retry_limit = self.config.limits.retry_limit
        updated = self._run_single_task(run_id, task)
        while (
            updated.status == TaskStatus.NEEDS_RETRY and updated.attempts <= retry_limit
        ):
            updated.prompt = (
                f"{updated.prompt}\n\nOutput MUST conform to the requested JSON schema."
            )
            updated = self._run_single_task(run_id, updated)

        if updated.status == TaskStatus.COMPLETED and updated.result:
            self._append_decisions(run_id, updated)
        upsert_task(state, updated)
        save_state(state_path, state)
        return updated

    def run(self, goal: str, run_id: str, state_path: Path) -> RunState:
        self.goal = goal
        self.paths.base.mkdir(parents=True, exist_ok=True)
        self.paths.worktrees.mkdir(parents=True, exist_ok=True)
        self.paths.runs.mkdir(parents=True, exist_ok=True)

        run_state = RunState(
            run_id=run_id, goal=goal, spec_file=self.spec_file, spec_text=self.spec_text
        )
        save_state(state_path, run_state)
        navigator_role = self.config.role_for("navigator")
        if not navigator_role:
            raise ValueError("Navigator role missing in config")

        navigator_prompt = goal
        if navigator_role.prompt_template:
            navigator_prompt = render_template(
                navigator_role.prompt_template, goal=goal, prompt=goal
            )

        navigator_task = TaskRecord(
            task_id="T0001", role="navigator", prompt=navigator_prompt
        )
        nav_result = self.run_task(run_id, state_path, run_state, navigator_task)
        if nav_result.status != TaskStatus.COMPLETED or not nav_result.result:
            return run_state

        if not isinstance(nav_result.result, PlanResult):
            return run_state

        plan_payload = nav_result.result
        tasks_data = plan_payload.tasks
        queue: list[TaskRecord] = []
        next_index = 2
        for item in tasks_data:
            task_id = f"T{next_index:04d}"
            queue.append(
                TaskRecord(task_id=task_id, role=item.role, prompt=item.prompt)
            )
            next_index += 1

        limit = self.config.limits.max_tasks
        iterations = 0
        while (
            queue
            and len(run_state.tasks) < limit
            and iterations < self.config.limits.max_iterations
        ):
            current = queue.pop(0)
            updated = self.run_task(run_id, state_path, run_state, current)
            if updated.result and updated.result.next_tasks:
                for nxt in updated.result.next_tasks:
                    task_id = f"T{next_index:04d}"
                    queue.append(
                        TaskRecord(task_id=task_id, role=nxt.role, prompt=nxt.prompt)
                    )
                    next_index += 1
            iterations += 1
        return run_state


def load_orchestrator(
    config_path: Path, repo_root: Path
) -> tuple[Orchestrator, ResolvedPaths]:
    config, paths = load_config(config_path, repo_root)
    orchestrator = Orchestrator(config=config, paths=paths, repo_root=repo_root)
    return orchestrator, paths
