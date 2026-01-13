from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    _role_order: tuple[str, ...] = ("navigator", "implementer", "reviewer", "tester")

    def __init__(
        self, config: OrchestratorConfig, paths: ResolvedPaths, repo_root: Path
    ):
        self.config = config
        self.paths = paths
        self.repo_root = repo_root
        self.goal: str | None = None
        self.spec_file: Path | None = None
        self.spec_text: str | None = None
        self._state_lock = threading.Lock()
        self._cached_allowed_roles: tuple[str, str] | None = None

    def _allowed_roles_text(self) -> tuple[str, str]:
        if self._cached_allowed_roles:
            return self._cached_allowed_roles
        role_names = sorted({role.name for role in self.config.roles})
        alias_pairs = [
            f"{alias}->{target}"
            for alias, target in sorted(self.config.role_aliases.items())
        ]
        allowed = ", ".join(role_names)
        aliases = ", ".join(alias_pairs)
        self._cached_allowed_roles = (allowed, aliases)
        return allowed, aliases

    def _infer_role_blockers(
        self,
        role: str,
        tasks: dict[str, TaskRecord],
        skip_task_id: str | None = None,
    ) -> list[str]:
        """Return blockers based on role ordering (navigator->implementer->reviewer->tester)."""
        if role not in self._role_order:
            return []
        idx = self._role_order.index(role)
        if idx == 0:
            return []
        prev_role = self._role_order[idx - 1]
        blockers: list[str] = []
        for task_id, task in tasks.items():
            if task_id == skip_task_id:
                continue
            if task.role == prev_role:
                blockers.append(task_id)
        return blockers

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

    def _record_prompt_stats(
        self, jsonl_path: Path, task: TaskRecord, prompt: str
    ) -> None:
        try:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            stats = {
                "type": "prompt.stats",
                "task_id": task.task_id,
                "role": task.role,
                "prompt_chars": len(prompt),
                "prompt_words": len(prompt.split()),
            }
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats) + "\n")
        except Exception:
            # Non-blocking best-effort logging; ignore failures.
            return

    def _warm_tldr(self) -> None:
        if not self.config.warm_tldr:
            return
        try:
            subprocess.run(
                ["poetry", "run", "tldr", "warm", "."],
                cwd=self.repo_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            # If poetry/tldr is missing, continue without blocking the run.
            return

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
                allowed_roles=self._allowed_roles_text()[0],
                role_aliases=self._allowed_roles_text()[1],
            )
        # Only attach the spec to navigator prompts to reduce token load.
        if self.spec_text and task.role == "navigator":
            rendered_prompt = f"{rendered_prompt}\n\nSpecification:\n{self.spec_text}"

        self._record_prompt_stats(jsonl_path, task, rendered_prompt)

        cmd = build_codex_command(
            role=role_config,
            worktree_path=spec.path,
            schema_path=schema_path,
            output_path=output_path,
            prompt=rendered_prompt,
            codex=self.config.codex,
        )

        env = None
        mem_src = self.repo_root / "app/common/codex-mem/src"
        if mem_src.exists():
            env = os.environ.copy()
            pythonpath_parts = [str(mem_src)]
            existing = env.get("PYTHONPATH")
            if existing:
                pythonpath_parts.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        execution = run_codex_process(
            cmd, jsonl_log_path=jsonl_path, output_path=output_path, env=env
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
        self,
        run_id: str,
        state_path: Path,
        state: RunState,
        task: TaskRecord,
        state_lock: threading.Lock | None = None,
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
            if state_lock:
                with state_lock:
                    self._append_decisions(run_id, updated)
            else:
                self._append_decisions(run_id, updated)
        if state_lock:
            with state_lock:
                upsert_task(state, updated)
                save_state(state_path, state)
        else:
            upsert_task(state, updated)
            save_state(state_path, state)
        return updated

    def run(self, goal: str, run_id: str, state_path: Path) -> RunState:
        self.goal = goal
        self.paths.base.mkdir(parents=True, exist_ok=True)
        self.paths.worktrees.mkdir(parents=True, exist_ok=True)
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        self._warm_tldr()

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
                navigator_role.prompt_template,
                goal=goal,
                prompt=goal,
                allowed_roles=self._allowed_roles_text()[0],
                role_aliases=self._allowed_roles_text()[1],
            )
        else:
            allowed_roles, aliases = self._allowed_roles_text()
            navigator_prompt = (
                f"{navigator_prompt}\n\nAllowed roles: {allowed_roles}."
            )
            if aliases:
                navigator_prompt += f" Aliases: {aliases}."

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

        # Validate roles before enqueuing tasks; fail fast if plan includes
        # roles outside the configured set (after alias resolution).
        invalid_roles: set[str] = set()
        for item in tasks_data:
            resolved = self.config.role_aliases.get(item.role, item.role)
            if not self.config.role_for(resolved):
                invalid_roles.add(item.role)
        if invalid_roles:
            nav_result.status = TaskStatus.FAILED
            nav_result.error = (
                "Plan contains invalid roles: "
                + ", ".join(sorted(invalid_roles))
                + f". Allowed: {', '.join(sorted({r.name for r in self.config.roles}))}."
            )
            upsert_task(run_state, nav_result)
            save_state(state_path, run_state)
            return run_state

        limit = self.config.limits.max_tasks
        plan_id_map: dict[str, str] = {}
        planned_tasks: list[tuple[str, TaskRecord]] = []
        next_index = 2
        for item in tasks_data:
            task_id = f"T{next_index:04d}"
            plan_key = item.id or task_id
            plan_id_map[plan_key] = task_id
            planned_tasks.append(
                (
                    task_id,
                    TaskRecord(
                        task_id=task_id,
                        role=item.role,
                        prompt=item.prompt,
                        blocked_by=item.blocked_by,
                    ),
                )
            )
            next_index += 1

        tasks_by_id: dict[str, TaskRecord] = {}
        for task_id, task in planned_tasks:
            mapped_blockers = [plan_id_map.get(dep, dep) for dep in task.blocked_by]
            task.blocked_by = mapped_blockers
            resolved_role = self.config.role_aliases.get(task.role, task.role)
            if not self.config.role_for(resolved_role):
                task.status = TaskStatus.FAILED
                task.error = f"Role not found in config: {resolved_role}"
                tasks_by_id[task_id] = task
                upsert_task(run_state, task)
                continue
            task.role = resolved_role
            tasks_by_id[task_id] = task
            upsert_task(run_state, task)
        # Apply default role-based blocking when none was provided.
        combined_for_blocking = {t.task_id: t for t in run_state.tasks}
        combined_for_blocking.update(tasks_by_id)
        for task_id, task in tasks_by_id.items():
            if task.blocked_by:
                continue
            inferred = self._infer_role_blockers(
                task.role, combined_for_blocking, skip_task_id=task_id
            )
            if inferred:
                task.blocked_by = inferred
                upsert_task(run_state, task)
        save_state(state_path, run_state)

        pending = {tid for tid, t in tasks_by_id.items() if t.status == TaskStatus.PENDING}
        running: dict[Any, str] = {}
        iterations = 0
        max_workers = max(1, self.config.limits.max_workers)

        def persist(task: TaskRecord) -> None:
            if self._state_lock:
                with self._state_lock:
                    upsert_task(run_state, task)
                    save_state(state_path, run_state)
            else:
                upsert_task(run_state, task)
                save_state(state_path, run_state)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while (
                pending
                and len(run_state.tasks) < limit
                and iterations < self.config.limits.max_iterations
            ):
                progress = False
                for task_id in list(pending):
                    if len(running) >= max_workers:
                        break
                    task = tasks_by_id[task_id]
                    tasks_lookup = {
                        **{t.task_id: t for t in run_state.tasks},
                        **tasks_by_id,
                    }
                    missing = [dep for dep in task.blocked_by if dep not in tasks_lookup]
                    if missing:
                        task.status = TaskStatus.FAILED
                        task.error = f"Missing dependencies: {', '.join(missing)}"
                        persist(task)
                        pending.remove(task_id)
                        progress = True
                        continue
                    dep_statuses = [tasks_lookup[dep].status for dep in task.blocked_by]
                    if any(
                        status in {TaskStatus.FAILED, TaskStatus.NEEDS_RETRY}
                        for status in dep_statuses
                    ):
                        task.status = TaskStatus.FAILED
                        failed_deps = [
                            dep
                            for dep in task.blocked_by
                            if tasks_lookup[dep].status != TaskStatus.COMPLETED
                        ]
                        task.error = (
                            "Blocked because dependencies failed: "
                            + ", ".join(failed_deps)
                        )
                        persist(task)
                        pending.remove(task_id)
                        progress = True
                        continue
                    if all(status == TaskStatus.COMPLETED for status in dep_statuses):
                        task.status = TaskStatus.RUNNING
                        persist(task)
                        future = executor.submit(
                            self.run_task,
                            run_id,
                            state_path,
                            run_state,
                            task,
                            self._state_lock,
                        )
                        running[future] = task_id
                        pending.remove(task_id)
                        progress = True
                if not running and not progress:
                    for task_id in list(pending):
                        task = tasks_by_id[task_id]
                        task.status = TaskStatus.FAILED
                        task.error = "Blocked by unmet dependencies"
                        persist(task)
                        pending.remove(task_id)
                    break

                if not running:
                    break

                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    finished_id = running.pop(future)
                    updated = future.result()
                    tasks_by_id[finished_id] = updated
                    if isinstance(updated.result, TaskResult) and updated.result.next_tasks:
                        for nxt in updated.result.next_tasks:
                            task_id = f"T{next_index:04d}"
                            new_task = TaskRecord(
                                task_id=task_id, role=nxt.role, prompt=nxt.prompt
                            )
                            inferred_blockers = [finished_id]
                            combined_for_blocking = {t_id: t for t_id, t in tasks_by_id.items()}
                            combined_for_blocking.update(
                                {t.task_id: t for t in run_state.tasks}
                            )
                            role_blockers = self._infer_role_blockers(
                                nxt.role, combined_for_blocking, skip_task_id=task_id
                            )
                            for blocker in role_blockers:
                                if blocker not in inferred_blockers:
                                    inferred_blockers.append(blocker)
                            new_task.blocked_by = inferred_blockers
                            tasks_by_id[task_id] = new_task
                            pending.add(task_id)
                            persist(new_task)
                            next_index += 1
                iterations += 1
        return run_state


def load_orchestrator(
    config_path: Path, repo_root: Path
) -> tuple[Orchestrator, ResolvedPaths]:
    config, paths = load_config(config_path, repo_root)
    orchestrator = Orchestrator(config=config, paths=paths, repo_root=repo_root)
    return orchestrator, paths
