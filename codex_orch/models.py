from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import StrEnum
from pathlib import Path
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


@dataclass
class RoleConfig:
    name: str
    profile: str | None = None
    sandbox: str = "workspace-write"
    full_auto: bool = True
    output_schema: Path | None = None
    prompt_template: Path | None = None


@dataclass
class LimitsConfig:
    max_workers: int = 1
    max_tasks: int = 50
    max_iterations: int = 10
    retry_limit: int = 1


@dataclass
class PathsConfig:
    base_dir: Path = Path(".orchestrator")
    runs_dir: Path = Path("runs")
    worktrees_dir: Path = Path("worktrees")
    schemas_dir: Path = Path("schemas")
    prompts_dir: Path = Path("prompts")
    decisions_log: Path = Path("docs/ai/decisions.md")

    def resolve_in_repo(self, repo_root: Path) -> "ResolvedPaths":
        root = repo_root / self.base_dir
        return ResolvedPaths(
            base=root,
            runs=root / self.runs_dir,
            worktrees=root / self.worktrees_dir,
            schemas=root / self.schemas_dir,
            prompts=root / self.prompts_dir,
            decisions=repo_root / self.decisions_log,
        )


@dataclass
class ResolvedPaths:
    base: Path
    runs: Path
    worktrees: Path
    schemas: Path
    prompts: Path
    decisions: Path


@dataclass
class GitConfig:
    base_ref: str = "HEAD"
    branch_template: str = "orch/{date}/run-{run_id}/{task_id}-{role}"


@dataclass
class ConcurrencyConfig:
    allow: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)


@dataclass
class CodexConfig:
    model: str | None = None
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    roles: list[RoleConfig]
    limits: LimitsConfig
    paths: PathsConfig
    git: GitConfig
    codex: CodexConfig
    concurrency: ConcurrencyConfig
    use_single_workspace: bool = False

    def role_for(self, name: str) -> RoleConfig | None:
        return next((role for role in self.roles if role.name == name), None)


@dataclass
class TaskChange:
    path: str
    intent: str | None = None


@dataclass
class NextTask:
    role: str
    prompt: str


@dataclass
class TaskResult:
    summary: str | None = None
    decisions: list[str] = field(default_factory=list)
    changes: list[TaskChange] = field(default_factory=list)
    commands_ran: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_tasks: list[NextTask] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskResult":
        changes = [
            TaskChange(**c) for c in payload.get("changes", []) if isinstance(c, dict)
        ]
        next_tasks = [
            NextTask(**c) for c in payload.get("next_tasks", []) if isinstance(c, dict)
        ]
        return cls(
            summary=payload.get("summary"),
            decisions=list(payload.get("decisions", [])),
            changes=changes,
            commands_ran=list(payload.get("commands_ran", [])),
            risks=list(payload.get("risks", [])),
            next_tasks=next_tasks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "decisions": self.decisions,
            "changes": [asdict(c) for c in self.changes],
            "commands_ran": self.commands_ran,
            "risks": self.risks,
            "next_tasks": [asdict(n) for n in self.next_tasks],
        }


@dataclass
class CodexInvocation:
    cmd: list[str] = field(default_factory=list)
    exit_code: int | None = None
    output_last_message_path: Path | None = None
    jsonl_log_path: Path | None = None


@dataclass
class TaskRecord:
    task_id: str
    role: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    worktree_path: Path | None = None
    branch: str | None = None
    codex: CodexInvocation | None = None
    result: TaskResult | PlanResult | None = None
    error: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "role": self.role,
            "prompt": self.prompt,
            "status": self.status.value,
            "worktree_path": str(self.worktree_path) if self.worktree_path else None,
            "branch": self.branch,
            "attempts": self.attempts,
            "error": self.error,
        }
        if self.codex:
            data["codex"] = {
                "cmd": self.codex.cmd,
                "exit_code": self.codex.exit_code,
                "output_last_message_path": str(self.codex.output_last_message_path)
                if self.codex.output_last_message_path
                else None,
                "jsonl_log_path": str(self.codex.jsonl_log_path)
                if self.codex.jsonl_log_path
                else None,
            }
        if self.result:
            data["result"] = self.result.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRecord":
        codex_payload = payload.get("codex") or {}
        result_payload = payload.get("result")
        codex = None
        if codex_payload:
            codex = CodexInvocation(
                cmd=codex_payload.get("cmd", []),
                exit_code=codex_payload.get("exit_code"),
                output_last_message_path=Path(codex_payload["output_last_message_path"])
                if codex_payload.get("output_last_message_path")
                else None,
                jsonl_log_path=Path(codex_payload["jsonl_log_path"])
                if codex_payload.get("jsonl_log_path")
                else None,
            )
        result = None
        if isinstance(result_payload, dict):
            if "tasks" in result_payload and "decisions" not in result_payload:
                result = PlanResult.from_dict(result_payload)
            else:
                result = TaskResult.from_dict(result_payload)
        return cls(
            task_id=payload["task_id"],
            role=payload["role"],
            prompt=payload.get("prompt", ""),
            status=TaskStatus(payload.get("status", TaskStatus.PENDING.value)),
            worktree_path=Path(payload["worktree_path"])
            if payload.get("worktree_path")
            else None,
            branch=payload.get("branch"),
            codex=codex,
            result=result,
            error=payload.get("error"),
            attempts=payload.get("attempts", 0),
        )


@dataclass
class RunState:
    run_id: str
    tasks: list[TaskRecord] = field(default_factory=list)
    goal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunState":
        tasks = [TaskRecord.from_dict(t) for t in payload.get("tasks", [])]
        return cls(run_id=payload["run_id"], tasks=tasks, goal=payload.get("goal"))


@dataclass
class PlanTask:
    role: str
    prompt: str
    acceptance: str | None = None


@dataclass
class PlanResult:
    goal: str | None = None
    assumptions: list[str] = field(default_factory=list)
    tasks: list[PlanTask] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanResult":
        tasks_payload = payload.get("tasks") or []
        tasks = [PlanTask(**item) for item in tasks_payload if isinstance(item, dict)]
        return cls(
            goal=payload.get("goal"),
            assumptions=list(payload.get("assumptions", [])),
            tasks=tasks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "assumptions": self.assumptions,
            "tasks": [asdict(t) for t in self.tasks],
        }
