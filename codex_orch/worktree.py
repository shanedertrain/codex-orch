from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


@dataclass
class WorktreeSpec:
    path: Path
    branch: str
    base_ref: str


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def ensure_repo(cwd: Path) -> None:
    result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    if result.returncode != 0:
        raise WorktreeError(f"Not inside a git repository: {cwd}")


def current_branch(cwd: Path) -> str:
    result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if result.returncode != 0:
        raise WorktreeError(
            result.stderr.strip() or "Failed to determine current branch"
        )
    return result.stdout.strip()


def create_worktree(spec: WorktreeSpec, repo_root: Path) -> None:
    ensure_repo(repo_root)
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["worktree", "prune"], cwd=repo_root)

    def _add_worktree() -> subprocess.CompletedProcess:
        return _run_git(
            ["worktree", "add", "-B", spec.branch, str(spec.path), spec.base_ref],
            cwd=repo_root,
        )

    add_proc = _add_worktree()
    if add_proc.returncode != 0:
        # Retry once after pruning stale worktrees/refs.
        _run_git(["worktree", "prune"], cwd=repo_root)
        add_proc = _add_worktree()
    if add_proc.returncode != 0:
        raise WorktreeError(add_proc.stderr.strip() or "Failed to create worktree")
    # Prefer local submodule objects (no remote fetch) so local-only commits work.
    init_proc = _run_git(
        ["submodule", "update", "--init", "--recursive", "--no-fetch"],
        cwd=spec.path,
    )
    if init_proc.returncode != 0:
        init_proc = _run_git(
            ["submodule", "update", "--init", "--recursive"],
            cwd=spec.path,
        )
        if init_proc.returncode != 0:
            raise WorktreeError(
                init_proc.stderr.strip() or "Failed to initialize submodules"
            )


def remove_worktree(
    path: Path, repo_root: Path, keep_branch: bool = False, branch: str | None = None
) -> None:
    ensure_repo(repo_root)
    _run_git(["worktree", "remove", "--force", str(path)], repo_root)
    if keep_branch or not branch:
        return
    _run_git(["branch", "-D", branch], repo_root)
