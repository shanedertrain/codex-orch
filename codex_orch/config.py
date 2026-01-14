from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .models import (
    CodexConfig,
    ConcurrencyConfig,
    GitConfig,
    LimitsConfig,
    OrchestratorConfig,
    PathsConfig,
    ResolvedPaths,
    RoleConfig,
)


def _path_from_config(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(
    config_path: Path, repo_root: Path
) -> tuple[OrchestratorConfig, ResolvedPaths]:
    raw = yaml.safe_load(config_path.read_text()) or {}

    paths = PathsConfig(**raw.get("paths", {}))
    resolved_paths = paths.resolve_in_repo(repo_root)

    raw_roles = raw.get("roles") or []
    roles: list[RoleConfig] = []
    for item in raw_roles:
        role_data = dict(item)
        role_data.pop("model", None)
        output_schema = _path_from_config(
            role_data.get("output_schema"), resolved_paths.schemas
        )
        prompt_template = _path_from_config(
            role_data.get("prompt_template"), resolved_paths.prompts
        )
        roles.append(
            RoleConfig(
                name=role_data["name"],
                profile=role_data.get("profile"),
                sandbox=role_data.get("sandbox", "workspace-write"),
                full_auto=bool(role_data.get("full_auto", True)),
                output_schema=output_schema,
                prompt_template=prompt_template,
            )
        )

    limits = LimitsConfig(**(raw.get("limits") or {}))
    git = GitConfig(**(raw.get("git") or {}))
    codex_raw = dict(raw.get("codex") or {})
    codex_raw.pop("model", None)
    codex = CodexConfig(**codex_raw)
    concurrency = ConcurrencyConfig(**(raw.get("concurrency") or {}))
    use_single_workspace = bool(raw.get("use_single_workspace", True))
    role_aliases = dict(raw.get("role_aliases") or {})
    warm_tldr = bool(raw.get("warm_tldr", True))

    config = OrchestratorConfig(
        roles=roles,
        limits=limits,
        paths=paths,
        git=git,
        codex=codex,
        concurrency=concurrency,
        use_single_workspace=use_single_workspace,
        role_aliases=role_aliases,
        warm_tldr=warm_tldr,
    )
    return config, resolved_paths


def default_config() -> dict[str, Any]:
    """Return a dict for the default YAML config."""
    paths_cfg = PathsConfig()
    return {
        "paths": {
            "base_dir": str(paths_cfg.base_dir),
            "runs_dir": str(paths_cfg.runs_dir),
            "worktrees_dir": str(paths_cfg.worktrees_dir),
            "schemas_dir": str(paths_cfg.schemas_dir),
            "prompts_dir": str(paths_cfg.prompts_dir),
            "decisions_log": str(paths_cfg.decisions_log),
        },
        "limits": asdict(LimitsConfig()),
        "git": asdict(GitConfig()),
        "codex": asdict(CodexConfig()),
        "concurrency": asdict(ConcurrencyConfig()),
        "use_single_workspace": True,
        "warm_tldr": True,
        "role_aliases": {
            "Implementer": "implementer",
            "developer": "implementer",
            "Documenter": "implementer",
            "coder": "implementer",
            "dev": "implementer",
            "self": "implementer",
            "user": "implementer",
        },
        "roles": [
            {
                "name": "navigator",
                "profile": "navigator",
                "sandbox": "workspace-write",
                "full_auto": False,
                "output_schema": "plan.schema.json",
                "prompt_template": "navigator.md",
            },
            {
                "name": "implementer",
                "profile": "implementer",
                "sandbox": "workspace-write",
                "full_auto": False,
                "output_schema": "task_result.schema.json",
                "prompt_template": "implementer.md",
            },
            {
                "name": "tester",
                "profile": "tester",
                "sandbox": "workspace-write",
                "full_auto": False,
                "output_schema": "task_result.schema.json",
                "prompt_template": "tester.md",
            },
            {
                "name": "reviewer",
                "profile": "reviewer",
                "sandbox": "read-only",
                "full_auto": False,
                "output_schema": "task_result.schema.json",
                "prompt_template": "reviewer.md",
            },
        ],
    }
