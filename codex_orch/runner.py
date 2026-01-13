from __future__ import annotations

import json
import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .jsonl import JsonlCapture
from .models import CodexConfig, RoleConfig


@dataclass
class CodexExecutionResult:
    exit_code: int
    output: dict[str, Any] | None
    output_path: Path | None
    jsonl_log: Path
    raw_last_message: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    rate_limit_hits: int = 0


def build_codex_command(
    role: RoleConfig,
    worktree_path: Path,
    schema_path: Path | None,
    output_path: Path,
    prompt: str,
    codex: CodexConfig,
) -> list[str]:
    cmd: list[str] = [
        "codex",
        "exec",
        "--json",
        "--cd",
        str(worktree_path),
        "--sandbox",
        role.sandbox,
    ]
    if role.profile:
        cmd += ["--profile", role.profile]
    if role.full_auto:
        cmd.append("--full-auto")
    model = role.model or codex.model
    if model:
        cmd += ["--model", model]
    if schema_path:
        cmd += ["--output-schema", str(schema_path)]
    cmd += ["-o", str(output_path)]
    cmd += codex.extra_flags
    cmd.append(prompt)
    return cmd


def run_codex_process(
    cmd: list[str],
    jsonl_log_path: Path,
    output_path: Path,
    env: dict[str, str] | None = None,
) -> CodexExecutionResult:
    jsonl_capture = JsonlCapture(jsonl_log_path)
    output_data: dict[str, Any] | None = None
    raw_last_message: str | None = None

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env or os.environ.copy(),
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            raw_last_message = stripped
            jsonl_capture.consume_line(stripped)
        proc.wait()
        jsonl_capture.flush()
        jsonl_capture.close()
        if output_path.exists():
            try:
                output_data = json.loads(output_path.read_text())
                # Re-write as pretty JSON for readability of final.json.
                output_path.write_text(
                    json.dumps(output_data, indent=2, ensure_ascii=False)
                )
            except json.JSONDecodeError:
                output_data = None
        return CodexExecutionResult(
            exit_code=proc.returncode or 0,
            output=output_data,
            output_path=output_path if output_path.exists() else None,
            jsonl_log=jsonl_log_path,
            raw_last_message=raw_last_message,
            input_tokens=jsonl_capture.input_tokens,
            output_tokens=jsonl_capture.output_tokens,
            rate_limit_hits=jsonl_capture.rate_limit_hits,
        )


def load_config_and_paths(config_path: Path, repo_root: Path):
    return load_config(config_path, repo_root)
