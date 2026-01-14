from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def ensure_codex_mem_running(repo_root: Path, log_dir: Optional[Path] = None) -> None:
    """Best-effort start of codex-mem-serve using the shared script if available."""
    script = repo_root / "scripts" / "start_codex_mem.sh"
    if not script.exists():
        return
    env = None
    if log_dir:
        env = {"LOG_DIR_OVERRIDE": str(log_dir)}
    try:
        subprocess.run(
            [str(script)],
            cwd=repo_root,
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        # Ignore if script cannot be executed.
        return
