from pathlib import Path

import yaml

from codex_orch.config import default_config, load_config
from codex_orch.models import TaskRecord
from codex_orch.router import Orchestrator


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
