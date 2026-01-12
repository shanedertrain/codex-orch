from __future__ import annotations

import json
from pathlib import Path

from .models import RunState, TaskRecord


def load_state(path: Path) -> RunState:
    payload = json.loads(path.read_text())
    return RunState.from_dict(payload)


def save_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def upsert_task(state: RunState, task: TaskRecord) -> None:
    for idx, existing in enumerate(state.tasks):
        if existing.task_id == task.task_id:
            state.tasks[idx] = task
            return
    state.tasks.append(task)
