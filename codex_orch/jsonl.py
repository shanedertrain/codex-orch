from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JsonlCapture:
    log_path: Path
    events: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def consume_line(self, line: str) -> None:
        self._handle.write(line)
        if not line.endswith("\n"):
            self._handle.write("\n")
        try:
            parsed = json.loads(line)
            self.events.append(parsed)
        except json.JSONDecodeError:
            self.events.append({"raw": line})

    def flush(self) -> None:
        self._handle.flush()
