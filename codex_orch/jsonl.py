from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JsonlCapture:
    log_path: Path
    events: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    rate_limit_hits: int = 0

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
            usage = parsed.get("usage") if isinstance(parsed, dict) else None
            if isinstance(usage, dict):
                self.input_tokens += int(usage.get("input_tokens") or 0)
                self.output_tokens += int(usage.get("output_tokens") or 0)
            if isinstance(parsed, dict):
                text = parsed.get("message") or parsed.get("text") or ""
                if isinstance(text, str) and "Rate limit reached" in text:
                    self.rate_limit_hits += 1
        except json.JSONDecodeError:
            self.events.append({"raw": line})
            if "Rate limit reached" in line:
                self.rate_limit_hits += 1

    def flush(self) -> None:
        self._handle.flush()
