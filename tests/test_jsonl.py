from pathlib import Path

from codex_orch.jsonl import JsonlCapture


def test_jsonl_capture_handles_mixed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    capture = JsonlCapture(log_path)
    capture.consume_line('{"a": 1}')
    capture.consume_line("not-json")
    capture.consume_line('{"b": "x"}')
    capture.flush()
    capture.close()

    assert log_path.exists()
    assert len(capture.events) == 3
    assert capture.events[0]["a"] == 1
    assert "raw" in capture.events[1]
