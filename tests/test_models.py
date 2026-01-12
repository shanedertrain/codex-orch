from codex_orch.models import PlanResult, TaskResult


def test_plan_result_parsing() -> None:
    payload = {
        "goal": "demo",
        "assumptions": ["a1"],
        "tasks": [{"role": "implementer", "prompt": "do it"}],
    }
    plan = PlanResult.from_dict(payload)
    assert plan.goal == "demo"
    assert plan.tasks[0].role == "implementer"


def test_task_result_parsing() -> None:
    payload = {
        "summary": "done",
        "decisions": ["d1"],
        "changes": [{"path": "file.txt", "intent": "edit"}],
        "commands_ran": ["echo hi"],
        "risks": ["r1"],
        "next_tasks": [{"role": "tester", "prompt": "test it"}],
    }
    result = TaskResult.from_dict(payload)
    assert result.summary == "done"
    assert result.changes[0].path == "file.txt"
    assert result.next_tasks[0].role == "tester"
