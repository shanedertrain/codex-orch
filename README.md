# codex-orch

Local orchestrator that drives multiple Codex workers (navigator/implementer/tester/reviewer) with isolated git worktrees, JSONL capture, and schema-validated outputs.

## Setup
```
cd codex_orch
pip install -e .[dev]
```

Initialize a repo with default config/schemas/prompts:
```
codex-orch init
```

## Usage
- `codex-orch run --goal "Add feature X"` — run navigator then execute planned tasks.
- `codex-orch task --role implementer "Fix bug"` — run a single role.
- `codex-orch resume --run-id run-...` — re-run pending/failed tasks.
- `codex-orch report --run-id run-...` — write a markdown summary.
- `codex-orch clean --run-id run-...` — remove worktrees (keep branches with `--keep-branches`).
- `codex-orch merge --run-id run-... --allow-automerge` — fast-forward merge task branches.

Config lives at `.orchestrator/orchestrator.yaml` (YAML). Templates live under `.orchestrator/schemas` and `.orchestrator/prompts`. Decisions are appended to `docs/ai/decisions.md` in the host repo.

Concurrency heuristics support allow/ignore globs for path overlap checks (config keys `concurrency.allow` / `concurrency.ignore`), though the default run is sequential.

## Notes
- Workers are spawned with `codex exec --json` and validated against the configured schemas.
- Each task runs in its own git worktree/branch using the pattern `orch/{date}/run-{id}/{task-role}`.
- A retry is attempted when schema validation fails, with an explicit reminder to conform to the schema.
