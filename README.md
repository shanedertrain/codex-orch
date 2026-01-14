# codex-orch

Local orchestrator that drives multiple Codex workers (navigator/implementer/tester/reviewer) with a shared git worktree per run (or per-task worktrees if disabled), JSONL capture, and schema-validated outputs.

## Setup
- Install the Codex CLI separately (npm, not Poetry): `npm i -g @openai/codex` and verify `codex --version`. See `docs/ignore/codex-wsl-install.md` for WSL-specific steps.
- In this repo: `pip install -e .[dev]` (or `poetry install --with dev --sync` from the monorepo root, then use `poetry run codex-orch`).

Initialize a repo with default config/schemas/prompts:
```
poetry run codex-orch init    # or codex-orch init if installed globally
```

## Usage
- `codex-orch run --plan-file specs/feature_implementation_plan.json [--spec-file specs/feature.md]` — execute a finalized plan file (no navigator prompting).
- `codex-orch task --role implementer "Fix bug"` — run a single role prompt.
- codex-mem MCP server is started automatically if `scripts/start_codex_mem.sh` is present; logs under `.codex-mem/`.
- If `.orchestrator/` is missing, `codex-orch` will create default config and templates on first run.
- `codex-orch resume --run-id run-...` — re-run pending/failed tasks.
- `codex-orch report --run-id run-...` — write a markdown summary.
- `codex-orch clean --run-id run-...` — remove worktrees (keep branches with `--keep-branches`).
- `codex-orch prune [--older-than N] [--run-id ...] [--apply] [--no-resume]` — sweep runs, optionally resume pending tasks, and clean worktrees/run dirs (dry-run by default).
- `codex-orch merge --run-id run-... --allow-automerge` — fast-forward merge task branches.
- `--spec-file path/to/spec.md` — optional flag for run/task/resume/prune to inject a shared spec into worker prompts (validated, UTF-8 text only).

Config lives at `.orchestrator/orchestrator.yaml` (YAML). Templates live under `.orchestrator/schemas` and `.orchestrator/prompts`. Decisions are appended to `docs/ai/decisions.md` in the host repo.

Concurrency heuristics support allow/ignore globs for path overlap checks (config keys `concurrency.allow` / `concurrency.ignore`), though the default run is sequential.

## Notes
- Workers are spawned with `codex exec --json` and validated against the configured schemas.
- Default: all tasks in a run share a single worktree under `.orchestrator/worktrees/<run_id>` on branch `orch/{date}/run-{id}/workspace-shared`. Set `use_single_workspace: false` in the config to fall back to per-task worktrees using the branch template.
- A retry is attempted when schema validation fails, with an explicit reminder to conform to the schema.
- Memory/search (optional but recommended):
  - Start codex-mem MCP server: `poetry run codex-mem-serve` (or use `bash app/common/scripts/dev_memory.sh` in the monorepo to warm tldr and start the server).
  - Warm tldr summaries before runs: `poetry run tldr warm .`
  - Navigator uses `workspace-write` sandbox by default so it can create `.tldr/` artifacts and read mem context.
