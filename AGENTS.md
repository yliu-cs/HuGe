# AGENTS.md

## Scope and precedence

This file applies to the repository root and every directory below it unless a
more specific `AGENTS.md` exists in a subdirectory. Direct user instructions
take precedence over this file.

Run all Huge commands from the repository root:

```bash
cd /inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Huge
```

Many model, data, config, virtual-environment, and output paths are relative to
the current working directory. Running an entry point elsewhere can select the
wrong environment or write to the wrong `resource/` or `.runs/` tree.

## Git policy — highest priority

Never invoke Git automatically. Unless the user explicitly requests a Git
operation in the current instruction:

- Do not run any Git command, including read-only commands such as `git status`,
  `git diff`, `git log`, `git show`, `git grep`, `git blame`, or
  `git rev-parse`.
- Do not use an IDE Git integration, Git MCP/tool, GitHub CLI, GitHub API, or
  another wrapper around Git.
- Do not create commits, branches, tags, worktrees, stashes, pull requests, or
  releases. Do not reset, restore, clean, rebase, merge, or amend anything.
- Inspect and verify work with ordinary file reads, `rg`, `find`, targeted
  tests, syntax checks, and direct output inspection.
- Assume the workspace may already contain user changes. Preserve unrelated
  files and do not overwrite or clean work that is outside the current task.

The user has explicitly granted one standing exception: as part of a training
job the user explicitly requested, the existing training code may internally
execute exactly `git rev-parse HEAD` to record `source_git_commit` in checkpoint
metadata. This read-only subprocess is allowed; it does not authorize the agent
to invoke that command directly or to run any other Git operation.

### Explicit commit and push instructions

Git authorization is narrow and applies only to the current instruction.

- A user instruction to **commit** authorizes only `git status --short`,
  task-path-scoped `git diff -- <paths>` and
  `git diff --cached -- <paths>`, `git add -- <paths>`, and creation of one
  commit. It does not authorize history browsing, branch operations, remote
  inspection, or a push.
- A user instruction to **push** authorizes checking the current branch and its
  configured upstream, followed by a normal push of the requested commits. It
  does not authorize creating another branch or upstream, force-pushing, or
  changing remote configuration.
- Stage explicit paths with `git add -- <path>...`; never use `git add -A`,
  `git add .`, or broad staging that may capture unrelated user work.
- Before committing, inspect the actual scoped diff and run relevant
  verification. If unrelated changes overlap the requested files, stop and ask
  the user rather than staging them silently.
- Never use `--amend`, `--force`, `--force-with-lease`, rebase, reset, restore,
  clean, or history rewriting unless the user explicitly names that operation.
- After an authorized commit or push, report the commit identifier, validation
  performed, and pushed remote/ref. Do not perform an additional Git operation
  merely to produce a report unless it is covered by the instruction.

Commit descriptions follow the repository's established style:

- Write the subject in English as an imperative sentence beginning with a verb
  such as `Add`, `Fix`, `Unify`, `Replace`, or `Propagate`.
- Do not add Conventional Commit prefixes such as `feat:` or `fix:` and do not
  end the subject with a period.
- Keep a focused change to a single subject line. For a broad change, add a
  blank line and two to five `- ` bullets describing behavior, architecture,
  important constraints, and user-visible impact rather than listing files.
- Generate the message from the actual scoped diff. Do not claim unimplemented
  behavior or validation that was not run.
- Do not add `Co-Authored-By` or other trailers automatically. Add one only
  when the user explicitly requests it and supplies or confirms the identity.

Example:

```text
Unify HumanoidArena preprocessing and stage3 training matrix

- Add atomic multi-task preprocessing and shared proprio stats
- Add task filtering and architecture/tuning-mode training variants
- Improve distributed progress logging and virtualenv cuDNN setup
```

## Repository artifact policy — high priority

Do not add planning artifacts or tests to the repository unless the user
explicitly requests those files in the current instruction.

- When the user explicitly requests an execution plan, design document,
  specification, or workflow record, write it as `resource/info/*.md`. Do not
  place such documents under `docs/superpowers/` or elsewhere in the project.
- Without such an explicit request, keep internal planning in the conversation
  or the plan tool and do not create planning artifacts in the repository. A
  skill or plugin that normally requests an in-repository plan does not
  override either the explicit-request requirement or the required
  `resource/info/*.md` location and format.
- Do not create new test files, test scripts, fixtures, snapshots, test data,
  or test directories, including a new `tests/` tree. A general request to
  implement, fix, refactor, or verify code does not implicitly authorize new
  tests.
- Existing tests may be inspected and run for verification. Do not modify,
  move, rename, or delete existing test assets unless the user explicitly puts
  them in scope.
- Put temporary verification scripts and scratch artifacts under `/tmp`, not
  inside the repository, and remove or leave them outside the project after
  use.

## Python environment

The root project uses Python 3.12, `uv`, `pyproject.toml`, `uv.lock`, and the
repository-local `.venv`. The package uses a `src/` layout and is installed
editable into that environment.

- Create or restore the locked environment with `uv sync --frozen`.
- For normal automation, call `.venv/bin/python` directly. Activation is not
  required, and direct invocation avoids an implicit environment sync.
- `uv run --frozen python ...` is acceptable when an explicit `uv` invocation
  is useful. Do not use an unqualified `uv run` if it could update the lock.
- Do not run core Huge code with bare `python`, `python3`, system `pip`, or a
  Conda interpreter. The download script's env shebang is not sufficient.
- For robust script, test, or tmux calls, use
  `PYTHONPATH=src .venv/bin/python ...`. The editable install normally makes
  `PYTHONPATH` unnecessary, but setting it documents the intended import root.
- Do not install packages manually into `.venv`. If the user explicitly asks
  for a dependency change, use `uv add`/`uv remove` so `pyproject.toml` and
  `uv.lock` stay consistent. Do not hand-edit `uv.lock` or run `uv lock` merely
  as a routine check.
- `GR00T-WholeBodyControl/gear_sonic` is a local editable dependency declared
  by `pyproject.toml`; do not move or remove that directory.

### Separate runtime environments

Do not mix these environments:

- Huge source, preprocessing, training, and inference server: root `.venv`
  (Python 3.12).
- WBC MuJoCo simulation: `GR00T-WholeBodyControl/.venv_sim/bin/python`
  (Python 3.10). The proprietary ego pipeline deliberately uses this only for
  simulation-side steps.
- WBC VLA inference client: `GR00T-WholeBodyControl/.venv_inference` (Python
  3.10), managed by that project's installer. Do not assume it exists.
- HumanoidArena Isaac Lab evaluation: use
  `scripts/eval/humanoidarena/run_eval.sh`; its `env.sh` and
  `runtime_paths.sh` resolve the dedicated Conda/Isaac Lab interpreters. Never
  replace them with the root `.venv`.

## Project map

- `src/huge/data/`: schemas, datasets, video reading, transforms, collators,
  samplers, normalization, stats, auditing, and HumanoidArena preprocessing.
- `src/huge/models/`: `QwenSonicFM` plus the `qwenpi_v3`, `qwengr00t`, and
  `qwenoft` action-head implementations.
- `src/huge/training/`: stage defaults, Accelerate/DeepSpeed runner, losses,
  checkpoint metadata, and checkpoint loading/saving.
- `src/huge/inference/`: checkpoint policy, observation assembly, ZMQ server,
  SONIC adapter, and deployment logging.
- `scripts/preprocess/`: per-source preprocessing and statistics entry points.
- `scripts/train/`: the supported single-stage launcher and experiment-matrix
  launcher.
- `scripts/eval/`: offline comparison, inference helpers, and HumanoidArena
  simulation evaluation.
- `configs/deepspeed/`: ZeRO-2 and ZeRO-3 configurations.
- `tests/`: standard-library `unittest` tests.
- `resource/`: local datasets, models, derived caches, and large generated
  resources; it is not source code.
- `.runs/`: training/evaluation outputs and background-task logs.
- `GR00T-WholeBodyControl/` and `HumanoidArena/`: local external/reference
  projects with their own runtime conventions. Do not modify them unless the
  task explicitly puts them in scope. If they are in scope, first read their
  applicable `AGENTS.md`, `CLAUDE.md`, and README files and follow the most
  specific environment and build instructions.

## Core data and model contracts

Huge's canonical action schema is 66-D:

- dimensions `0:64`: SONIC motion token;
- dimensions `64:66`: left/right binary hand state;
- hand polarity: closed is `1`, open is `0`;
- default action horizon: 50 steps;
- control frequency: 50 Hz.

The canonical proprioception schema is 34-D in this exact order:

1. left leg: 6;
2. right leg: 6;
3. waist: 3;
4. left arm: 7;
5. right arm: 7;
6. left hand: 1;
7. right hand: 1;
8. projected gravity: 3.

Do not locally change dimensions, ordering, hand polarity, horizon, or control
frequency in one dataset/model path. A schema change requires coordinated
updates and tests across `schemas.py`, preprocessing, datasets, collators,
normalizers, checkpoint metadata, training, inference, and evaluation.

`QwenSonicFM` requires a local Qwen3-VL model; the character-token fallback has
been removed. The default model path is
`resource/model/Qwen3-VL-2B-Instruct`, and SONIC assets live under
`resource/model/sonic`.

## Data-path rules

The codebase currently contains two different path conventions:

- The current HumanoidArena download, preprocessing, and matrix-training flow
  uses `resource/datasets/` (plural).
- Several legacy Stage 1/2, sim/real, audit, and other preprocessing defaults
  still use `resource/dataset/` (singular).
- The current workspace's data is under `resource/datasets/`; there is no
  equivalent `resource/dataset/` tree to assume automatically.

Therefore, do not globally replace `dataset` with `datasets`, do not assume the
paths are interchangeable, and do not trust an old README example over the
current entry-point code. Before running an old entry point, inspect its CLI,
shell variables, dataset builder, and actual resource locations; pass explicit
roots and stats paths when needed.

Current HumanoidArena defaults are:

- raw, read-only source: `resource/datasets/HumanoidArena_raw`;
- unified cache: `resource/datasets/HumanoidArena`;
- union stats:
  `resource/datasets/HumanoidArena/meta/proprio_stats_union_sonic.json`.

A published HumanoidArena cache should contain `_SUCCESS`,
`meta/conversion_manifest.json`, `meta/language_map.json`, union stats,
`sonic_tokens/`, `g1_proprio/`, and `videos/`. Some videos can be relative
symlinks back into the raw tree, so do not move, rewrite, or delete the raw
dataset after preprocessing.

The HumanoidArena preprocessor is a strict, atomic, all-task pipeline. It
preflights every episode, converts into a staging directory, validates the
result, and publishes only on success. Do not edit the raw tree or a published
cache by hand. `--rebuild` replaces an existing cache; use it only when the
user explicitly requests a rebuild.

## Supported entry points

From the repository root, use these invocation patterns.

HumanoidArena preprocessing:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/preprocess/proprietary/calc_sonic_tokens_from_humanoid_arena.py
```

The download script has side effects: running it without arguments starts a
detached tmux download, and its hidden worker performs network downloads. Use
`--help` for safe inspection; start it only when the user requests a download.

Training must go through the launcher:

```bash
bash scripts/train/train_huge.sh stage1
bash scripts/train/train_huge.sh stage2
bash scripts/train/train_huge.sh stage3
```

Do not run `src/huge/training/train_stage.py` directly. The launcher configures
Accelerate, DeepSpeed, process counts, CUDA/cuDNN paths, and distributed
settings. Stage 1 uses EgoDex; Stage 2 uses Humanoid-Everyday plus G1-WBT;
Stage 3 uses proprietary real/sim data or an explicitly supplied
HumanoidArena cache. Training has no validation loop.

`scripts/train/run_huge.sh` is a high-cost experiment-matrix launcher. Its
default HumanoidArena mode runs `HOI_pp_box` across three architectures and
three tuning modes (nine jobs), not a smoke test. Inspect it first with a
temporary dry run and never start the real matrix without an explicit request:

```bash
tmp_dir="$(mktemp -d)"
HUGE_DRY_RUN=1 \
HUGE_STAGE3_DATA=humanoid_arena \
HUGE_HUMANOID_ARENA_ARCHITECTURES=qwenpi_v3 \
HUGE_HUMANOID_ARENA_TUNING_MODES=full \
HUGE_LOCK_FILE="$tmp_dir/run.lock" \
HUGE_RUN_ROOT="$tmp_dir/runs" \
bash scripts/train/run_huge.sh
```

The inference server uses the root environment but binds ports and loads a
large local model, so start it only on request:

```bash
PYTHONPATH=src .venv/bin/python src/huge/inference/run_huge_server.py --help
```

HumanoidArena full simulation evaluation must use its environment-aware shell
launcher:

```bash
bash scripts/eval/humanoidarena/run_eval.sh <task> <checkpoint_dir> [backend]
```

## Editing conventions

- Search with `rg`/`rg --files` and inspect the narrowest relevant files first.
- Make focused changes and preserve unrelated user work. Avoid broad rewrites
  and repository-wide formatting.
- Use `apply_patch` for hand-authored edits.
- Follow existing Python style: `from __future__ import annotations`, type
  annotations, `pathlib.Path`, focused helpers, `argparse`, `main()`, and an
  `if __name__ == "__main__"` guard where appropriate.
- Follow existing shell style: `#!/usr/bin/env bash`, `set -euo pipefail`,
  quoted variables, arrays for commands, and `BASH_SOURCE`-based path
  resolution.
- Preserve atomic-output patterns for datasets and stats. When the user
  explicitly requests test work, tests must use temporary directories, mocks,
  or dependency injection and must not touch real datasets, caches,
  checkpoints, tmux sessions, networks, GPUs, or ports.
- Do not introduce unrelated dependencies, refactors, compatibility layers,
  or generated artifacts.

## Tests and verification

The existing repository tests use `unittest`. The project does not define a
project-wide pytest, Ruff, Black, isort, Mypy, Pyright, ShellCheck, or shfmt
configuration, so do not invent or claim those checks.

When verification is appropriate, run existing focused tests first, then the
existing full suite when appropriate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Example focused discovery:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests \
  -p 'test_humanoid_arena_preprocess_*.py' -v
```

Send Python bytecode outside the repository for syntax verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/huge-agent-pycache \
.venv/bin/python -m compileall -q src scripts tests
```

Check shell syntax without executing scripts:

```bash
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

The data audit writes a report by default and still inspects several legacy
singular paths. When specifically needed, keep its generated report outside
the repository, and do not treat it as proof that the plural HumanoidArena
cache passed a complete audit:

```bash
PYTHONPATH=src .venv/bin/python src/huge/data/audit.py \
  --output /tmp/huge-data-audit.md
```

Do not claim a task is fixed or complete without fresh, risk-proportionate
verification and inspection of its exit code and output.

## Long-running and destructive operations

Downloads, full preprocessing, cache rebuilds, GPU training, inference
servers, and Isaac Sim evaluation are costly and/or stateful. Start them only
when explicitly requested.

- Before starting, verify paths, expected outputs, available resources, and
  whether an older process or tmux session is already running.
- Use a meaningful, unique tmux session name for background work.
- Put logs under `.runs/<category>/`; create the log's parent directory before
  using `tee` because `tee` does not create directories.
- Use `set -o pipefail` in a logged pipeline so Python failures propagate.
- A tmux command must use an inner `bash -lc`, explicitly `cd` to the absolute
  repository root, and invoke the correct environment by absolute or
  repository-relative path.
- After launch, check the session and initial log, then report the session
  name, log path, and actual output directory.
- Before restart, inspect the old process/session and output state to avoid
  duplicate writers.
- Never manually delete raw data, caches, checkpoints, logs, or pre-existing or
  user-owned staging directories and lock files. Tool-owned atomic cleanup and
  release of a lock created by the same invocation are allowed. Never add
  `--rebuild` as a convenience flag.
