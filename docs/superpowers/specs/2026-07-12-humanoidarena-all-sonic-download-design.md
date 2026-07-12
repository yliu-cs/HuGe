# HumanoidArena All-Task SONIC Download Design

## Goal

Change `scripts/download_humanoidarena_vision_navi.py` so it downloads the
`sonic` data directory from every task in `Twang2026/HumanoidArena_raw`, while
excluding non-SONIC task content.

## Scope

- Keep the existing script path and detached tmux launcher interface.
- Match both `sonic` and `SONIC` directory spellings immediately below any
  top-level task directory.
- Automatically include future top-level tasks that follow the same layout.
- Do not download or modify the dataset during implementation or testing.

## Download Behavior

Pass these ModelScope allow patterns to `dataset_snapshot_download`:

```python
["*/sonic/**", "*/SONIC/**"]
```

The patterns preserve each task's directory hierarchy under
`resource/dataset/HumanoidArena_raw`. ModelScope's installed downloader uses
`fnmatch` matching, for which `*` can match each top-level task name.

Update user-facing names and messages from the single
`HSI_vision_navi` task to all-task SONIC semantics. This includes the module
description, worker function, tmux session name, log filename, and completion
summary. The script filename remains unchanged so existing callers do not
break.

## Validation

After the download call, inspect immediate children of the raw dataset root for
`sonic` or `SONIC` directories. Validation must:

1. Fail if no matching task directories exist.
2. Count `.npz` and `.mp4` files recursively for every matching directory.
3. Fail if any discovered task SONIC directory contains no `.npz` files.
4. Return and log aggregate task, NPZ, and MP4 counts.

Existing files outside matching SONIC directories do not influence validation.

## Error Handling

Keep the existing launcher and worker exception boundary: failures produce an
`error:` message on stderr and exit with status 1. ModelScope errors and local
validation errors remain visible in the tmux log.

## Testing

Add focused unit tests that import the script without requiring ModelScope
network access. Tests will cover:

- the downloader receives both all-task SONIC allow patterns;
- validation aggregates multiple tasks and both supported directory spellings;
- validation rejects an empty raw root;
- validation rejects a discovered task SONIC directory without NPZ files;
- the tmux command uses the updated session and log semantics.

Tests use temporary directories and an injected download callable, so they do
not download dataset files.
