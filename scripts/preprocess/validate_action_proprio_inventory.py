#!/usr/bin/env python3
"""Publish the fixed Stage 1+2+HumanoidArena action/proprio audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from huge.data.validation_manifest import build_validation_sources, ensure_validation_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="resource/datasets")
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data = Path(args.dataset_root)
    stats = Path(args.stats_path) if args.stats_path else data / "meta/stage123_humanoid_arena_proprio_stats_union_sonic.json"
    result = ensure_validation_manifest(
        data,
        sources=build_validation_sources(data),
        stage_scope="stage123_humanoid_arena",
        stats_path=stats,
        force=args.force,
    )
    print(
        f"validation manifest: {result['publication']} "
        f"audited={','.join(result['audited_tags']) or 'none'} "
        f"cached={','.join(result['cached_tags']) or 'none'}"
    )


if __name__ == "__main__":
    main()
