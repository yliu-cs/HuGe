"""Build HumanoidArena and HE+WBT+HIW+HA union stats."""

from __future__ import annotations

import argparse
from pathlib import Path

from huge.data.humanoid_arena_preprocess import preprocess_humanoid_arena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly preprocess every HumanoidArena task into one atomic cache. "
            "The raw tree is never modified, and any invalid episode aborts publication."
        )
    )
    parser.add_argument(
        "--raw-root",
        default="resource/datasets/HumanoidArena_raw",
        help="read-only raw root containing all task directories",
    )
    parser.add_argument(
        "--out-root",
        default="resource/datasets/HumanoidArena",
        help="unified Stage 3 cache destination",
    )
    parser.add_argument(
        "--he-cache-root",
        default="resource/datasets/humanoid-everyday-cache",
        help="Humanoid Everyday cache root used for union stats",
    )
    parser.add_argument(
        "--wbt-source-root",
        default="resource/datasets/G1_WBT",
        help="G1 WBT source/cache root used for union stats",
    )
    parser.add_argument(
        "--hiw-source-root",
        default="resource/datasets/HIW-500",
        help="HIW source root used for union stats",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="atomically replace an existing cache even if its source inventory changed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = preprocess_humanoid_arena(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        he_cache_root=Path(args.he_cache_root),
        wbt_source_root=Path(args.wbt_source_root),
        hiw_source_root=Path(args.hiw_source_root),
        rebuild=args.rebuild,
    )
    counts = manifest["counts"]
    print(
        f"HumanoidArena ready: {counts['episodes']} episodes / {counts['frames']} frames "
        f"at {args.out_root}"
    )
    print(f"union stats: {Path(args.out_root) / 'meta/proprio_stats_union_sonic.json'}")


if __name__ == "__main__":
    main()
