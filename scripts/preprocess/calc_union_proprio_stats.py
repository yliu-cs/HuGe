"""Compute Stage 2 + Stage 3 grouped 34-D proprio statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from huge.data.proprio_stats import write_proprio_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-cache-root", default="resource/datasets/humanoid-everyday-cache")
    parser.add_argument("--wbt-source-root", default="resource/datasets/G1_WBT")
    parser.add_argument("--hiw-source-root", default="resource/datasets/HIW-500")
    parser.add_argument("--prop-root", default=None)
    parser.add_argument("--humanoid-arena-root", default="resource/datasets/HumanoidArena")
    parser.add_argument("--out", default="resource/datasets/HumanoidArena/meta/proprio_stats_union_sonic.json")
    args = parser.parse_args()
    sources = {
        "HE": Path(args.he_cache_root) / "g1_proprio",
        "WBT": Path(args.wbt_source_root) / "g1_proprio",
        "HIW": Path(args.hiw_source_root) / "g1_proprio",
    }
    if args.prop_root:
        sources["PROP"] = Path(args.prop_root) / "g1_proprio"
    if args.humanoid_arena_root:
        sources["HUMANOID_ARENA"] = Path(args.humanoid_arena_root) / "g1_proprio"
    out = Path(args.out)
    write_proprio_stats(sources, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
