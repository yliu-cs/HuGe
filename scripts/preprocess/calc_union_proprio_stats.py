"""Compute strict grouped 34-D proprio stats over configured dataset roots."""

from __future__ import annotations

import argparse
from pathlib import Path

from huge.data.proprio_stats import write_proprio_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-cache-root", default="resource/dataset/humanoid-everyday-cache")
    parser.add_argument("--wbt-source-root", default="resource/dataset/G1_WBT")
    parser.add_argument("--prop-root", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530")
    parser.add_argument("--humanoid-arena-root", default=None, help="e.g. resource/dataset/HumanoidArena/HOI_pp_box")
    parser.add_argument("--out", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json")
    args = parser.parse_args()
    sources = {
        "HE": Path(args.he_cache_root) / "g1_proprio",
        "WBT": Path(args.wbt_source_root) / "g1_proprio",
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
