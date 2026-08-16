"""Compute exact cumulative 34-D proprio statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from huge.data.proprio_stats import EXACT_STAGE_TAGS, write_proprio_stats


def _source(value: str) -> tuple[str, Path]:
    tag, separator, root = value.partition("=")
    if not separator or not tag or not root:
        raise argparse.ArgumentTypeError("--source must be TAG=ROOT")
    return tag, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-scope", choices=sorted(EXACT_STAGE_TAGS), required=True)
    parser.add_argument("--source", action="append", type=_source, required=True, metavar="TAG=ROOT")
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    sources: dict[str, Path] = {}
    for tag, root in args.source:
        if tag in sources:
            parser.error(f"duplicate --source tag: {tag}")
        sources[tag] = root
    out = Path(args.out)
    write_proprio_stats(
        sources,
        out,
        show_progress=args.show_progress,
        scratch_dir=args.scratch_dir,
        stage_scope=args.stage_scope,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
