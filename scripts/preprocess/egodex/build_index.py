"""Build EgoDex episode index ahead of training.

Walks the repacked EgoDex tree, probes each ``token_state.npy`` shape, reads
``llm_description.txt``, and caches the result to
``<cache_root>/meta/egodex_index.json``. Run this once per data root before
launching multi-rank training so workers don't race on the build.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EgoDex dataset index")
    parser.add_argument(
        "--data-root",
        default="resource/dataset/EgoDex/egodex_repacked_simple",
        help="Path to the EgoDex `episodes/` parent directory.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Where to drop meta/egodex_index.json. Defaults to <data-root> if "
             "writable, otherwise <data-root>-cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Threads for shape/description probing.",
    )
    args = parser.parse_args()

    from src.data import EgoDexDataset

    ds = EgoDexDataset(
        data_root=args.data_root,
        cache_root=args.cache_root,
        index_workers=args.workers,
    )
    print(f"Indexed {len(ds._ep_paths)} episodes; total token rows = {len(ds)}.")
    print(f"Index written to {ds._index_path()}")


if __name__ == "__main__":
    main()
