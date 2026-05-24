"""Build the proprietary-ego episode index ahead of training.

Walks one ``<data_root>/<task>/<ep>/`` capture day, probes each
``trackingData_<TS>_latent_tokens_66d.npy`` shape, reads
``llm_description.txt``, and caches the result to
``<cache_root>/meta/prop_ego_index.json``. Run this once per capture day
before launching multi-rank training so workers don't race on the build.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proprietary ego dataset index")
    parser.add_argument(
        "--data-root",
        default="resource/dataset/proprietary/ego/2026-5-23",
        help="Path to one capture-day root (parent of <task>/<ep>/...).",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help=(
            "Where to drop meta/prop_ego_index.json. Defaults to <data-root> "
            "if writable, otherwise <data-root>-cache."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Threads for shape/description probing.",
    )
    args = parser.parse_args()

    from src.data import PropEgoDataset

    ds = PropEgoDataset(
        data_root=args.data_root,
        cache_root=args.cache_root,
        index_workers=args.workers,
    )
    print(f"Indexed {len(ds._ep_paths)} episodes; total token rows = {len(ds)}.")
    print(f"Index written to {ds._index_path()}")


if __name__ == "__main__":
    main()
