"""Build a lightweight EgoDex token index for audit/debug."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="resource/dataset/EgoDex/egodex_repacked_simple")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.data_root)
    rows: list[dict[str, object]] = []
    for token in sorted(root.rglob("*.npy")):
        arr = np.load(token, mmap_mode="r")
        if arr.ndim >= 2 and arr.shape[-1] in {64, 66}:
            rows.append({"token_path": str(token), "length": int(arr.shape[0]), "dim": int(arr.shape[-1])})
    if not rows:
        raise SystemExit(f"no EgoDex token npy files under {root}")
    out = Path(args.out) if args.out else root / "meta" / "egodex_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"episodes": rows}, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"indexed {len(rows)} token files -> {out}")


if __name__ == "__main__":
    main()
