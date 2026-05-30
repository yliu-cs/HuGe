"""Estimate per-side hand open/close thresholds from Humanoid-Everyday joints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="resource/dataset/humanoid-everyday")
    parser.add_argument("--out", default="resource/dataset/humanoid-everyday-cache/meta/hand_thresholds.json")
    args = parser.parse_args()
    root = Path(args.data_root)
    arrays: list[np.ndarray] = []
    for path in sorted(root.rglob("*hand*.npy")):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] >= 14:
            arrays.append(np.asarray(arr[:, -14:], dtype=np.float32))
    if not arrays:
        raise SystemExit("no hand joint npy files found; pass thresholds explicitly or run source-specific export first")
    flat = np.concatenate(arrays, axis=0)
    payload = {
        "left": np.median(flat[:, :7], axis=0).astype(float).tolist(),
        "right": np.median(flat[:, 7:14], axis=0).astype(float).tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
