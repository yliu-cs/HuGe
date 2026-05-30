"""Build a JSON index for proprietary ego token captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="resource/dataset/proprietary/ego/2026-5-23")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.data_root)
    rows: list[dict[str, object]] = []
    for token in sorted(root.rglob("*_latent_tokens_66d.npy")):
        arr = np.load(token, mmap_mode="r")
        desc = token.parent / "llm_description.txt"
        rows.append(
            {
                "episode_id": token.parent.relative_to(root).as_posix(),
                "token_path": str(token),
                "length": int(arr.shape[0]),
                "language": desc.read_text(encoding="utf-8").strip() if desc.exists() else "perform the task",
            }
        )
    if not rows:
        raise SystemExit(f"no *_latent_tokens_66d.npy files under {root}")
    out = Path(args.out) if args.out else root / "meta" / "prop_ego_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"episodes": rows}, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"indexed {len(rows)} episodes -> {out}")


if __name__ == "__main__":
    main()
