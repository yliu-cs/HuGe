"""G1-WBT SONIC cache entry point for the new huge package layout."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="resource/dataset/G1_WBT")
    parser.add_argument("--require-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    missing = [p for p in (root / "sonic_tokens", root / "g1_proprio") if not p.exists()]
    if args.require_existing and missing:
        raise SystemExit(
            "WBT SONIC materialization requires the source-specific encoder export; "
            f"missing: {', '.join(str(p) for p in missing)}"
        )
    print(f"WBT cache ready under {root}")


if __name__ == "__main__":
    main()
