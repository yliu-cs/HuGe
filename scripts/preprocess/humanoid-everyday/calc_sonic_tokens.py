"""Humanoid-Everyday SONIC cache entry point.

This rewrite keeps the script under the new package path. The current project
expects the materialized outputs under
``resource/dataset/humanoid-everyday-cache/{sonic_tokens,g1_proprio}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="resource/dataset/humanoid-everyday-cache")
    parser.add_argument("--require-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    cache = Path(args.cache_root)
    missing = [p for p in (cache / "sonic_tokens", cache / "g1_proprio") if not p.exists()]
    if args.require_existing and missing:
        raise SystemExit(
            "HE SONIC materialization requires the source-specific encoder export; "
            f"missing: {', '.join(str(p) for p in missing)}"
        )
    print(f"HE cache ready under {cache}")


if __name__ == "__main__":
    main()
