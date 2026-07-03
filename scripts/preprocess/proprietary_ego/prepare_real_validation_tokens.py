"""One-off adapter: mirror a real-robot episode's 66d sonic token into the
ego/cola on-disk layout (`<cola_root>/<episode_num>/*_latent_tokens_64d.npy`
and `_66d.npy`) so the existing `run_episode_pipeline.py`/`export_pseudo_proprio.py`
can be reused unmodified to sim-replay real-robot tokens for validation against
their own ground-truth g1_proprio (see resource/info/ego_cola_pseudo_proprio_handoff.md
and the validation plan in this session).

Real dataset's sonic_tokens/episode_NNNNNN.npy is already [T,66]
(motion_token[:64] + hand_binary[64:66]); the 64d file is just a slice, not a
re-encoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_REAL_ROOT = "resource/dataset/proprietary/real/cola_merged_cleaned_0530"
DEFAULT_OUT_ROOT = "resource/tmp/real_validation_cola_root"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-num", type=int, required=True)
    parser.add_argument("--real-root", default=DEFAULT_REAL_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    src = Path(args.real_root) / "sonic_tokens" / f"episode_{args.episode_num:06d}.npy"
    tok66 = np.load(src)
    if tok66.shape[-1] != 66:
        raise ValueError(f"expected 66d token at {src}, got shape {tok66.shape}")

    ep_dir = Path(args.out_root) / str(args.episode_num)
    ep_dir.mkdir(parents=True, exist_ok=True)
    stem = f"real_episode_{args.episode_num:06d}"
    np.save(ep_dir / f"{stem}_latent_tokens_64d.npy", tok66[:, :64].astype(np.float32))
    np.save(ep_dir / f"{stem}_latent_tokens_66d.npy", tok66.astype(np.float32))
    print(f"wrote {ep_dir}/{stem}_latent_tokens_{{64,66}}d.npy from {src} (shape {tok66.shape})")


if __name__ == "__main__":
    main()
