"""Compare sim-replayed pseudo proprio against real-robot ground truth for the
SAME episode (see the validation plan in resource/info/ego_cola_pseudo_proprio_handoff.md
session notes): real/cola_merged_cleaned_0530 episodes have both a real
sonic_tokens/*.npy (fed back through the sim replay pipeline via
prepare_real_validation_tokens.py + run_episode_pipeline.py) and a real
g1_proprio/*.npy ground truth, so this is a direct numeric check instead of
the self-check-only validation available for ego/cola.

Sim and real robot are different physical systems (mass/friction/controller
gains/contacts differ), so exact trajectory equality is NOT expected -- growing
RMSE / falling correlation over a long horizon is a normal sim2real gap, not
necessarily a pipeline bug. What a pipeline bug (wrong dim order/sign/frame
offset) looks like instead: a non-zero best-lag, or a dimension that stays
strongly anti-correlated/offset from the start.

Hand dims (29:31) are excluded from the "signal" verdict: both sides derive
hand state from the same recorded hand_binary action field (see
export_pseudo_proprio.py's correct_hand_state_from_action), so a match there
mostly confirms data plumbing, not sim replay accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kappa.data.schemas import PROPRIO_LAYOUT

DEFAULT_REAL_ROOT = "resource/dataset/proprietary/real/cola_merged_cleaned_0530"
# +-5 (100ms) was too narrow: several dims' true optimum sat at the window edge.
# +-20 (400ms) separates normal WBC-tracking latency/ZMQ jitter (lag stays small,
# r0 already high) from a real alignment bug (lag would blow past this too).
MAX_LAG = 20


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG) -> tuple[int, float]:
    """Slide b against a by lag in [-max_lag, max_lag]; return (lag, pearson r) at the
    lag with highest |r|. Positive lag means b is shifted forward (b[t] ~ a[t+lag])."""
    best = (0, float("nan"))
    best_abs = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > best_abs:
            best_abs = abs(r)
            best = (lag, r)
    return best


def compare(sim: np.ndarray, real: np.ndarray) -> dict:
    t = min(sim.shape[0], real.shape[0])
    sim, real = sim[:t], real[:t]

    groups = {}
    for name, lo, hi in PROPRIO_LAYOUT:
        dims = []
        for d in range(lo, hi):
            r = float(np.corrcoef(sim[:, d], real[:, d])[0, 1]) if np.std(sim[:, d]) > 1e-9 and np.std(real[:, d]) > 1e-9 else float("nan")
            rmse = float(np.sqrt(np.mean((sim[:, d] - real[:, d]) ** 2)))
            # r at lag=0 (r) vs best_lag_r: a small gap at a small lag is normal WBC-tracking
            # latency; a large gap that keeps growing toward the edge of MAX_LAG is the
            # signature of a real alignment/offset bug, not sim2real physical divergence.
            lag, lag_r = best_lag_corr(real[:, d], sim[:, d])
            dims.append({"dim": d, "pearson_r": r, "rmse": rmse, "best_lag": lag, "best_lag_r": lag_r})
        groups[name] = dims

    return {"num_frames_compared": t, "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-num", type=int, required=True)
    parser.add_argument("--sim-proprio", default=None, help="override sim g1_proprio path")
    parser.add_argument("--real-root", default=DEFAULT_REAL_ROOT)
    parser.add_argument("--sim-root", default="resource/tmp/real_validation_pseudo_proprio")
    args = parser.parse_args()

    ep_name = f"episode_{args.episode_num:06d}.npy"
    sim_path = Path(args.sim_proprio) if args.sim_proprio else Path(args.sim_root) / "g1_proprio" / ep_name
    real_path = Path(args.real_root) / "g1_proprio" / ep_name

    sim = np.load(sim_path)
    real = np.load(real_path)
    report = compare(sim, real)
    report["episode_num"] = args.episode_num
    report["sim_path"] = str(sim_path)
    report["real_path"] = str(real_path)

    print(json.dumps(report, indent=2))
    print("\n=== summary (excluding hand dims, see module docstring) ===")
    for name, dims in report["groups"].items():
        if name in ("left_hand", "right_hand"):
            print(f"{name}: SKIPPED in signal verdict (same-source hand action, not a sim-accuracy signal)")
            continue
        rs = [d["pearson_r"] for d in dims if not np.isnan(d["pearson_r"])]
        lags = [d["best_lag"] for d in dims]
        print(f"{name}: mean_r={np.mean(rs):.3f} rmse={[round(d['rmse'], 4) for d in dims]} best_lags={lags}")


if __name__ == "__main__":
    main()
