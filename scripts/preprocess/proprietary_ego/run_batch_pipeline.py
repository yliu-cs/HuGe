"""Small-batch runner over `run_episode_pipeline.py` -- validates the single-episode
orchestrator holds up across multiple episodes before committing to the full
400-episode cola set (see resource/info/ego_cola_pseudo_proprio_handoff.md
section 0 hard constraint #4: batch beyond one episode needs explicit
re-confirmation, which was given for this small validation batch only).

Runs episodes sequentially (each episode's sim+deploy is torn down in
run_episode_pipeline.py's `finally` block before the next one starts) and
keeps going on a per-episode failure so one bad episode doesn't kill the batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
ORCHESTRATOR = REPO_ROOT / "scripts" / "preprocess" / "proprietary_ego" / "run_episode_pipeline.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="episode numbers to run")
    parser.add_argument("--out-root", default="resource/dataset/proprietary/ego/cola_pseudo_proprio")
    args = parser.parse_args()

    results = []
    for ep in args.episodes:
        work_dir = REPO_ROOT / "resource" / "tmp" / "ego_pipeline" / f"ep_{ep}"
        print(f"\n=== episode {ep} ===", flush=True)
        proc = subprocess.run(
            [str(VENV_PYTHON), str(ORCHESTRATOR), "--episode-num", str(ep),
             "--out-root", args.out_root, "--work-dir", str(work_dir)],
            cwd=REPO_ROOT,
        )
        report_path = REPO_ROOT / args.out_root / "g1_proprio" / f"episode_{ep:06d}.report.json"
        report = json.loads(report_path.read_text()) if proc.returncode == 0 and report_path.exists() else None
        results.append({"episode": ep, "returncode": proc.returncode, "report": report})

    print("\n=== batch summary ===")
    ok = [r for r in results if r["returncode"] == 0]
    print(f"{len(ok)}/{len(results)} episodes succeeded")
    for r in results:
        if r["report"] is not None:
            rep = r["report"]
            print(f"  ep {r['episode']}: OK clip_frac={rep['clip_frac']:.3f} "
                  f"missing_tail_filled={rep['missing_tail_filled']} "
                  f"hand_corrected={rep['hand_frames_corrected']}/{rep['hand_frames_total']}")
        else:
            print(f"  ep {r['episode']}: FAILED (returncode={r['returncode']})")

    if len(ok) < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
