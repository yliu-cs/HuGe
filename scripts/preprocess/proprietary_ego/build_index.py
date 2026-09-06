"""Build a JSON index for proprietary ego token captures."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np

from huge.data.schemas import PROPRIO_LAYOUT_VERSION


def video_info(path: Path) -> dict[str, object]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = int(stream.frames) or sum(1 for _ in container.decode(stream))
        duration = float(stream.duration * stream.time_base) if stream.duration else frames / fps
        result = {"frames": frames, "fps": fps, "duration": duration,
                  "width": stream.width, "height": stream.height}
        if frames <= 0 or fps <= 0:
            raise ValueError(f"empty video or invalid FPS: {path}")
        # Each audit worker owns its container; the training reader cache is process-local.
        for index in (0, frames // 2, frames - 1):
            target = index / fps
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            decoded = None
            for frame in container.decode(stream):
                decoded = frame
                if frame.pts is not None and float(frame.pts * stream.time_base) >= target:
                    break
            if decoded is None:
                raise ValueError(f"could not decode frame {index}: {path}")
            decoded.to_ndarray(format="rgb24")
    return result


def build_paired_index(root: Path, pseudo: Path, syn: Path, out: Path) -> dict[str, object]:
    ego_root = out.parent.parent.resolve()
    raw_ids = {path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit()}
    syn_ids = {path.name for path in syn.iterdir() if path.is_dir() and path.name.isdigit()}
    expected = {str(i) for i in range(400)}
    if raw_ids != expected or syn_ids != expected:
        raise ValueError(f"expected paired episode IDs 0..399; raw difference={sorted(raw_ids ^ expected)}, syn difference={sorted(syn_ids ^ expected)}")
    files = {f"episode_{i:06d}.npy" for i in range(400)}
    for folder in ("sonic_tokens", "g1_proprio"):
        actual = {p.relative_to(pseudo / folder).as_posix() for p in (pseudo / folder).rglob("*.npy")}
        if actual != files:
            raise ValueError(f"{folder} inventory differs: {sorted(actual ^ files)}")

    def record(number: int) -> dict[str, object]:
        episode = root / str(number)
        tokens = list(episode.glob("trackingData_*_latent_tokens_66d.npy"))
        if len(tokens) != 1:
            raise ValueError(f"episode {number}: expected one original 66D token file")
        original = tokens[0]
        stem = original.name.removeprefix("trackingData_").removesuffix("_latent_tokens_66d.npy")
        token = pseudo / "sonic_tokens" / f"episode_{number:06d}.npy"
        proprio = pseudo / "g1_proprio" / token.name
        action = np.load(token, allow_pickle=False)
        state = np.load(proprio, allow_pickle=False)
        if action.ndim != 2 or action.shape[1] != 66 or len(action) == 0 or state.shape != (len(action), 34):
            raise ValueError(f"episode {number}: invalid action/proprio shapes {action.shape}/{state.shape}")
        if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (action, state)):
            raise ValueError(f"episode {number}: expected finite float32 arrays")
        if not np.array_equal(action, np.load(original, allow_pickle=False)):
            raise ValueError(f"episode {number}: paired action differs from original")
        if not np.isin(action[:, 64:], (0, 1)).all() or not np.array_equal(state[:, 29:31], action[:, 64:]):
            raise ValueError(f"episode {number}: invalid binary hands/action proxy")
        if not np.allclose(np.linalg.norm(state[:, 31:34], axis=1), 1.0, atol=1e-4):
            raise ValueError(f"episode {number}: projected gravity is not unit length")
        report_path = proprio.with_suffix(".report.json")
        report = json.loads(report_path.read_text())
        if report.get("hand_state_source") not in {"action", "action_proxy"}:
            raise ValueError(f"episode {number}: unexpected hand source in {report_path}")
        raw_video = episode / f"CameraRecord_{stem}_left.mp4"
        syn_video = syn / str(number) / "video_g1_dexterous_occlusion.mp4"
        raw_info, syn_info = video_info(raw_video), video_info(syn_video)
        if raw_info["frames"] != syn_info["frames"] or not np.isclose(raw_info["fps"], 2 * syn_info["fps"], rtol=1e-3):
            raise ValueError(f"episode {number}: raw/syn frame or FPS mismatch {raw_info}/{syn_info}")
        if not np.isclose(syn_info["duration"], 2 * raw_info["duration"], atol=2 / syn_info["fps"]):
            raise ValueError(f"episode {number}: raw/syn duration mismatch")
        description = episode / "llm_description.txt"
        language = description.read_text(encoding="utf-8").strip()
        if not language:
            raise ValueError(f"episode {number}: empty language")
        paths = {"token_path": token, "proprio_path": proprio, "raw_video_path": raw_video,
                 "syn_video_path": syn_video, "text_path": description, "replay_report_path": report_path}
        repair_path = syn_video.with_suffix(".repair.json")
        return {"episode_id": f"{number:06d}", "length": len(action), "language": language,
                **{key: p.resolve().relative_to(ego_root).as_posix() for key, p in paths.items()},
                "raw_video": raw_info, "syn_video": syn_info, "hand_state_source": "action_proxy",
                **({"video_repair": json.loads(repair_path.read_text())} if repair_path.exists() else {})}

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = []
        for row in executor.map(record, range(400)):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"validated {len(rows)}/400 paired episodes", flush=True)
    total = sum(row["length"] for row in rows)
    if total != 442074:
        raise ValueError(f"expected 442074 Ego steps, got {total}")
    return {"format_version": 1, "path_base": "index_parent_parent", "episodes": rows,
            "n_frames": total, "proprio_layout": PROPRIO_LAYOUT_VERSION,
            "cache_root": pseudo.resolve().relative_to(ego_root).as_posix(),
            "frame_mapping": "floor_t_times_video_frames_over_action_length",
            "alignment_basis": "user_confirmed_same_frame_order_fps_halved",
            "decoded_probe_frames": "first_middle_last_per_video"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="resource/dataset/proprietary/ego/2026-5-23")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pseudo-root", default=None)
    parser.add_argument("--syn-video-root", default=None)
    args = parser.parse_args()
    root = Path(args.data_root)
    if bool(args.pseudo_root) != bool(args.syn_video_root):
        parser.error("--pseudo-root and --syn-video-root must be provided together")
    if args.pseudo_root:
        if args.out is None:
            parser.error("paired indexing requires --out at <ego-root>/meta/<index>.json")
        out = Path(args.out)
        payload = build_paired_index(root, Path(args.pseudo_root), Path(args.syn_video_root), out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"indexed {len(payload['episodes'])} paired episodes ({payload['n_frames']} steps) -> {out}")
        return
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
