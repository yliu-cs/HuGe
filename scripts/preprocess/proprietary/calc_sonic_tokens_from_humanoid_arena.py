"""Convert raw HumanoidArena SONIC npz episodes into Huge's 66-D token / 34-D proprio cache.

Step 0: validate every raw episode (schema/shape/contiguity/video presence), skip and
record anything that fails. Step 1: reorder qpos into Huge's grouped 29-D layout, build the
66-D token and 34-D proprio arrays, symlink the referenced video, and write a language map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from huge.data.preprocess import atomic_save_npy, projected_gravity_from_quat_wxyz

REQUIRED_SCHEMA_VERSION = "sonic_episode_v3"

# Copied verbatim from HumanoidArena/isaaclab_twist2_g1/tools/data_tools/sonic2lerobot.py
# (SONIC_JOINT_NAMES) — the interleaved order robot_qpos_before_decimation is already in.
CANONICAL_NAMES_29 = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Target grouped order — matches PROPRIO_LAYOUT in huge.data.schemas (left_leg/right_leg/
# waist/left_arm/right_arm), copied from cola_merged_cleaned_0530/meta/info.json's
# observation.state name order (hand joints excluded; those come from grip binary instead).
GROUP_JOINTS = {
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                 "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}
GROUP_IDX = {g: [CANONICAL_NAMES_29.index(n) for n in names] for g, names in GROUP_JOINTS.items()}
GROUP_ORDER = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")

# ponytail: task field is a fixed simulator name, not a sentence — hand-write the phrasing.
# Add an entry here if a future HumanoidArena task is converted with this script.
TASK_DESCRIPTIONS = {
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "pick up the box and place it",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    return str(data[key].item())


def validate_episode(npz_path: Path) -> str | None:
    """Return None if the episode is usable, else a short reason it was skipped."""
    with np.load(npz_path, allow_pickle=True) as data:
        schema = _scalar(data, "schema_version")
        if schema != REQUIRED_SCHEMA_VERSION:
            return f"schema_version={schema!r}"

        for key, shape_tail in (
            ("encoder_latent", (64,)),
            ("robot_qpos_before_decimation", (29,)),
            ("robot_root_orientation", (4,)),
        ):
            if key not in data.files:
                return f"missing {key}"
            arr = data[key]
            if arr.ndim != 2 or arr.shape[1:] != shape_tail:
                return f"{key} shape {arr.shape} != (T,{shape_tail[0]})"

        if "episode_step" not in data.files:
            return "missing episode_step"
        step = data["episode_step"]
        if step.ndim != 1 or not np.array_equal(np.diff(step), np.ones(len(step) - 1, dtype=step.dtype)):
            return "episode_step not contiguous"

        for key in ("pico_left_grip_binary", "pico_right_grip_binary"):
            if key not in data.files:
                return f"missing {key}"

        video_key = "vision_rgb_video_path"
        if video_key not in data.files:
            return f"missing {video_key}"
        video_path = npz_path.parent / _scalar(data, video_key)
        if not video_path.is_file():
            return f"video missing: {video_path.name}"

    return None


def reorder_qpos_grouped(qpos: np.ndarray) -> np.ndarray:
    """(T,29) interleaved SONIC order -> (T,29) left_leg/right_leg/waist/left_arm/right_arm."""
    return np.concatenate([qpos[:, GROUP_IDX[g]] for g in GROUP_ORDER], axis=-1)


def convert_episode(npz_path: Path, out_root: Path, episode_idx: int) -> str:
    stem = f"episode_{episode_idx:06d}"
    with np.load(npz_path, allow_pickle=True) as data:
        encoder_latent = np.asarray(data["encoder_latent"], dtype=np.float32)
        left_grip = np.asarray(data["pico_left_grip_binary"], dtype=np.float32).reshape(-1, 1)
        right_grip = np.asarray(data["pico_right_grip_binary"], dtype=np.float32).reshape(-1, 1)
        token = np.concatenate([encoder_latent, left_grip, right_grip], axis=-1)

        qpos = reorder_qpos_grouped(np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32))
        gravity = projected_gravity_from_quat_wxyz(np.asarray(data["robot_root_orientation"], dtype=np.float32))
        # ponytail: no continuous hand-state field survives to a single scalar (raw 7-DOF
        # hand_action_left/right is a duplicate of human_*_hand, not a hand "state"); reuse
        # the same grip binary already used for the action token's hand dims.
        proprio = np.concatenate([qpos, left_grip, right_grip, gravity], axis=-1)

        video_src = npz_path.parent / str(data["vision_rgb_video_path"].item())

    atomic_save_npy(out_root / "sonic_tokens" / f"{stem}.npy", token)
    atomic_save_npy(out_root / "g1_proprio" / f"{stem}.npy", proprio)

    video_dst = out_root / "videos" / "chunk-000" / "observation.images.ego_view" / f"{stem}.mp4"
    video_dst.parent.mkdir(parents=True, exist_ok=True)
    if video_dst.exists() or video_dst.is_symlink():
        video_dst.unlink()
    video_dst.symlink_to(video_src.resolve())

    task_name = str(npz_path.stem).rsplit("_sonic_", 1)[0]
    return TASK_DESCRIPTIONS.get(task_name, task_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="resource/dataset/HumanoidArena_dataset_raw/HOI_pp_box/sonic/yb")
    parser.add_argument("--out-root", default="resource/dataset/HumanoidArena/HOI_pp_box")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    npz_paths = sorted(raw_root.glob("*.npz"))

    skipped: dict[str, str] = {}
    valid: list[Path] = []
    for npz_path in tqdm(npz_paths, desc="validating"):
        reason = validate_episode(npz_path)
        if reason is None:
            valid.append(npz_path)
        else:
            skipped[npz_path.name] = reason

    language_map: dict[str, str] = {}
    for episode_idx, npz_path in enumerate(tqdm(valid, desc="converting")):
        stem = f"episode_{episode_idx:06d}"
        language_map[stem] = convert_episode(npz_path, out_root, episode_idx)

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "language_map.json").write_text(json.dumps(language_map, indent=2), encoding="utf-8")
    (meta_dir / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "raw_root": str(raw_root),
                "out_root": str(out_root),
                "schema_version": REQUIRED_SCHEMA_VERSION,
                "valid_count": len(valid),
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"validated {len(npz_paths)} episodes: {len(valid)} ok, {len(skipped)} skipped")
    for name, reason in skipped.items():
        print(f"  skip {name}: {reason}")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
