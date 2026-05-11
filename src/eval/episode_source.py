"""Random sampler over the humanoid-everyday training set, used as the
observation source for offline policy validation.

Each episode contributes an instruction (from ``meta/episodes.jsonl``) and a
video (frames decoded lazily via decord). The validation pipeline anchors
sim time t=0 to episode frame 0 and pulls the closest video frame at every
policy query, so the policy sees the obs distribution it was trained on
while the simulator independently evaluates the *physical execution* of
each predicted action chunk.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import decord


HE_VIDEO_FPS = 30.0
# The SONIC token / cached g1_proprio timeline is 50Hz (calc_sonic_tokens.py
# upsamples from the 30Hz parquet), so proprio frames are indexed at this rate.
HE_PROPRIO_FPS = 50.0


@dataclass
class HEEpisode:
    """A single HE episode picked for validation."""
    episode_index: int
    chunk: int
    video_path: Path
    instruction: str
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / HE_VIDEO_FPS


class HEEpisodeSource:
    """Random-with-seed sampler over HE episodes; per-episode lazy frame
    decoding cached behind ``frame_at(t_seconds)`` and per-episode mmap'd
    proprio behind ``proprio_at(episode_index, t_seconds)``.
    """

    def __init__(
        self,
        data_root: str | Path = "resource/dataset/humanoid-everyday",
        proprio_cache_root: str | Path = "resource/dataset/humanoid-everyday-cache/g1_proprio",
        g1_action_cache_root: str | Path | None = None,
        robot_type: str = "g1",
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.proprio_cache_root = Path(proprio_cache_root)
        # Default the g1_actions cache to the sibling of g1_proprio. Both are
        # written together by calc_sonic_tokens.py so they always live in the
        # same parent directory.
        self.g1_action_cache_root = (
            Path(g1_action_cache_root)
            if g1_action_cache_root is not None
            else self.proprio_cache_root.parent / "g1_actions"
        )
        # Lazy per-episode pelvis pose cache. Built on first ``pelvis_pose_at``
        # call by reading the parquet's ``observation.odometry.{position,quat}``
        # at 30 Hz and upsampling to the 50 Hz token timeline. Not part of
        # ``g1_proprio`` because position is only needed for the GT-replay
        # visualization, not for policy input.
        self._pelvis_pose: dict[int, np.ndarray] = {}
        # Per-episode (vx_body, vy_body, wz) timelines derived from pelvis_pose
        # via finite differencing; built lazily on first ``loco_cmd_at`` call.
        self._loco_cmd: dict[int, np.ndarray] = {}
        self.robot_type = robot_type.lower()
        episodes_path = self.data_root / "meta" / "episodes.jsonl"
        tasks_path = self.data_root / "meta" / "tasks.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing HE episodes index: {episodes_path}")
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing HE tasks index: {tasks_path}")
        with open(episodes_path) as f:
            all_episodes = [json.loads(line) for line in f]
        # HE is a G1+H1 mix; the SONIC token / proprio caches only cover G1
        # because ``calc_sonic_tokens.py`` filters by robot_type. Mirror that
        # filter here so random sampling never selects an H1 episode without
        # a cached proprio (and so eval rolls on the same population the
        # policy was trained on).
        if self.robot_type == "all":
            self._episodes = all_episodes
        else:
            self._episodes = [
                e for e in all_episodes
                if str(e.get("robot_type", "")).lower() == self.robot_type
            ]
        if not self._episodes:
            raise ValueError(
                f"No episodes match robot_type={self.robot_type!r} in {episodes_path}"
            )
        # ``episodes.jsonl``'s ``instruction`` field has been collapsed to a
        # single boilerplate string for every episode in this dataset dump,
        # so we look up the real task text via ``tasks[0] → tasks.jsonl``.
        with open(tasks_path) as f:
            tasks = [json.loads(line) for line in f]
        self._task_by_index: dict[int, dict] = {int(t["task_index"]): t for t in tasks}
        self._rng = random.Random(seed)
        self._proprio_mmap: dict[int, np.ndarray] = {}
        self._g1_action_mmap: dict[int, np.ndarray] = {}

    def sample(self, n: int, seed: Optional[int] = None) -> list[HEEpisode]:
        rng = random.Random(seed) if seed is not None else self._rng
        idx = rng.sample(range(len(self._episodes)), n)
        return [self._make(i) for i in idx]

    def by_index(self, episode_index: int) -> HEEpisode:
        for e in self._episodes:
            if int(e["episode_index"]) == int(episode_index):
                return self._make_from_meta(e)
        raise KeyError(f"episode_index {episode_index} not found")

    def proprio_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return the cached 64-D training-time proprio at the frame closest
        to ``t_seconds`` (50Hz timeline). The cached vector layout is
        ``[jp(29), hand_bits(2), jv(29), odom_quat_xyzw(4)]`` — bit-identical
        to what was fed to the policy during training.

        Sourcing proprio from the training cache (rather than the simulator)
        keeps the policy's input distribution anchored to training. Otherwise
        the sim's pinned-base default-pose proprio drifts OOD frame-by-frame
        and the policy emits jittery wrist commands ('flying hands').
        """
        ep = int(episode_index)
        arr = self._proprio_mmap.get(ep)
        if arr is None:
            path = self.proprio_cache_root / f"episode_{ep:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached proprio for episode {ep}: {path}. "
                    f"Re-run scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py "
                    f"or pass --proprio-cache-root to point at an existing cache."
                )
            arr = np.load(str(path), mmap_mode="r")
            self._proprio_mmap[ep] = arr
        idx = int(round(t_seconds * HE_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)  # copy out of mmap

    def g1_action_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return the cached 31-D training-time G1 action at the frame closest
        to ``t_seconds`` (50 Hz timeline).

        Layout matches what the policy was trained to predict and what the
        env's PD loop consumes: ``[jp(29 MuJoCo-order), hand_bits(2)]``. Used
        by the sim-validation plot to overlay the dataset's ground-truth
        action against the model's commanded action and the sim's actual
        qpos, so the model's fit quality is visible at a glance.
        """
        ep = int(episode_index)
        arr = self._g1_action_mmap.get(ep)
        if arr is None:
            path = self.g1_action_cache_root / f"episode_{ep:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached g1_action for episode {ep}: {path}. "
                    f"Re-run scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py "
                    f"or pass --g1-action-cache-root to point at an existing cache."
                )
            arr = np.load(str(path), mmap_mode="r")
            self._g1_action_mmap[ep] = arr
        idx = int(round(t_seconds * HE_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)

    def pelvis_pose_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return the dataset's pelvis pose at sim time ``t_seconds``.

        Layout: ``(7,) float32 = [x, y, z, qw, qx, qy, qz]`` in HE world frame.
        Loaded on demand from the source parquet's
        ``observation.odometry.{position,quat}`` and upsampled from the native
        30 Hz parquet rate to the 50 Hz token timeline so it lines up with
        ``g1_action_at`` / ``proprio_at`` indexing.

        Used by the GT-replay head in the sim validator to make the floating
        base actually translate / rotate the way the recorded episode did,
        instead of staying pinned to the spawn pose (which makes a walking
        episode look like in-place stepping).
        """
        ep = int(episode_index)
        arr = self._pelvis_pose.get(ep)
        if arr is None:
            arr = self._load_pelvis_pose(ep)
            self._pelvis_pose[ep] = arr
        idx = int(round(t_seconds * HE_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)

    def loco_cmd_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return (vx_body, vy_body, wz) derived from the dataset's pelvis odometry.

        Used to feed the WBC head an apples-to-apples replay command: at time
        ``t``, what velocity was the operator commanding via the real
        platform's WBC. The dataset doesn't store loco_cmd directly, so we
        reconstruct it by finite-differencing ``pelvis_pose_at`` over a 5-frame
        centered window (~100 ms at 50 Hz; smoother than ±1 frame, sharper
        than ±5).

        Output is in the robot's body frame at frame ``t``, clipped to the
        WBC's training range (±1.0 m/s linear, ±1.5 rad/s yaw). Values outside
        that range push the policy OOD.
        """
        ep = int(episode_index)
        arr = self._loco_cmd.get(ep)
        if arr is None:
            arr = self._build_loco_cmd(ep)
            self._loco_cmd[ep] = arr
        idx = int(round(t_seconds * HE_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)

    def _build_loco_cmd(self, episode_index: int) -> np.ndarray:
        """Compute the full (T, 3) body-frame loco_cmd timeline for an episode.

        Pipeline:
          1. Pull 50 Hz pelvis pose (pos[T,3] + quat_wxyz[T,4]).
          2. Extract yaw, unwrap to remove ±π jumps.
          3. Centered finite difference (window=5 frames) → world-frame
             linear velocity + yaw rate.
          4. Project linear velocity into body frame via current yaw.
          5. Clip to WBC training range.
        """
        pose = self.pelvis_pose_at(episode_index, 0.0)  # warms the cache
        full = self._pelvis_pose[int(episode_index)]
        pos = full[:, :3].astype(np.float64)
        quat = full[:, 3:7].astype(np.float64)  # wxyz
        T = pos.shape[0]
        dt = 1.0 / HE_PROPRIO_FPS

        # Yaw from wxyz quat, then unwrap.
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_unwrapped = np.unwrap(yaw)

        # Centered finite difference, window=5 (= ±2 frames). At the edges fall
        # back to a forward/backward diff so the timeline length is preserved.
        W = 2
        v_world = np.zeros((T, 2), dtype=np.float64)
        wz = np.zeros(T, dtype=np.float64)
        for i in range(T):
            i0 = max(0, i - W)
            i1 = min(T - 1, i + W)
            span = (i1 - i0) * dt
            if span <= 0:
                continue
            v_world[i] = (pos[i1, :2] - pos[i0, :2]) / span
            wz[i] = (yaw_unwrapped[i1] - yaw_unwrapped[i0]) / span

        # Project world-frame planar velocity into body frame using yaw at i.
        c, s = np.cos(yaw), np.sin(yaw)
        vx_body = v_world[:, 0] * c + v_world[:, 1] * s
        vy_body = -v_world[:, 0] * s + v_world[:, 1] * c

        cmd = np.stack([vx_body, vy_body, wz], axis=1)
        # WBC training range: ±1.0 m/s linear, ±1.5 rad/s yaw. Outside these
        # the policy is OOD; clipping is safer than letting it extrapolate.
        cmd[:, 0] = np.clip(cmd[:, 0], -1.0, 1.0)
        cmd[:, 1] = np.clip(cmd[:, 1], -1.0, 1.0)
        cmd[:, 2] = np.clip(cmd[:, 2], -1.5, 1.5)
        return cmd.astype(np.float32)

    def _load_pelvis_pose(self, episode_index: int) -> np.ndarray:
        """Read parquet odometry for ``episode_index`` and upsample to 50 Hz."""
        import os

        os.environ.setdefault(
            "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage",
        )
        import polars as pl
        from src.data.preprocess import resample_linear, resample_quat_wxyz

        chunk = episode_index // 1000
        parquet_path = (
            self.data_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing parquet for pelvis pose lookup: {parquet_path}"
            )
        df = pl.read_parquet(
            parquet_path,
            columns=["observation.odometry.position", "observation.odometry.quat"],
        )
        pos = np.asarray(df["observation.odometry.position"].to_list(), dtype=np.float32)
        quat_wxyz = np.asarray(df["observation.odometry.quat"].to_list(), dtype=np.float32)
        n_src = pos.shape[0]
        # Match the resampling pass calc_sonic_tokens.py applies to jp/proprio
        # so the pelvis pose lands on the same indices as ``g1_action_at(t)``.
        n_tgt = max(1, int(round(n_src * HE_PROPRIO_FPS / HE_VIDEO_FPS)))
        if n_tgt != n_src:
            pos = resample_linear(pos.astype(np.float64), n_src, n_tgt).astype(np.float32)
            quat_wxyz = resample_quat_wxyz(quat_wxyz, n_src, n_tgt)
        pose = np.concatenate([pos, quat_wxyz], axis=-1).astype(np.float32)
        return pose

    def _make(self, list_idx: int) -> HEEpisode:
        return self._make_from_meta(self._episodes[list_idx])

    def _make_from_meta(self, meta: dict) -> HEEpisode:
        ep = int(meta["episode_index"])
        chunk = ep // 1000
        video_path = (
            self.data_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / "egocentric"
            / f"episode_{ep:06d}.mp4"
        )
        # Resolve the task text via tasks.jsonl rather than the (broken)
        # per-episode ``instruction`` field. Prefer ``description`` (free
        # text) over ``task`` (path-style label) because the model was
        # trained on natural-language instructions.
        task_idx = int(meta["tasks"][0]) if meta.get("tasks") else None
        task_meta = self._task_by_index.get(task_idx) if task_idx is not None else None
        if task_meta is not None:
            instruction = str(
                task_meta.get("description")
                or task_meta.get("task")
                or "perform the task"
            )
        else:
            instruction = str(meta.get("instruction", "perform the task"))
        return HEEpisode(
            episode_index=ep,
            chunk=chunk,
            video_path=video_path,
            instruction=instruction,
            n_frames=int(meta.get("length", 0)),
        )


class FrameStream:
    """Lazy frame reader for a single video. Returns a PIL.Image.RGB at the
    frame closest to ``t_seconds`` (clamped to the last frame)."""

    def __init__(self, video_path: Path, fps: float = HE_VIDEO_FPS):
        self.video_path = Path(video_path)
        self._vr = decord.VideoReader(str(self.video_path), num_threads=1)
        self.fps = float(fps)
        self.n_frames = len(self._vr)

    def frame_at(self, t_seconds: float) -> Image.Image:
        idx = int(round(t_seconds * self.fps))
        idx = max(0, min(idx, self.n_frames - 1))
        arr = self._vr[idx].asnumpy()  # (H, W, 3) uint8
        return Image.fromarray(arr)

    def close(self) -> None:
        # decord keeps file handles open until GC'd; explicit del helps when
        # iterating many episodes.
        self._vr = None
