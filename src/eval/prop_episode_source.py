"""Episode source for the proprietary 2026-05-16 dataset.

Mirrors :class:`src.eval.episode_source.HEEpisodeSource`'s public API so
``scripts/eval/sim_validate.py`` can swap dataset sources via ``--dataset``
without restructuring the rollout loops.

Differences from the HE source:

  * Video / proprio timeline are both native 50 Hz (HE: 30 Hz video, 50 Hz
    upsampled proprio). The ``video_fps`` attribute on each episode lets
    :class:`FrameStream` index correctly without a hard-coded constant.
  * No positional pelvis odometry — the parquet ships
    ``observation.root_orientation`` (4-D base quat) but no
    ``observation.odometry.position``. ``pelvis_pose_at`` returns identity
    position with the recorded base quat; ``loco_cmd_at`` returns zeros.
    Consequence: the ``gt_wbc`` head's ``base_xy_drift_*`` metric becomes
    "absolute walk-away from spawn", not "drift vs GT trajectory". The
    ``sonic`` head is unaffected (only proprio + image + instruction feed
    the policy).
  * ``planner_movement / planner_speed / planner_mode / delta_heading`` are
    all sentinels (zeros / -1) in this dataset dump, so they can't replace
    the missing odometry-derived loco_cmd.
  * ``episodes.jsonl`` ships task text inline (``tasks: ["<text>"]``)
    rather than indices into ``tasks.jsonl``, so the lookup path differs.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


PROP_VIDEO_FPS = 50.0
PROP_PROPRIO_FPS = 50.0


@dataclass
class PropEpisode:
    """A single proprietary episode picked for validation."""
    episode_index: int
    chunk: int
    video_path: Path
    instruction: str
    n_frames: int
    video_fps: float = PROP_VIDEO_FPS

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.video_fps


class PropEpisodeSource:
    """Random-with-seed sampler over the proprietary dataset's 60 episodes.

    Cached 50 Hz proprio is mmap'd from
    ``<data_root>/g1_proprio/episode_*.npy`` (same 64-D layout as HE).
    Base orientation is lazily read from the parquet when first requested.
    """

    def __init__(
        self,
        data_root: str | Path = "resource/dataset/proprietary/2026-05-16",
        proprio_cache_root: str | Path | None = None,
        sonic_tokens_root: str | Path | None = None,
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.proprio_cache_root = (
            Path(proprio_cache_root)
            if proprio_cache_root is not None
            else self.data_root / "g1_proprio"
        )
        # SONIC token cache lives next to ``g1_proprio`` by default; for the
        # proprietary dump that's ``<data-root>/sonic_tokens/``.
        self.sonic_tokens_root = (
            Path(sonic_tokens_root)
            if sonic_tokens_root is not None
            else self.proprio_cache_root.parent / "sonic_tokens"
        )
        episodes_path = self.data_root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing proprietary episodes index: {episodes_path}")
        with open(episodes_path) as f:
            self._episodes = [json.loads(line) for line in f]
        if not self._episodes:
            raise ValueError(f"No episodes in {episodes_path}")
        self._rng = random.Random(seed)
        self._proprio_mmap: dict[int, np.ndarray] = {}
        self._sonic_mmap: dict[int, np.ndarray] = {}
        self._root_quat: dict[int, np.ndarray] = {}

    def sample(self, n: int, seed: Optional[int] = None) -> list[PropEpisode]:
        rng = random.Random(seed) if seed is not None else self._rng
        idx = rng.sample(range(len(self._episodes)), n)
        return [self._make(i) for i in idx]

    def by_index(self, episode_index: int) -> PropEpisode:
        for e in self._episodes:
            if int(e["episode_index"]) == int(episode_index):
                return self._make_from_meta(e)
        raise KeyError(f"episode_index {episode_index} not found")

    def proprio_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return the cached 64-D proprio at the frame closest to ``t_seconds``.

        Layout: ``[jp(29), hand_bits(2), jv(29), odom_quat_wxyz(4)]`` —
        bit-identical to the HE cache because both go through the same
        ``calc_sonic_tokens.py`` pipeline.
        """
        ep = int(episode_index)
        arr = self._proprio_mmap.get(ep)
        if arr is None:
            path = self.proprio_cache_root / f"episode_{ep:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached proprio for episode {ep}: {path}. "
                    f"Re-run scripts/preprocess/proprietary/calc_sonic_tokens.py "
                    f"or pass --proprio-cache-root."
                )
            arr = np.load(str(path), mmap_mode="r")
            self._proprio_mmap[ep] = arr
        idx = int(round(t_seconds * PROP_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)

    def sonic_token_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return the cached 66-D SONIC token at the frame closest to
        ``t_seconds``. Layout matches the proprietary
        ``sonic_tokens/episode_NNNNNN.npy``: 64-D ``action.motion_token``
        latent concatenated with 2 zero hand bits (fake fingers per the
        dataset contract).
        """
        ep = int(episode_index)
        arr = self._sonic_mmap.get(ep)
        if arr is None:
            path = self.sonic_tokens_root / f"episode_{ep:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached SONIC tokens for episode {ep}: {path}. "
                    f"Re-run scripts/preprocess/proprietary/calc_sonic_tokens.py."
                )
            arr = np.load(str(path), mmap_mode="r")
            self._sonic_mmap[ep] = arr
        idx = int(round(t_seconds * PROP_PROPRIO_FPS))
        idx = max(0, min(idx, arr.shape[0] - 1))
        return np.asarray(arr[idx], dtype=np.float32)

    def pelvis_pose_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """Return ``(7,) = [x, y, z, qw, qx, qy, qz]`` for the proprietary
        dataset's base at the frame closest to ``t_seconds``.

        ``x, y, z`` are forced to zero because this dataset doesn't record
        positional odometry — every call returns identity position. This
        breaks the GT-replay base-drift metric (``base_xy_drift_*`` in the
        ``gt_wbc`` head) but is harmless for the ``sonic`` head, which
        only uses pelvis quat to anchor :class:`SimProprioAssembler` at
        ``t=0``.
        """
        ep = int(episode_index)
        quat_arr = self._root_quat.get(ep)
        if quat_arr is None:
            quat_arr = self._load_root_quat(ep)
            self._root_quat[ep] = quat_arr
        idx = int(round(t_seconds * PROP_PROPRIO_FPS))
        idx = max(0, min(idx, quat_arr.shape[0] - 1))
        pose = np.zeros(7, dtype=np.float32)
        pose[3:7] = quat_arr[idx]
        return pose

    def loco_cmd_at(self, episode_index: int, t_seconds: float) -> np.ndarray:
        """No GT locomotion command — proprietary's ``teleop.planner_*`` and
        ``teleop.delta_heading`` columns are sentinels, and there's no
        odometry to finite-difference. Always returns zeros.
        """
        return np.zeros(3, dtype=np.float32)

    def _load_root_quat(self, episode_index: int) -> np.ndarray:
        """Read the 50 Hz base-quat timeline from the parquet."""
        os.environ.setdefault(
            "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage",
        )
        import polars as pl

        chunk = episode_index // 1000
        parquet_path = (
            self.data_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing parquet for root-quat lookup: {parquet_path}"
            )
        df = pl.read_parquet(
            parquet_path, columns=["observation.root_orientation"],
        )
        return np.asarray(
            df["observation.root_orientation"].to_list(), dtype=np.float32,
        )

    def _make(self, list_idx: int) -> PropEpisode:
        return self._make_from_meta(self._episodes[list_idx])

    def _make_from_meta(self, meta: dict) -> PropEpisode:
        ep = int(meta["episode_index"])
        chunk = ep // 1000
        video_path = (
            self.data_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / "observation.images.ego_view"
            / f"episode_{ep:06d}.mp4"
        )
        tasks = meta.get("tasks") or []
        if tasks and isinstance(tasks[0], str):
            instruction = tasks[0]
        else:
            instruction = "perform the task"
        return PropEpisode(
            episode_index=ep,
            chunk=chunk,
            video_path=video_path,
            instruction=instruction,
            n_frames=int(meta.get("length", 0)),
            video_fps=PROP_VIDEO_FPS,
        )
