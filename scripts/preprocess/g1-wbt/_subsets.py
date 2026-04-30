"""Shared subset table + per-episode metadata helpers for G1-WBT preprocessing.

LeRobot v3 layout: every G1-WBT subdataset packs many episodes into a small
number of ``data/chunk-NNN/file-NNN.parquet`` files, with the per-episode index
(length, video file/timestamp ranges) stored separately under
``meta/episodes/chunk-NNN/file-NNN.parquet``. The 11 subsets vary along three
axes:

* ``hand_type`` (``inspire`` / ``brainco`` / ``dex1``) — picks the columns of
  ``observation.state.hand_state`` that go through the open/close binarizer.
* ``video_key`` (``head_stereo_left`` / ``cam_0``) — which mp4 stream we treat
  as egocentric.
* ``nested_dir`` — three ``*_MainCamOnly`` subsets put the LeRobot tree inside
  an extra ``G1_WB_Dex5_<Task>/`` directory.

Both preprocessing scripts and ``WBTDataset`` derive paths from this single
table so that adding a subset is a one-line change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import polars as pl


# (subset_dir_name, hand_type, video_key, nested_dir_or_None)
_SUBSETS: list[tuple[str, str, str, str | None]] = [
    ("G1_WBT_Brainco_Collect_Plates_Into_Dishwasher", "brainco", "head_stereo_left", None),
    ("G1_WBT_Brainco_Make_The_Bed",                   "brainco", "head_stereo_left", None),
    ("G1_WBT_Brainco_Pickup_Pillow",                  "brainco", "head_stereo_left", None),
    ("G1_WBT_Dex1_Put_Clothes_into_Washing_Machine",  "dex1",    "head_stereo_left", None),
    ("G1_WBT_Inspire_Put_Clothes_Into_Basket",         "inspire", "head_stereo_left", None),
    ("G1_WBT_Inspire_Put_Clothes_into_Washing_Machine","inspire", "head_stereo_left", None),
    ("G1_WBT_Inspire_Put_Drinks_Into_Fridge",          "inspire", "head_stereo_left", None),
    ("G1_WBT_Inspire_Put_Vegetables_Into_Basket",      "inspire", "head_stereo_left", None),
    ("G1_WBT_Inspire_Collect_Clothes_MainCamOnly",                 "inspire", "cam_0", "G1_WB_Dex5_Collect_Clothes"),
    ("G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",                   "inspire", "cam_0", "G1_WB_Dex5_Pickup_Pillow"),
    ("G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly","inspire", "cam_0", "G1_WB_Dex5_Put_Clothes_into_Washing_Machine"),
]

# (left_idx, right_idx) into observation.state.hand_state for binarization.
# Inspire/BrainCo: 12D = two 6D blocks; Dex1: 2D = one scalar per hand.
HAND_INDEX_TABLE: dict[str, tuple[int, int]] = {
    "inspire": (4, 10),  # thumb open/close (per analysis report §2.4)
    "brainco": (0, 6),   # thumb open/close (BrainCo block ordering)
    "dex1":    (0, 1),   # gripper open/close scalars
}


@dataclass(frozen=True)
class Subset:
    name: str           # directory name under <data_root>/
    hand_type: str      # "inspire" | "brainco" | "dex1"
    video_key: str      # mp4 directory under videos/observation.images.<key>/
    nested_dir: str | None  # extra subdirectory for *_MainCamOnly variants

    @property
    def root(self) -> str:
        """Directory holding ``data/``, ``meta/``, ``videos/`` for this subset.

        Always returned as a relative path under ``<data_root>``; callers join
        it onto whatever root they're configured against.
        """
        return self.name if self.nested_dir is None else os.path.join(self.name, self.nested_dir)

    def abs_root(self, data_root: str) -> str:
        return os.path.join(data_root, self.root)


def list_subsets(data_root: str | None = None) -> list[Subset]:
    """Return the (filtered to existing) Subset entries.

    When ``data_root`` is given, drop subsets whose directory does not exist on
    disk so a partial download is still usable. Otherwise return the full list.
    """
    out = [Subset(*s) for s in _SUBSETS]
    if data_root is None:
        return out
    return [s for s in out if os.path.isdir(s.abs_root(data_root))]


# ---------------------------------------------------------------------------
# Episode metadata loading
# ---------------------------------------------------------------------------

def _episodes_meta_paths(subset_root: str) -> list[str]:
    """All ``meta/episodes/chunk-*/file-*.parquet`` shards in deterministic order."""
    base = os.path.join(subset_root, "meta", "episodes")
    out: list[str] = []
    if not os.path.isdir(base):
        return out
    for chunk in sorted(os.listdir(base)):
        cdir = os.path.join(base, chunk)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".parquet"):
                out.append(os.path.join(cdir, f))
    return out


def load_subset_episode_meta(subset: Subset, data_root: str) -> pl.DataFrame:
    """Load per-episode index for one subset, normalized to a uniform schema.

    Returns a DataFrame with columns:

    ``episode_index``        i64 — episode index within the subset.
    ``length``               i64 — frame count at native (30 Hz) rate.
    ``data_chunk``           i64 — points into ``data/chunk-NNN/``.
    ``data_file``            i64 — points into ``file-NNN.parquet``.
    ``video_chunk``          i64 — points into ``videos/<key>/chunk-NNN/``.
    ``video_file``           i64 — points into ``file-NNN.mp4``.
    ``video_from_ts``        f32 — start timestamp inside the mp4 (seconds).
    ``video_to_ts``          f32 — end timestamp inside the mp4 (seconds).
    ``task``                 str — concatenated task description(s) for the LLM prompt.

    The ``tasks`` column in LeRobot v3 is ``list[str]`` (multi-task episodes); we
    join with ``" / "`` so a single string lands in the dataloader.
    """
    paths = _episodes_meta_paths(subset.abs_root(data_root))
    if not paths:
        raise FileNotFoundError(
            f"No episode meta files under {subset.abs_root(data_root)}/meta/episodes"
        )
    vk = subset.video_key
    vid_chunk_col = f"videos/observation.images.{vk}/chunk_index"
    vid_file_col = f"videos/observation.images.{vk}/file_index"
    vid_from_col = f"videos/observation.images.{vk}/from_timestamp"
    vid_to_col   = f"videos/observation.images.{vk}/to_timestamp"

    keep = [
        "episode_index", "length", "tasks",
        "data/chunk_index", "data/file_index",
        vid_chunk_col, vid_file_col, vid_from_col, vid_to_col,
    ]
    frames = [pl.read_parquet(p, columns=keep) for p in paths]
    df = pl.concat(frames).sort("episode_index")

    return df.select([
        pl.col("episode_index"),
        pl.col("length"),
        pl.col("data/chunk_index").alias("data_chunk"),
        pl.col("data/file_index").alias("data_file"),
        pl.col(vid_chunk_col).alias("video_chunk"),
        pl.col(vid_file_col).alias("video_file"),
        pl.col(vid_from_col).cast(pl.Float32).alias("video_from_ts"),
        pl.col(vid_to_col).cast(pl.Float32).alias("video_to_ts"),
        pl.col("tasks").list.join(" / ").alias("task"),
    ])


def iter_data_files(subset: Subset, data_root: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(chunk_idx, file_idx, abs_path)`` for every ``data/chunk-*/file-*.parquet``."""
    base = os.path.join(subset.abs_root(data_root), "data")
    for chunk_name in sorted(os.listdir(base)):
        cdir = os.path.join(base, chunk_name)
        if not os.path.isdir(cdir):
            continue
        chunk_idx = int(chunk_name.split("-")[1])
        for fname in sorted(os.listdir(cdir)):
            if not fname.endswith(".parquet"):
                continue
            file_idx = int(fname[len("file-"):-len(".parquet")])
            yield chunk_idx, file_idx, os.path.join(cdir, fname)


def load_episode_rows(
    data_root: str,
    subset: Subset,
    data_chunk: int,
    data_file: int,
    episode_index: int,
) -> dict[str, np.ndarray]:
    """Read the rows belonging to one episode out of a multi-episode parquet.

    Returns dict with:
        joint_pos: (n, 29) float64 — robot_q_current[:, 7:36], MuJoCo ordering.
        odom_quat: (n, 4)  float32 — robot_q_current[:, 3:7], WXYZ.
        hand_state: (n, D) float32 — raw, dims depend on ``subset.hand_type``.

    ``D = 12`` for inspire/brainco, ``D = 2`` for dex1.
    """
    parquet_path = os.path.join(
        subset.abs_root(data_root), "data",
        f"chunk-{data_chunk:03d}", f"file-{data_file:03d}.parquet",
    )
    cols = [
        "episode_index",
        "observation.state.robot_q_current",
        "observation.state.hand_state",
    ]
    df = pl.read_parquet(parquet_path, columns=cols).filter(
        pl.col("episode_index") == episode_index
    )
    if df.is_empty():
        raise ValueError(
            f"No rows for episode_index={episode_index} in {parquet_path}"
        )

    rq = np.asarray(df["observation.state.robot_q_current"].to_list(), dtype=np.float32)
    hs = np.asarray(df["observation.state.hand_state"].to_list(), dtype=np.float32)
    if rq.shape[1] != 36:
        raise ValueError(f"robot_q_current expected 36D, got {rq.shape[1]} in {parquet_path}")

    joint_pos = rq[:, 7:36].astype(np.float64, copy=False)
    odom_quat = rq[:, 3:7].astype(np.float32, copy=False)
    return {"joint_pos": joint_pos, "odom_quat": odom_quat, "hand_state": hs}
