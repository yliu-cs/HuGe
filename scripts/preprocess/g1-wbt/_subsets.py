"""Shared subset table + per-episode metadata helpers for G1-WBT preprocessing.

Layout: data tables stay in v3-style ``data/chunk-NNN/file-NNN.parquet``
(many episodes per file), with the per-episode index in
``meta/episodes/chunk-NNN/file-NNN.parquet``; videos are now per-episode,
under ``episode_videos/observation.images.<key>/episode_NNNNNN.mp4`` (one
mp4 per episode, starting at frame 0). The 11 subsets vary along three axes:

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

# Per-vendor open/close binarization rule for ``observation.state.hand_state``,
# read directly off each subset's README rather than fitted from the data:
#
#   Inspire (12D): per-finger scalar in [0.0, 1.0], 0 = open → 1 = closed.
#                  Thumb open/close lives at index 4 (left) / 10 (right).
#   BrainCo (12D): per-finger scalar in [0.0, 1.0], 0 = open → 1 = closed.
#                  Thumb open/close at 0 (left) / 6 (right).
#   Dex1    (2D):  single gripper scalar per hand at 0 (left) / 1 (right);
#                  range is 5.5 = open → 0.0 = closed (NOTE: inverted polarity).
#
# Each row is ``(left_idx, right_idx, threshold, comparator)``; ``comparator``
# is ``">"`` when "closed" means the value crossed above the threshold and
# ``"<"`` when "closed" means the value dropped below it.
HAND_RULES: dict[str, tuple[int, int, float, str]] = {
    "inspire": (4, 10, 0.5,  ">"),
    "brainco": (0, 6,  0.5,  ">"),
    "dex1":    (0, 1,  2.75, "<"),
}


def compute_hand_bits(hand_state: "np.ndarray", hand_type: str) -> "np.ndarray":
    """Return ``(n, 2)`` ``[left_closed, right_closed]`` float32 bits.

    Applies the README-defined rule for ``hand_type``. Unlike the HE pipeline
    we do *not* take ``abs()`` (the data is already non-negative) and we do
    *not* run the Octo-style reverse-scan smoothing — the dataset's hand_state
    is monotonic and bounded, so a fixed midpoint threshold gives a clean,
    polarity-correct label without further filtering.
    """
    if hand_type not in HAND_RULES:
        raise KeyError(f"Unknown hand_type {hand_type!r}; expected one of {list(HAND_RULES)}")
    left_idx, right_idx, thr, cmp_ = HAND_RULES[hand_type]
    if hand_state.ndim != 2 or hand_state.shape[1] <= max(left_idx, right_idx):
        raise ValueError(
            f"hand_state shape {hand_state.shape} too small for hand_type={hand_type!r} "
            f"(needs > {max(left_idx, right_idx)})"
        )
    left = hand_state[:, left_idx]
    right = hand_state[:, right_idx]
    if cmp_ == ">":
        l_closed = (left > thr)
        r_closed = (right > thr)
    else:
        l_closed = (left < thr)
        r_closed = (right < thr)
    return np.stack([l_closed, r_closed], axis=1).astype(np.float32)


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

    ``episode_index``        i64 — episode index within the subset (also indexes
                                   ``episode_videos/.../episode_NNNNNN.mp4``).
    ``length``               i64 — frame count at native (30 Hz) rate; equals
                                   the per-episode mp4 frame count.
    ``data_chunk``           i64 — points into ``data/chunk-NNN/``.
    ``data_file``            i64 — points into ``file-NNN.parquet``.
    ``task``                 str — concatenated task description(s) for the LLM prompt.

    The ``tasks`` column in LeRobot v3 is ``list[str]`` (multi-task episodes); we
    join with ``" / "`` so a single string lands in the dataloader.
    """
    paths = _episodes_meta_paths(subset.abs_root(data_root))
    if not paths:
        raise FileNotFoundError(
            f"No episode meta files under {subset.abs_root(data_root)}/meta/episodes"
        )

    keep = [
        "episode_index", "length", "tasks",
        "data/chunk_index", "data/file_index",
    ]
    frames = [pl.read_parquet(p, columns=keep) for p in paths]
    df = pl.concat(frames).sort("episode_index")

    return df.select([
        pl.col("episode_index"),
        pl.col("length"),
        pl.col("data/chunk_index").alias("data_chunk"),
        pl.col("data/file_index").alias("data_file"),
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
        odom_pos:  (n, 3)  float32 — robot_q_current[:, 0:3], pelvis xyz.
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
    odom_pos = rq[:, 0:3].astype(np.float32, copy=False)
    odom_quat = rq[:, 3:7].astype(np.float32, copy=False)
    return {
        "joint_pos": joint_pos,
        "odom_pos": odom_pos,
        "odom_quat": odom_quat,
        "hand_state": hs,
    }
