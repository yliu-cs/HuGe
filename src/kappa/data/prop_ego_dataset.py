"""Proprietary ego dataset using EgoDex-style token masking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, pad_egodex_action
from kappa.data.transforms import token_to_video_frame_index

_TOK_RE = re.compile(r"^trackingData_(?P<stem>.+)_latent_tokens_66d\.npy$")


class PropEgoDataset:
    def __init__(self, data_root: str | Path, *, action_horizon: int = ACTION_HORIZON, sample_stride: int = 1, max_episodes: int | None = None) -> None:
        self.data_root = Path(data_root)
        self.action_horizon = action_horizon
        self.sample_stride = sample_stride
        self.episodes = self._discover(max_episodes)
        self.index = [(ei, t) for ei, ep in enumerate(self.episodes) for t in range(0, ep["length"], sample_stride)]

    def _discover(self, max_episodes: int | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tok in sorted(self.data_root.rglob("*_latent_tokens_66d.npy")):
            m = _TOK_RE.match(tok.name)
            if not m:
                continue
            arr = np.load(tok, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
                continue
            stem = m.group("stem")
            desc = tok.parent / "llm_description.txt"
            video = tok.parent / f"CameraRecord_{stem}_left.mp4"
            out.append({
                "token_path": tok,
                "video_path": video if video.exists() else None,
                "language": desc.read_text(encoding="utf-8").strip() if desc.exists() else "perform the task",
                "episode_id": tok.parent.relative_to(self.data_root).as_posix(),
                "length": int(arr.shape[0]),
            })
            if max_episodes is not None and len(out) >= max_episodes:
                break
        if not out:
            raise FileNotFoundError(f"no proprietary ego token files under {self.data_root}")
        return out

    def __len__(self) -> int:
        return len(self.index)

    def episode_spans(self) -> np.ndarray:
        spans = np.empty((len(self.episodes), 2), dtype=np.int64)
        offset = 0
        for i, ep in enumerate(self.episodes):
            n = (ep["length"] + self.sample_stride - 1) // self.sample_stride
            spans[i] = (offset, n)
            offset += n
        return spans

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        raw = np.load(ep["token_path"])
        chunk = raw[t : t + self.action_horizon, :64]
        valid = len(chunk)
        action, mask = pad_egodex_action(chunk)
        if valid < self.action_horizon:
            padded = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded_mask = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded[:valid] = action
            padded_mask[:valid] = mask
            action, mask = padded, padded_mask
        video_frame_idx = None
        if ep["video_path"] is not None:
            video_len = max(1, int(round(ep["length"] * 30 / 50)))
            video_frame_idx = token_to_video_frame_index(t, ep["length"], video_len)
        return {
            "dataset_id": "proprietary_ego",
            "task_id": "proprietary_ego",
            "episode_id": ep["episode_id"],
            "frame_idx": t,
            "video_path": str(ep["video_path"]) if ep["video_path"] else None,
            "video_frame_idx": video_frame_idx,
            "language": ep["language"],
            "state": np.zeros((PROPRIO_DIM,), dtype=np.float32),
            "state_mask": np.zeros((1,), dtype=np.float32),
            "normalized_actions": action,
            "sonic_mask": mask,
            "image": None,
        }
