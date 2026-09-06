"""Proprietary ego dataset using EgoDex-style token masking."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

import numpy as np

from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, pad_egodex_action
from huge.data.transforms import token_to_video_frame_index
from huge.data.sonic_dataset import EpisodeRecord, SonicTokenDataset

PAIRED_INDEX = "meta/stage3_episode_index.json"


def load_paired_index(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    payload = json.loads((root / PAIRED_INDEX).read_text(encoding="utf-8"))
    if (payload.get("format_version") != 1
            or payload.get("path_base") != "index_parent_parent"
            or payload.get("frame_mapping") != "floor_t_times_video_frames_over_action_length"):
        raise ValueError(f"unsupported paired Ego index: {root / PAIRED_INDEX}")
    rows = payload["episodes"]
    ids = [row["episode_id"] for row in rows]
    if len(ids) != 400 or set(ids) != {f"{i:06d}" for i in range(400)}:
        raise ValueError("paired Ego index must contain each episode 000000..000399 exactly once")
    for row in rows:
        if row["length"] <= 0 or row["raw_video"]["frames"] <= 0 or row["raw_video"]["frames"] != row["syn_video"]["frames"]:
            raise ValueError(f"invalid paired lengths for {row['episode_id']}")
    if sum(row["length"] for row in rows) != payload["n_frames"]:
        raise ValueError("paired Ego index total length mismatch")
    return payload


class PairedPropEgoDataset(SonicTokenDataset):
    """Indexed raw/syn videos sharing the same full action and pseudo state."""

    def __init__(self, data_root: str | Path, *, video_source: str, episode_index: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if video_source not in {"raw", "syn"}:
            raise ValueError(f"invalid Ego video source: {video_source}")
        self.data_root = Path(data_root)
        self.video_source = video_source
        self.episode_index = load_paired_index(data_root) if episode_index is None else episode_index
        super().__init__(
            self.data_root / self.episode_index["cache_root"], dataset_id="proprietary_ego",
            language_map={row["episode_id"]: row["language"] for row in self.episode_index["episodes"]},
            strict_video=True, video_len_hint_mode="paired_frame_ratio", **kwargs,
        )

    def _discover_episodes(self) -> list[EpisodeRecord]:
        records = []
        for row in sorted(self.episode_index["episodes"], key=lambda row: row["episode_id"]):
            token = self.data_root / row["token_path"]
            proprio = self.data_root / row["proprio_path"]
            if not token.is_file() or not proprio.is_file():
                raise FileNotFoundError(f"missing indexed action/proprio: {token}/{proprio}")
            records.append(EpisodeRecord(
                self.dataset_id, row["episode_id"], token, proprio,
                self.data_root / row[f"{self.video_source}_video_path"],
                row["raw_video"]["frames"], row["length"],
            ))
        return records

    def data_contract(self) -> dict[str, Any]:
        return {**super().data_contract(), "video_source": self.video_source,
                "ego_index": self.episode_index, "hand_state_source": "action_proxy",
                "logical_root": str(self.data_root), "resolved_input_root": str(self.data_root.resolve())}

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
