"""Open-loop evaluation for a Kappa checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kappa.data.collator import KappaCollator
from kappa.data.datamodule import build_stage_dataset, split_dataset
from kappa.data.schemas import MOTION_TOKEN_DIM
from kappa.models.qwen_sonicfm import QwenSonicFM
from kappa.training.runner import move_batch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = QwenSonicFM.from_checkpoint(args.ckpt, map_location=device).to(device).eval()
    dataset = build_stage_dataset(args.stage, sample_stride=args.sample_stride)
    _, val_ds = split_dataset(dataset, max_train_samples=1, max_val_samples=args.max_val_samples)
    loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=KappaCollator(), num_workers=args.num_workers)

    sums = {"token_l2": 0.0, "token_cosine": 0.0, "hand_accuracy": 0.0, "bound_ratio": 0.0}
    n = 0
    for batch in tqdm(loader, desc="open-loop eval"):
        batch = move_batch(batch, device)
        pred = model.predict_action(input_ids=batch["input_ids"], images=batch["images"], state=batch["state"], state_mask=batch["state_mask"])
        target = batch["normalized_actions"]
        mask = batch["sonic_mask"]
        motion_mask = mask[..., :MOTION_TOKEN_DIM].mean(dim=-1) > 0
        pred_motion = pred[..., :MOTION_TOKEN_DIM]
        tgt_motion = target[..., :MOTION_TOKEN_DIM]
        if motion_mask.any():
            diff = pred_motion[motion_mask] - tgt_motion[motion_mask]
            sums["token_l2"] += float(diff.pow(2).mean().sqrt())
            sums["token_cosine"] += float(F.cosine_similarity(pred_motion[motion_mask], tgt_motion[motion_mask], dim=-1).mean())
            sums["bound_ratio"] += float((pred_motion[motion_mask].abs() > 1.25).float().mean())
        hand_mask = mask[..., MOTION_TOKEN_DIM:].mean(dim=-1) > 0
        if hand_mask.any():
            hand_pred = pred[..., MOTION_TOKEN_DIM:] >= 0.5
            hand_tgt = target[..., MOTION_TOKEN_DIM:] >= 0.5
            sums["hand_accuracy"] += float((hand_pred[hand_mask] == hand_tgt[hand_mask]).float().mean())
        n += 1
    metrics = {k: v / max(n, 1) for k, v in sums.items()}
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
