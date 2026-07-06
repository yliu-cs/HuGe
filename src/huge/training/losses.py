"""Losses for QwenSonicFM."""

from __future__ import annotations

import torch

from huge.data.schemas import ACTION_DIM


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(loss).sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def qwen_sonicfm_loss(
    pred: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    *,
    sonic_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if target_actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"target action dim {target_actions.shape[-1]} != {ACTION_DIM}")
    if "velocity" in pred:
        target_velocity = pred.get("target_velocity")
        if target_velocity is None:
            raise KeyError("pred missing target_velocity")
        velocity = pred["velocity"]
        velocity_dim = velocity.shape[-1]
        token_mask = None if sonic_mask is None else sonic_mask[..., :velocity_dim]
        target_velocity = target_velocity.to(velocity.dtype)
        if target_velocity.shape[-1] != velocity_dim:
            raise ValueError(f"target_velocity dim {target_velocity.shape[-1]} != velocity dim {velocity_dim}")
        loss_token = _masked_mean((velocity - target_velocity).pow(2), token_mask)
        return {
            "loss": loss_token,
            "loss_token": loss_token,
        }
    if "pred_actions" not in pred:
        raise KeyError("pred missing velocity or pred_actions")
    pred_actions = pred["pred_actions"]
    target = pred.get("target_actions", target_actions[:, : pred_actions.shape[1]])
    target = target.to(pred_actions.dtype)
    token_mask = None if sonic_mask is None else sonic_mask[:, : pred_actions.shape[1], : pred_actions.shape[-1]]
    loss_token = _masked_mean((pred_actions - target).abs(), token_mask)
    return {
        "loss": loss_token,
        "loss_token": loss_token,
    }
