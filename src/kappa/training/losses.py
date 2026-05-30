"""Losses for QwenSonicFM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kappa.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM


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
    lambda_hand: float = 0.5,
) -> dict[str, torch.Tensor]:
    if target_actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"target action dim {target_actions.shape[-1]} != {ACTION_DIM}")
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
    hand_logits = pred.get("hand_logits")
    if hand_logits is None:
        loss_hand = torch.zeros((), dtype=loss_token.dtype, device=loss_token.device)
        hand_acc = torch.zeros((), dtype=loss_token.dtype, device=loss_token.device)
        loss = loss_token
    else:
        hand_target = target_actions[..., MOTION_TOKEN_DIM:ACTION_DIM].to(hand_logits.dtype)
        hand_mask = None if sonic_mask is None else sonic_mask[..., MOTION_TOKEN_DIM:ACTION_DIM]
        loss_hand_raw = F.binary_cross_entropy_with_logits(hand_logits, hand_target, reduction="none")
        loss_hand = _masked_mean(loss_hand_raw, hand_mask)
        loss = loss_token + float(lambda_hand) * loss_hand
        with torch.no_grad():
            hand_acc = ((torch.sigmoid(hand_logits) >= 0.5) == (hand_target >= 0.5)).float()
            hand_acc = _masked_mean(hand_acc, hand_mask)
    return {
        "loss": loss,
        "loss_token": loss_token,
        "loss_hand": loss_hand,
        "hand_accuracy": hand_acc,
    }
