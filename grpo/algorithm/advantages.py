from __future__ import annotations

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    if normalize:
        std = rewards.std(dim=-1, keepdim=True).clamp_min(eps)
        return (rewards - mean) / std
    return rewards - mean
