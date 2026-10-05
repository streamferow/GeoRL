from __future__ import annotations

import torch


def k3_kl_penalty(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    ref_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)

    objective = surrogate
    if ref_logprobs is not None:
        objective = objective - kl_coef * k3_kl_penalty(logprobs, ref_logprobs)

    masked = objective * completion_mask
    lengths = completion_mask.sum(dim=-1).clamp_min(1)
    per_completion = masked.sum(dim=-1) / lengths
    return -per_completion.mean()
