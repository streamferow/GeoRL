from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from grpo.models.rollout import Rollout


@dataclass
class GroupBatch:
    """G rollouts sharing one prompt."""

    prompt_id: str
    rollouts: list[Rollout]
    rewards: torch.Tensor
    advantages: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class RolloutBuffer:
    def __init__(self) -> None:
        self.groups: list[GroupBatch] = []

    def add(self, group: GroupBatch) -> None:
        self.groups.append(group)

    def clear(self) -> None:
        self.groups.clear()

    def __len__(self) -> int:
        return len(self.groups)
