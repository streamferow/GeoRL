from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from grpo.algorithm.buffer import RolloutBuffer
from grpo.config import Config
from grpo.models.loader import ModelBundle
from grpo.models.processor import VLSample, VLProcessor
from grpo.models.rollout import RolloutGenerator
from grpo.rewards.pipeline import RewardPipeline


class GRPOTrainer:
    """
    GRPO loop:
    1. sample G completions per prompt
    2. score with verifiable reward pipeline
    3. group-relative advantages
    4. policy update with KL to ref
    """

    def __init__(
        self,
        bundle: ModelBundle,
        reward_pipeline: RewardPipeline,
        config: Config,
        ref_model: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.reward_pipeline = reward_pipeline
        self.config = config
        self.ref_model = ref_model
        self.generator = RolloutGenerator(
            bundle.model,
            VLProcessor(bundle.processor, config.processor),
            config.rollout,
        )
        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.AdamW(
            bundle.model.parameters(),
            lr=config.grpo.learning_rate,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        raise NotImplementedError

    def _score_group(self, rollouts: list, sample: VLSample) -> torch.Tensor:
        scores = []
        gt = (sample.metadata or {}).get("ground_truth")
        ctx = (sample.metadata or {}).get("reward_context")
        for r in rollouts:
            scores.append(
                self.reward_pipeline(
                    r.text,
                    ground_truth=gt,
                    context=ctx,
                )
            )
        return torch.tensor(scores, dtype=torch.float32)
