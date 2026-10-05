from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from grpo.config import RolloutConfig
from grpo.models.processor import VLSample, VLProcessor


@dataclass
class Rollout:
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    logprobs: torch.Tensor
    text: str
    metadata: dict[str, Any]


class RolloutGenerator:
    def __init__(
        self,
        model: Any,
        processor: VLProcessor,
        cfg: RolloutConfig,
    ) -> None:
        self.model = model
        self.processor = processor
        self.cfg = cfg

    def _generation_config(self, *, seed: int) -> GenerationConfig:
        common = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            seed=seed,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if self.cfg.temperature > 0:
            return GenerationConfig(
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                **common,
            )
        return GenerationConfig(do_sample=False, **common)

    @torch.no_grad()
    def generate_group(self, sample: VLSample) -> list[Rollout]:
        device = next(self.model.parameters()).device
        inputs = self.processor.encode(sample)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[-1]
        prompt_ids = inputs["input_ids"][0]
        hf_processor = self.processor.processor
        base_metadata = dict(sample.metadata or {})
        rollouts: list[Rollout] = []

        for i in range(self.cfg.num_generations):
            seed = secrets.randbits(63)
            outputs = self.model.generate(
                **inputs,
                generation_config=self._generation_config(seed=seed),
            )
            completion_ids = outputs.sequences[0, prompt_len:]
            scores = torch.stack(outputs.scores, dim=1)[0]
            logprobs = F.log_softmax(scores, dim=-1).gather(
                1, completion_ids.unsqueeze(-1)
            ).squeeze(-1)
            text = hf_processor.batch_decode(
                completion_ids.unsqueeze(0),
                skip_special_tokens=self.cfg.skip_special_tokens,
            )[0].strip()
            rollouts.append(
                Rollout(
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    text=text,
                    metadata={**base_metadata, "generation_idx": i, "seed": seed},
                )
            )

        return rollouts
