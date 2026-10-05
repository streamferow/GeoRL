from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
}


@dataclass
class ModelConfig:
    model_id: str
    dtype: torch.dtype
    device_map: str
    trust_remote_code: bool
    cache_dir: str


@dataclass
class ProcessorConfig:
    min_pixels: int | None
    max_pixels: int | None
    padding: bool
    add_generation_prompt: bool


@dataclass
class RolloutConfig:
    num_generations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    skip_special_tokens: bool


@dataclass
class GRPOConfig:
    learning_rate: float
    clip_eps: float
    kl_coef: float


@dataclass
class Config:
    model: ModelConfig
    processor: ProcessorConfig
    rollout: RolloutConfig
    grpo: GRPOConfig

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        config_path = Path(path) if path else CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text())

        model_raw = raw["model"]
        processor_raw = raw["processor"]
        rollout_raw = raw["rollout"]
        grpo_raw = raw["grpo"]

        return cls(
            model=ModelConfig(
                model_id=model_raw["model_id"],
                dtype=_DTYPE_MAP[model_raw["dtype"].lower()],
                device_map=model_raw["device_map"],
                trust_remote_code=model_raw["trust_remote_code"],
                cache_dir=str(Path(model_raw["cache_dir"]).expanduser()),
            ),
            processor=ProcessorConfig(
                min_pixels=processor_raw.get("min_pixels"),
                max_pixels=processor_raw.get("max_pixels"),
                padding=processor_raw["padding"],
                add_generation_prompt=processor_raw["add_generation_prompt"],
            ),
            rollout=RolloutConfig(
                num_generations=rollout_raw["num_generations"],
                max_new_tokens=rollout_raw["max_new_tokens"],
                temperature=rollout_raw["temperature"],
                top_p=rollout_raw["top_p"],
                top_k=rollout_raw["top_k"],
                skip_special_tokens=rollout_raw["skip_special_tokens"],
            ),
            grpo=GRPOConfig(
                learning_rate=grpo_raw["learning_rate"],
                clip_eps=grpo_raw["clip_eps"],
                kl_coef=grpo_raw["kl_coef"],
            ),
        )
