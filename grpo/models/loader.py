from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from grpo.config import ModelConfig, ProcessorConfig


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    device: torch.device


def _ensure_hf_auth() -> None:
    load_dotenv()


def load_qwen_vl(model_cfg: ModelConfig, processor_cfg: ProcessorConfig) -> ModelBundle:
    _ensure_hf_auth()

    processor_kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.trust_remote_code,
        "cache_dir": model_cfg.cache_dir,
    }
    if processor_cfg.min_pixels is not None:
        processor_kwargs["min_pixels"] = processor_cfg.min_pixels
    if processor_cfg.max_pixels is not None:
        processor_kwargs["max_pixels"] = processor_cfg.max_pixels

    processor = AutoProcessor.from_pretrained(model_cfg.model_id, **processor_kwargs)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_cfg.model_id,
        torch_dtype=model_cfg.dtype,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        cache_dir=model_cfg.cache_dir,
    )
    device = next(model.parameters()).device
    return ModelBundle(model=model, processor=processor, device=device)
