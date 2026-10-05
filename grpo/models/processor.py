from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grpo.config import ProcessorConfig


@dataclass
class VLSample:
    messages: list[dict[str, Any]]
    images: list[Any] | None = None
    metadata: dict[str, Any] | None = None


class VLProcessor:
    def __init__(self, processor: Any, cfg: ProcessorConfig) -> None:
        self.processor = processor
        self.cfg = cfg

    def _messages_with_images(
        self, messages: list[dict[str, Any]], images: list[Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        injected = False
        for msg in messages:
            if not injected and msg["role"] == "user" and isinstance(msg["content"], str):
                content: list[dict[str, Any]] = [
                    {"type": "image", "image": img} for img in images
                ]
                content.append({"type": "text", "text": msg["content"]})
                out.append({**msg, "content": content})
                injected = True
            else:
                out.append(msg)
        return out

    def encode(self, sample: VLSample) -> dict[str, Any]:
        messages = sample.messages
        if sample.images:
            messages = self._messages_with_images(messages, sample.images)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.cfg.add_generation_prompt,
        )

        kwargs: dict[str, Any] = {
            "text": [text],
            "padding": self.cfg.padding,
            "return_tensors": "pt",
        }

        if sample.images:
            kwargs["images"] = sample.images

        return self.processor(**kwargs)
