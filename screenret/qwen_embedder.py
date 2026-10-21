"""Minimal Qwen3-VL-Embedding wrapper: processor + model + last-token pooling, images and text only.

Trimmed from Qwen's reference `qwen3_vl_embedding.py` (Apache-2.0), which ships with Qwen3-VL-Embedding. Kept identical where it matters
for the vectors: chat formatting, pixel limits, `process_vision_info(image_patch_size=16)`, processor call arguments, last-token pooling.
Removed: video inputs, token truncation helper, the automatic move to CUDA in __init__ (the caller picks the device).
The module tree is unchanged (`Qwen3VLForEmbedding.model` = HF `Qwen3VLModel`), so LoRA state-dict keys of the reference resolve as is.
"""
import unicodedata
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLPreTrainedModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

MAX_LENGTH = 8192
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = "Represent the user's input."


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3VLModel without an LM head; the embedding is the final hidden state of the last token."""
    _checkpoint_conversion_mapping = {}      # (no `from __future__ import annotations` here: transformers reads `config`'s class)
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(self, **inputs):
        return self.model(**inputs)


class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path: str, dtype: torch.dtype = torch.float32, max_length: int = MAX_LENGTH,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
        self.max_length, self.min_pixels, self.max_pixels = max_length, min_pixels, max_pixels
        self.model = Qwen3VLForEmbedding.from_pretrained(model_name_or_path, dtype=dtype).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")

    def without_model(self) -> "Qwen3VLEmbedder":
        """Processor-only shallow copy (cheap to send to DataLoader workers started with spawn, e.g. on macOS)."""
        light = object.__new__(Qwen3VLEmbedder)
        light.__dict__.update({k: v for k, v in self.__dict__.items() if k != "model"})
        return light

    def format_model_input(self, text: Optional[str] = None, image: Optional[str] = None, instruction: Optional[str] = None) -> list:
        """One chat conversation: system = instruction, user = [image] + [text]."""
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction += "."
        content = []
        if image is not None:
            content.append({"type": "image", "image": image if image.startswith(("http://", "https://")) else "file://" + image,
                            "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})
        if text is not None:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [{"role": "system", "content": [{"type": "text", "text": instruction or DEFAULT_INSTRUCTION}]},
                {"role": "user", "content": content}]

    def preprocess(self, conversations: List[list]) -> dict:
        """Conversations -> processor tensors (CPU)."""
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, _, video_kwargs = process_vision_info(conversations, image_patch_size=16, return_video_metadata=True,
                                                      return_video_kwargs=True)
        return self.processor(text=text, images=images, videos=None, video_metadata=None, truncation=True, max_length=self.max_length,
                              padding=True, do_resize=False, return_tensors="pt", **video_kwargs)

    @staticmethod
    def pool_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Hidden state of the last non-padding token of each row (right padding)."""
        col = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), col]

    @torch.inference_mode()
    def embed(self, conversations: List[list], normalize: bool = True) -> torch.Tensor:
        """Stock (unmodified HF) forward of a batch of conversations -> (B, 2048) embeddings."""
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in self.preprocess(conversations).items()}
        hidden = self.model.model(**inputs).last_hidden_state
        e = self.pool_last(hidden, inputs["attention_mask"])
        return F.normalize(e, p=2, dim=-1) if normalize else e

    def embed_texts(self, texts: List[str], instruction: Optional[str] = None) -> torch.Tensor:
        return self.embed([self.format_model_input(text=t, instruction=instruction) for t in texts])

    def process(self, items: List[Dict[str, Any]]) -> torch.Tensor:
        """Compatibility with the reference API: items = [{"text"|"image"|"instruction": ...}]."""
        return self.embed([self.format_model_input(text=i.get("text"), image=i.get("image"), instruction=i.get("instruction"))
                           for i in items])
