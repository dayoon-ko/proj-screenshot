"""Model loading (+ LoRA, optional bf16 / bf16-llm-fp32 weight storage), the linear patch embedding and per-resolution constants.

Qwen3-VL-Embedding-2B = ViT (16x16-px patches) + LLM; the embedding is the normalised final hidden state of the last token.
A 1280x704 screenshot gives 80x44 patches; the ViT merger fuses each 2x2 group of patches (= one 32x32-px "block") into one LLM token,
so 880 blocks / visual tokens. The processor already orders patches block by block (4 consecutive rows = one block).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as MQ

from .config import MODEL_ID
from .qwen_embedder import Qwen3VLEmbedder


@dataclass
class Model:
    embedder: Qwen3VLEmbedder           # processor + HF model; also the text embedder of the OCR route
    device: torch.device
    dtype: torch.dtype
    grids: dict = field(default_factory=dict)   # (grid_thw, prompt length) -> Grid

    def __post_init__(self):
        base = self.embedder.model
        self.core = base.model                      # HF Qwen3VLModel
        self.vit = self.core.visual
        self.llm = self.core.language_model
        self.vcfg, self.tcfg = self.vit.config, self.llm.config
        self.image_token_id = base.config.image_token_id
        self.vit_heads = self.vcfg.num_heads
        self.vit_head_dim = self.vcfg.hidden_size // self.vcfg.num_heads
        self.deepstack_layers = list(self.vit.deepstack_visual_indexes)   # ViT layers whose output is added to early LLM layers
        pe = self.vit.patch_embed
        # changed: Conv3d (kernel == stride == patch) flattened once into a matrix for F.linear (see embed_patches)
        self.patch_w = pe.proj.weight.detach().reshape(pe.embed_dim, -1)
        self.patch_b = pe.proj.bias.detach()

    def sync(self):
        """Wait for the device so wall-clock timings cover the device work."""
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_mps_memory_cap(cap_gb: float) -> float:
    """Cap this process's MPS allocations (useful on a shared 16 GB Mac). Returns the fraction set."""
    frac = min(1.0, cap_gb * 2 ** 30 / torch.mps.recommended_max_memory())
    torch.mps.set_per_process_memory_fraction(frac)
    return frac


WEIGHT_STORAGE = ("native", "bf16", "bf16-llm-fp32")


def load_model(device: str = "auto", dtype: torch.dtype = torch.float16, lora: Optional[str] = None, weight_storage: str = "native",
               model_id: str = MODEL_ID, lora_merge: bool = True) -> Model:
    """Load Qwen3-VL-Embedding-2B on the CPU, add an optional LoRA, then move it to `device`.

    lora: PEFT LoRA state dict (.pt); None = zero-shot.
    lora_merge=True: W += B @ A at load, so the compute graph equals the base model (paper default, fp16 / fp32).
    lora_merge=False: keep the low-rank path separate, x W^T + (x A^T) B^T, as PEFT computes it (fp32 adapter path). Use this in bf16:
    rounding W + B @ A to bf16 (8-bit mantissa) loses much of the low-rank update.
    weight_storage="bf16" (CPU + float32 only): keep Linear weights in the checkpoint's bf16, compute in fp32 (store_linear_weights_bf16).
    weight_storage="bf16-llm-fp32": as "bf16", then the LLM decoder-layer weights are upcast to fp32 once (ViT and the input-embedding
    table stay bf16): no per-call upcast in the LLM, bitwise identical outputs, ~7 GB instead of ~4.3 GB.
    """
    assert weight_storage in WEIGHT_STORAGE, weight_storage
    device = pick_device(device)
    bf16_store = weight_storage in ("bf16", "bf16-llm-fp32")
    if bf16_store:
        assert device == "cpu" and dtype == torch.float32, "bf16 weight storage is the CPU fp32 memory saver"
    embedder = Qwen3VLEmbedder(model_id, dtype=torch.bfloat16 if bf16_store else dtype)
    if lora and lora_merge and embedder.model.dtype == torch.bfloat16:
        warnings.warn("merging the LoRA into bf16 weights rounds away part of the update; consider lora_merge=False (--no-lora-merge)")
    if lora:
        (merge_lora if lora_merge else attach_lora)(embedder.model, lora)
    embedder.model.to(device).eval()
    if bf16_store:
        store_linear_weights_bf16(embedder.model)
    if weight_storage == "bf16-llm-fp32":                  # changed: LLM weights upcast once at load (no per-call bf16 -> fp32 copy)
        for p in embedder.model.model.language_model.layers.parameters():
            p.data = p.data.float()
    return Model(embedder=embedder, device=torch.device(device), dtype=dtype)


def merge_lora(model: torch.nn.Module, path: str, scale: float = 1.0) -> int:
    """Merge a PEFT LoRA state dict (alpha / r = 1 for the paper's r 32 / alpha 32) into the base weights: W += scale * B @ A in fp32.

    Keys look like base_model.model.<module path>.lora_A.weight; the module path is relative to Qwen3VLForEmbedding.
    """
    pairs = _lora_pairs(path)
    for module, a, b in pairs:
        w = model.get_submodule(module).weight
        w.data = (w.data.float() + scale * (b.float() @ a.float())).to(w.dtype)
    return len(pairs)


def _lora_pairs(path: str):
    """(module path, A, B) per LoRA module of a PEFT state dict (keys base_model.model.<module path>.lora_A.weight)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    pairs = [(key[len("base_model.model."):-len(".lora_A.weight")], a, state[key.replace(".lora_A.", ".lora_B.")])
             for key, a in state.items() if key.endswith(".lora_A.weight")]
    assert pairs and len(pairs) == len(state) // 2, f"unexpected LoRA state dict {path}"
    return pairs


class LoRALinear(torch.nn.Module):
    """nn.Linear + separate low-rank update, computed as PEFT does: A, B stay fp32 (PEFT upcasts bf16 / fp16 adapters), the input is
    cast to fp32 for the low-rank path, base(x) + B(A(x)) * scale is summed in fp32 and cast back to the base output dtype.
    dropout (training only, PEFT's lora_dropout) acts on the fp32 input of the low-rank path; it is the identity in eval mode."""

    def __init__(self, base: torch.nn.Linear, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base, self.scale = base, scale
        self.lora_a = torch.nn.Parameter(a.float(), requires_grad=False)
        self.lora_b = torch.nn.Parameter(b.float(), requires_grad=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()

    @property
    def weight(self):                   # expose the base weight (dtype / shape lookups by callers)
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        return (y + F.linear(F.linear(self.dropout(x.to(self.lora_a.dtype)), self.lora_a), self.lora_b) * self.scale).to(y.dtype)


def attach_lora(model: torch.nn.Module, path: str, scale: float = 1.0, dropout: float = 0.0) -> int:
    """Unmerged LoRA: replace each target nn.Linear by a LoRALinear (alpha / r = 1 for the paper's r 32 / alpha 32)."""
    pairs = _lora_pairs(path)
    for module, a, b in pairs:
        parent, _, name = module.rpartition(".")
        setattr(model.get_submodule(parent), name, LoRALinear(model.get_submodule(module), a, b, scale, dropout))
    return len(pairs)


def lora_state(model: torch.nn.Module) -> dict:
    """PEFT-format state dict (base_model.model.<module path>.lora_A / lora_B.weight) of the attached LoRALinear modules: the file
    format every --lora option reads."""
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            state[f"base_model.model.{name}.lora_A.weight"] = mod.lora_a.detach().cpu().clone()
            state[f"base_model.model.{name}.lora_B.weight"] = mod.lora_b.detach().cpu().clone()
    return state


def store_linear_weights_bf16(model: torch.nn.Module) -> None:
    """CPU memory saver: nn.Linear weights (and the token-embedding table) stay bf16 and are upcast per call; everything else is fp32.

    bf16 -> fp32 is exact, so outputs equal an fp32 model loaded from the bf16 checkpoint, at ~4.3 GB instead of ~8.5 GB resident.
    changed: patches nn.Linear / nn.Embedding forward for the whole process (the upcast is inside every timed forward).
    """
    nn = torch.nn
    for mod in model.modules():
        keep_bf16 = isinstance(mod, nn.Linear) or (isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 100000)
        for name, p in mod.named_parameters(recurse=False):
            if p.is_floating_point() and not (keep_bf16 and name == "weight"):
                p.data = p.data.float()
        for name, b in mod.named_buffers(recurse=False):
            if b.is_floating_point():
                setattr(mod, name, b.float())
    embedding_forward = nn.Embedding.forward

    def linear_upcast(self, x):
        w = self.weight
        return F.linear(x, w if w.dtype == x.dtype else w.to(x.dtype), self.bias)

    nn.Linear.forward = linear_upcast
    nn.Embedding.forward = lambda self, x: embedding_forward(self, x).float()


def embed_patches(m: Model, pixels: torch.Tensor, linear: bool) -> torch.Tensor:
    """ViT patch embedding of flattened patches (n, 3*2*16*16)."""
    if linear:
        # changed: Conv3d patch embed -> F.linear (identical math; MPS Conv3d runs at ~0.02 TFLOP/s)
        return F.linear(pixels.reshape(-1, m.patch_w.shape[1]).to(m.patch_w.dtype), m.patch_w, m.patch_b)
    pe = m.vit.patch_embed
    return type(pe).forward(pe, pixels)        # stock HF Conv3d forward


@dataclass
class Grid:
    """Constants of one (resolution, prompt) pair; identical for every frame of a stream, so computed once and cached."""
    key: tuple
    input_ids: torch.Tensor
    vit_pos: torch.Tensor           # learned ViT position embedding per patch
    vit_cos: torch.Tensor           # ViT 2-D RoPE per patch
    vit_sin: torch.Tensor
    token_emb: torch.Tensor         # (L_full, H) LLM input embeddings of the prompt (image slots are overwritten by ViT features)
    llm_pos: torch.Tensor           # (rows, L_full) mRoPE position ids of the full-grid prompt
    first: int                      # index of the first / last image token in the full-grid prompt
    last: int
    n_blocks: int                   # 880 for 1280x704
    n_patches: int                  # 4 * n_blocks
    length: int                     # L_full = prompt length with all blocks

    @property
    def text_tail(self) -> int:
        """Prompt tokens after the image (instruction end + final token)."""
        return self.length - self.last - 1

    def llm_len(self, n_kept: int) -> int:
        """LLM sequence length when only n_kept blocks are fed (Token Pruning drops the rest)."""
        return self.first + n_kept + self.text_tail


@torch.inference_mode()
def grid(m: Model, inputs: dict) -> Grid:
    """Position embeddings / RoPE / prompt embeddings, computed once per resolution.
    changed: hoisted out of the per-frame forward (stock HF recomputes them every call)."""
    thw, ids = inputs["image_grid_thw"], inputs["input_ids"]
    key = (tuple(thw[0].tolist()), ids.shape[1])
    g = m.grids.get(key)
    if g is not None:
        assert torch.equal(g.input_ids, ids), "same resolution and prompt length must mean the same prompt"
        return g
    V, cv = m.vit, m.vcfg
    ii, iw = MQ.get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=V.num_grid_per_side, mode=V.interpolation_mode,
                                                             align_corners=V.interpolation_align_corners,
                                                             spatial_merge_size=cv.spatial_merge_size, kwargs={})
    pid = MQ.get_vision_position_ids(thw, V.spatial_merge_size, kwargs={})
    pos = (V.pos_embed(ii) * iw[:, :, None]).sum(1)
    cos, sin = V.rotary_pos_emb(pos, pid)
    mm = inputs.get("mm_token_type_ids", (ids == m.image_token_id).int())
    x0 = m.core.get_input_embeddings()(ids)
    m.core.rope_deltas = None
    p3 = m.core.compute_3d_position_ids(input_ids=ids, inputs_embeds=x0, image_grid_thw=thw, attention_mask=inputs["attention_mask"],
                                        mm_token_type_ids=mm)
    ipos = (ids[0] == m.image_token_id).nonzero()[:, 0]
    first, last, n = int(ipos[0]), int(ipos[-1]), len(ipos)
    assert last - first + 1 == n and int(thw.prod()) == 4 * n, "expected one contiguous run of image tokens, 4 patches each"
    g = Grid(key=key, input_ids=ids, vit_pos=pos, vit_cos=cos, vit_sin=sin, token_emb=x0[0], llm_pos=p3[:, 0],
             first=first, last=last, n_blocks=n, n_patches=4 * n, length=ids.shape[1])
    m.grids[key] = g
    return g
