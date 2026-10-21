"""Optional torch.compile of the dense per-layer parts (`--compile`; a CPU speed option, off by default).

What is compiled (inductor, dynamic shapes, so a new token count does not recompile):
  - ViT, every block: norm1 -> qkv -> RoPE, and proj + residual -> norm2 -> MLP + residual (vit.KERNELS)
  - LLM, KV Reuse (LLM) path: input norm -> q/k/v -> q/k norm -> mRoPE, and o_proj + residual -> norm -> MLP + residual (llm.KERNELS)
  - LLM, exact path (full screenshot / Token Pruning / + KV Reuse (ViT)): the whole HF language-model forward
Attention, the K/V cache reads / writes and all host-side planning stay eager. Linear layers still call the same BLAS matmuls; the gain
comes from fusing the element-wise work around them (norms, RoPE, SiLU * up, residual adds) and fewer dispatches. Same ops, same order:
outputs match eager up to floating-point reassociation (cos >= 0.99999).
The first frames compile (tens of seconds; untimed warm-up in screenret.timing). Process-wide: the KERNELS tables are swapped in place.
macOS Command Line Tools without libc++ headers next to clang: export CPLUS_INCLUDE_PATH=<SDK>/usr/include/c++/v1 (see README).
"""
from __future__ import annotations

import torch

from . import llm, vit
from .model import Model


def enable(m: Model) -> None:
    torch._dynamo.config.cache_size_limit = 256          # one graph serves all blocks of a stack (modules are inlined as inputs)
    compile_ = lambda f: torch.compile(f, dynamic=True)  # noqa: E731
    for table in (vit.KERNELS, llm.KERNELS):
        for name, fn in list(table.items()):
            table[name] = compile_(fn)
    m.llm.forward = compile_(m.llm.forward)
