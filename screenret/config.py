"""Method settings: which efficiency components are on, their thresholds, and named presets.

Pixel values from the Qwen3-VL processor are normalised to [-1, 1], so one 8-bit grey level is 1/127.5. Thresholds are given in grey
levels on the command line (tau 8, delta 8) and converted with LEVEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

MODEL_ID = os.environ.get("SCREENRET_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
LEVEL = 1 / 127.5                       # one 8-bit grey level in processor units
QUERY_INSTRUCTION = "Retrieve the computer screen that matches the user's description."
MAX_OCR_CHARS = 12000                   # OCR text is cut to this many characters before text embedding


@dataclass(frozen=True)
class MethodConfig:
    """Efficiency components of the screenshot route.

    spatial             Token Pruning rule: "none" | "lt" (drop a block equal within tau to its left OR upper block) | "lt-grid"
                        (paper: lt + Grid Keep, one block per flat region per 2x2-block tile kept) | "pixelprune" (Pred-2D)
    tau_levels          Token Pruning threshold in grey levels (8 = 8/255)
    temporal_vit        KV Reuse (ViT): recompute only changed blocks in the ViT (cached K/V for the rest); implies exact LLM prefix reuse
    delta_levels        KV Reuse change threshold in grey levels (block changed if max |pixel - pixel at its last recompute| > delta)
    refresh_rho         KV Reuse: full refresh when more than this fraction of the kept blocks changed ...
    refresh_every       ... or every K-th frame since the last refresh
    approx_llm          KV Reuse (LLM): recompute only changed visual tokens + text after the first change (needs temporal_vit)
    linear_patch_embed  linear patch embedding ("+pe"): ViT patch embedding as F.linear instead of Conv3d (identical math; MPS Conv3d
                        is very slow)
    """
    spatial: str = "none"
    tau_levels: float = 8.0
    temporal_vit: bool = False
    delta_levels: float = 8.0
    refresh_rho: float = 0.9
    refresh_every: int = 10
    approx_llm: bool = False
    linear_patch_embed: bool = False

    def __post_init__(self):
        assert self.spatial in ("none", "lt", "lt-grid", "pixelprune"), self.spatial
        assert self.temporal_vit or not self.approx_llm, "KV Reuse (LLM) reuses per-block state that only KV Reuse (ViT) keeps"

    @property
    def tau(self) -> float:
        return self.tau_levels * LEVEL

    @property
    def delta(self) -> float:
        return self.delta_levels * LEVEL

    @property
    def name(self) -> str:
        """Short tag, e.g. LT8t2TL+pe (Token Pruning rule + tau, t2 = Grid Keep, T = KV Reuse (ViT), L = KV Reuse (LLM), +pe = linear
        patch embedding)."""
        s = {"none": "FULL", "lt": f"LT{self.tau_levels:g}", "lt-grid": f"LT{self.tau_levels:g}t2",
             "pixelprune": f"PP{self.tau_levels:g}"}[self.spatial]
        if self.spatial == "none" and self.temporal_vit:
            s = ""
        s += ("T" if self.temporal_vit else "") + ("L" if self.approx_llm else "")
        if self.temporal_vit and (self.refresh_rho, self.refresh_every, self.delta_levels) != (0.9, 10, 8.0):
            s += f"_r{self.refresh_rho:g}k{self.refresh_every}d{self.delta_levels:g}"
        return s + ("+pe" if self.linear_patch_embed else "")

    def with_(self, **kw) -> "MethodConfig":
        return replace(self, **kw)


# Named presets (paper rows). Explicit CLI flags override preset values.
PRESETS: dict[str, MethodConfig] = {
    "full": MethodConfig(),                                                             # Full screenshot, stock Conv3d patch embedding
    "full+pe": MethodConfig(linear_patch_embed=True),                                   # Full screenshot, linear patch embedding
    "S": MethodConfig(spatial="lt", linear_patch_embed=True),                           # Token Pruning
    "S+T": MethodConfig(spatial="lt", temporal_vit=True, linear_patch_embed=True),      # + KV Reuse (ViT) (+ exact LLM prefix reuse)
    "S+T+L": MethodConfig(spatial="lt", temporal_vit=True, approx_llm=True, linear_patch_embed=True),   # + KV Reuse (LLM)
    "S+G": MethodConfig(spatial="lt-grid", linear_patch_embed=True),                    # Token Pruning with Grid Keep
    "S+G+T+L": MethodConfig(spatial="lt-grid", temporal_vit=True, approx_llm=True, linear_patch_embed=True),   # paper method
}
