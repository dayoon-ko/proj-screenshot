"""Command-line flags shared by every entry point: method steps (Token Pruning / Grid Keep / KV Reuse (ViT, LLM) / linear patch
embedding) and model loading."""
from __future__ import annotations

import argparse
import os

import torch

from .config import MODEL_ID, PRESETS, MethodConfig
from .model import WEIGHT_STORAGE, Model, load_model, pick_device, set_mps_memory_cap

BOOL = argparse.BooleanOptionalAction


def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("method components (a --preset sets defaults; explicit flags override it)")
    g.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="full | full+pe | S | S+T | S+T+L | S+G | S+G+T+L (default: S+G+T+L = the paper's method)")
    g.add_argument("--spatial", choices=["none", "lt", "lt-grid", "pixelprune"],
                   help="Token Pruning rule: lt = left-or-upper, lt-grid = with Grid Keep (paper), pixelprune = Pred-2D")
    g.add_argument("--tau", type=float, help="Token Pruning threshold in 8-bit grey levels (default 8 = 8/255)")
    g.add_argument("--temporal-vit", action=BOOL, help="KV Reuse (ViT): cached ViT K/V for unchanged blocks (+ exact LLM prefix reuse)")
    g.add_argument("--delta", type=float, help="KV Reuse change threshold in grey levels (default 8 = 8/255)")
    g.add_argument("--refresh-rho", type=float, help="KV Reuse: full refresh if more than this fraction of kept blocks changed (0.9)")
    g.add_argument("--refresh-every", type=int, help="KV Reuse: full refresh every K frames (default 10)")
    g.add_argument("--approx-llm", action=BOOL,
                   help="KV Reuse (LLM): cached K/V for unchanged visual tokens after the first change (needs --temporal-vit)")
    g.add_argument("--linear-patch-embed", action=BOOL, help="linear patch embedding: Conv3d as the identical F.linear (MPS fix)")


def method_from_args(args: argparse.Namespace) -> MethodConfig:
    cfg = PRESETS[args.preset or "S+G+T+L"]
    over = dict(spatial=args.spatial, tau_levels=args.tau, temporal_vit=args.temporal_vit, delta_levels=args.delta,
                refresh_rho=args.refresh_rho, refresh_every=args.refresh_every, approx_llm=args.approx_llm,
                linear_patch_embed=args.linear_patch_embed)
    return cfg.with_(**{k: v for k, v in over.items() if v is not None})


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (default)")
    g.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="compute dtype (default: float32 on cpu, else float16)")
    g.add_argument("--weight-storage", choices=list(WEIGHT_STORAGE), default="native",
                   help="CPU + fp32 only. bf16: Linear weights stored bf16, upcast per call (~4.3 GB instead of ~8.5 GB); "
                        "bf16-llm-fp32: as bf16 but the LLM weights are upcast once at load (~7 GB, faster; bitwise identical to bf16)")
    g.add_argument("--lora", default=os.environ.get("SCREENRET_LORA"),
                   help="LoRA checkpoint merged at load (paper default: the pixel retrieval LoRA), or 'none' for the zero-shot model. "
                        "Default: $SCREENRET_LORA")
    g.add_argument("--lora-merge", action=BOOL, default=True,
                   help="merge the LoRA into the weights at load (default). --no-lora-merge keeps the low-rank path separate as PEFT "
                        "does; use it with --dtype bfloat16 (merging into bf16 weights rounds away much of the update)")
    g.add_argument("--model-id", default=MODEL_ID, help="HF id or local path of Qwen3-VL-Embedding-2B")
    g.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    g.add_argument("--mps-cap-gb", type=float, default=None, help="cap MPS allocations of this process (GB)")


def add_cpu_speed_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("CPU speed (off by default)")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile (dynamic shapes) of the dense per-layer parts of the ViT / KV Reuse (LLM) and of the exact LLM "
                        "forward; attention, K/V cache I/O and host planning stay eager")


def load_from_args(args: argparse.Namespace) -> Model:
    if args.lora is None:
        raise SystemExit("--lora is required: pass the pixel retrieval LoRA checkpoint (paper default) or 'none' for the zero-shot model "
                         "(or set $SCREENRET_LORA)")
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    if device == "mps" and args.mps_cap_gb:
        set_mps_memory_cap(args.mps_cap_gb)
    dtype = getattr(torch, args.dtype or ("float32" if device == "cpu" else "float16"))
    lora = None if args.lora.lower() in ("none", "") else args.lora
    m = load_model(device, dtype, lora=lora, weight_storage=args.weight_storage, model_id=args.model_id, lora_merge=args.lora_merge)
    if getattr(args, "compile", False):
        from . import cpu_compile
        cpu_compile.enable(m)
    return m


def model_meta(args: argparse.Namespace, m: Model) -> dict:
    """Run metadata stored with every output (no personal paths: only the LoRA file name)."""
    return dict(device=str(m.device), dtype=str(m.dtype).split(".")[-1], weight_storage=args.weight_storage, model_id=args.model_id,
                lora=os.path.basename(args.lora) if args.lora and args.lora.lower() != "none" else "none", lora_merge=args.lora_merge,
                compile=getattr(args, "compile", False), torch=torch.__version__, threads=torch.get_num_threads())
