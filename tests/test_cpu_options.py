"""CPU speed options give the same embeddings (needs the model and consecutive frames; CPU, fp32; slow: ~3 model loads + compile).

Set SCREENRET_TEST_FRAMES to a directory with >= 6 consecutive screenshots (sorted by name), optionally SCREENRET_LORA. Skipped otherwise.
  * --weight-storage bf16-llm-fp32 is bitwise identical to bf16 (LLM weights upcast once instead of per call)
  * --compile matches eager (cos >= 0.99999; same ops, only floating-point reassociation)
"""
import glob
import os

import numpy as np
import pytest
import torch

from screenret import cpu_compile
from screenret.config import PRESETS
from screenret.engine import Engine, preprocess
from screenret.model import load_model

DIR = os.environ.get("SCREENRET_TEST_FRAMES")
pytestmark = pytest.mark.skipif(not DIR, reason="set SCREENRET_TEST_FRAMES to run model tests")
NAMES = ("full+pe", "S", "S+T", "S+T+L")


def run(m, frames, cfg):
    eng = Engine(m, cfg)
    return np.stack([eng.step(preprocess(m, p, cfg), first=(k == 0))[0].float().cpu().numpy() for k, p in enumerate(frames)])


def test_llm_fp32_storage_and_compile():
    frames = sorted(glob.glob(f"{DIR}/*.jpg") + glob.glob(f"{DIR}/*.png"))[:14]
    lora = os.environ.get("SCREENRET_LORA")
    lora = None if lora in (None, "none") else lora
    m = load_model("cpu", torch.float32, lora=lora, weight_storage="bf16")
    ref = {n: run(m, frames, PRESETS[n]) for n in NAMES}
    del m
    m = load_model("cpu", torch.float32, lora=lora, weight_storage="bf16-llm-fp32")
    for n in NAMES:
        assert np.array_equal(run(m, frames, PRESETS[n]), ref[n]), n
    cpu_compile.enable(m)
    for n in NAMES:
        e = run(m, frames, PRESETS[n])
        cos = (e * ref[n]).sum(1) / np.linalg.norm(e, axis=1) / np.linalg.norm(ref[n], axis=1)
        assert cos.min() >= 0.99999, (n, cos.min())
