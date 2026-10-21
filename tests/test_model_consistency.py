"""Consistency of the method on real screenshots (needs the model and a few consecutive frames).

Set SCREENRET_TEST_FRAMES to a directory with >= 6 consecutive screenshots (sorted by name), optionally SCREENRET_TEST_DEVICE (default
cpu, float32) and SCREENRET_LORA. Skipped otherwise. Checks properties that must hold for any input:
  * +pe is the same math as the Conv3d patch embedding
  * T_vit / T_llm with a refresh on every frame reduce to the exact computation (S alone)
  * MAC counts: FULL frame = full_frame_macs, and S+T+L <= S+T <= S <= FULL in total
"""
import glob
import os

import numpy as np
import pytest
import torch

from screenret.config import PRESETS
from screenret.engine import Engine, preprocess
from screenret.macs import full_frame_macs
from screenret.model import load_model

DIR = os.environ.get("SCREENRET_TEST_FRAMES")
pytestmark = pytest.mark.skipif(not DIR, reason="set SCREENRET_TEST_FRAMES to run model tests")


@pytest.fixture(scope="module")
def setup():
    dev = os.environ.get("SCREENRET_TEST_DEVICE", "cpu")
    lora = os.environ.get("SCREENRET_LORA")
    m = load_model(dev, torch.float32 if dev == "cpu" else torch.float16, lora=None if lora in (None, "none") else lora,
                   weight_storage="bf16" if dev == "cpu" else "native")
    frames = sorted(glob.glob(f"{DIR}/*.jpg") + glob.glob(f"{DIR}/*.png"))[:6]
    assert len(frames) >= 2
    return m, frames


def run(m, frames, cfg):
    eng = Engine(m, cfg)
    out = [eng.step(preprocess(m, p, cfg), first=(k == 0)) for k, p in enumerate(frames)]
    return np.stack([e.float().cpu().numpy() for e, _ in out]), [r for _, r in out]


def cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def test_linear_patch_embed_is_identical_math(setup):
    m, frames = setup
    a, _ = run(m, frames[:2], PRESETS["full"])
    b, _ = run(m, frames[:2], PRESETS["full+pe"])
    assert cos(a, b).min() > 0.9999


def test_refresh_every_frame_equals_exact(setup):
    m, frames = setup
    s, _ = run(m, frames, PRESETS["S"])
    st, rows_t = run(m, frames, PRESETS["S+T"].with_(refresh_every=1))
    stl, rows_l = run(m, frames, PRESETS["S+T+L"].with_(refresh_every=1))
    assert all(r["full"] for r in rows_t + rows_l)
    assert cos(s, st).min() > 0.9999 and cos(s, stl).min() > 0.9999


def test_macs_order(setup):
    m, frames = setup
    tot = {}
    for name in ("full+pe", "S", "S+T", "S+T+L"):
        _, rows = run(m, frames, PRESETS[name])
        tot[name] = sum(r["vit_macs"] + r["llm_macs"] for r in rows)
    g = next(iter(m.grids.values()))
    assert tot["full+pe"] == len(frames) * full_frame_macs(m, g)
    assert tot["S+T+L"] <= tot["S+T"] <= tot["S"] <= tot["full+pe"]
