"""Presets and flag overrides (no model needed)."""
import argparse

import pytest

from screenret.cli import add_method_args, add_model_args, method_from_args
from screenret.config import MethodConfig


def parse(*argv):
    p = argparse.ArgumentParser()
    add_method_args(p)
    add_model_args(p)
    return method_from_args(p.parse_args(list(argv)))


def test_default_is_full_method():
    c = parse()
    assert (c.spatial, c.temporal_vit, c.approx_llm, c.linear_patch_embed) == ("lt-grid", True, True, True)
    assert (c.tau_levels, c.delta_levels, c.refresh_rho, c.refresh_every) == (8, 8, 0.9, 10)
    assert c.name == "LT8t2TL+pe"


@pytest.mark.parametrize("preset,name", [("full", "FULL"), ("full+pe", "FULL+pe"), ("S", "LT8+pe"), ("S+T", "LT8T+pe"),
                                         ("S+T+L", "LT8TL+pe"), ("S+G", "LT8t2+pe"), ("S+G+T+L", "LT8t2TL+pe")])
def test_preset_names(preset, name):
    assert parse("--preset", preset).name == name


def test_flags_override_preset():
    c = parse("--preset", "S+T+L", "--spatial", "pixelprune", "--no-linear-patch-embed", "--refresh-rho", "0.6")
    assert c.spatial == "pixelprune" and not c.linear_patch_embed and c.refresh_rho == 0.6 and c.name == "PP8TL_r0.6k10d8"


def test_t_llm_needs_t_vit():
    with pytest.raises(AssertionError):
        MethodConfig(approx_llm=True)
