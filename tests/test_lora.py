"""Merged and unmerged LoRA compute the same function (fp32); bf16 merging is lossy; unmerged = PEFT (toy layers, no model needed)."""
import pytest
import torch

from screenret.model import attach_lora, merge_lora


def toy(dtype):
    torch.manual_seed(0)
    m = torch.nn.Sequential()
    m.add_module("proj", torch.nn.Linear(256, 256).to(dtype))
    return m


def lora_file(tmp_path):
    torch.manual_seed(1)
    state = {"base_model.model.proj.lora_A.weight": torch.randn(32, 256) * 0.01,
             "base_model.model.proj.lora_B.weight": torch.randn(256, 32) * 0.01}
    path = tmp_path / "lora.pt"
    torch.save(state, path)
    return str(path), state


@torch.no_grad()
def test_merged_equals_unmerged_fp32(tmp_path):
    path, _ = lora_file(tmp_path)
    x = torch.randn(8, 256)
    a, b = toy(torch.float32), toy(torch.float32)
    assert merge_lora(a, path) == 1 and attach_lora(b, path) == 1
    assert torch.allclose(a(x), b(x), atol=1e-5)


@torch.no_grad()
def test_bf16_merge_loses_part_of_the_update(tmp_path):
    path, st = lora_file(tmp_path)
    x = torch.randn(64, 256).bfloat16()
    delta_true = x.float() @ (st["base_model.model.proj.lora_B.weight"] @ st["base_model.model.proj.lora_A.weight"]).T
    base, merged, unmerged = toy(torch.bfloat16), toy(torch.bfloat16), toy(torch.bfloat16)
    merge_lora(merged, path)
    attach_lora(unmerged, path)
    err = lambda m: float(((m(x).float() - base(x).float()) - delta_true).norm() / delta_true.norm())   # noqa: E731
    assert err(unmerged) < err(merged)          # seeded toy: 0.10 vs 0.17 relative error of the LoRA update


@torch.no_grad()
def test_unmerged_equals_peft_bf16(tmp_path):
    peft = pytest.importorskip("peft")
    path, st = lora_file(tmp_path)
    x = torch.randn(16, 256).bfloat16()
    ours, ref = toy(torch.bfloat16), toy(torch.bfloat16)
    attach_lora(ours, path)
    ref = peft.get_peft_model(ref, peft.LoraConfig(r=32, lora_alpha=32, target_modules=["proj"])).eval()
    peft.set_peft_model_state_dict(ref, st)
    assert torch.equal(ours(x), ref(x))
