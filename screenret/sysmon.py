"""Machine-state readings logged next to the timings (to document contention). macOS readings degrade to empty values elsewhere."""
from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys

import psutil
import torch

IS_MAC = sys.platform == "darwin"
_PS = psutil.Process()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def vm_stat() -> dict:
    out = {}
    if IS_MAC:
        for line in _run(["vm_stat"]).splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                out[k.strip()] = int(v) if v.isdigit() else 0
    return out


def swaps() -> int:
    """Cumulative swap-ins + swap-outs (macOS) -- differences between two readings flag swapping during a frame."""
    d = vm_stat()
    return d.get("Swapins", 0) + d.get("Swapouts", 0)


def avail_gb() -> float:
    """Memory the OS can hand out without swapping (macOS: free + inactive + speculative pages)."""
    if IS_MAC:
        d = vm_stat()
        return (d.get("Pages free", 0) + d.get("Pages inactive", 0) + d.get("Pages speculative", 0)) * 16384 / 2 ** 30
    return psutil.virtual_memory().available / 2 ** 30


def battery() -> dict:
    """Battery level and power source (a nearly empty battery throttles a laptop)."""
    b = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    return dict(battery_pct=round(b.percent) if b else -1, power=("AC" if b.power_plugged else "Battery") if b else "n/a")


def memory(device: str) -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if IS_MAC else 2 ** 20)   # bytes on macOS, KB on Linux
    r = dict(rss_gb=_PS.memory_info().rss / 2 ** 30, rss_peak_gb=peak)
    if device == "mps":
        r.update(mps_alloc_gb=torch.mps.current_allocated_memory() / 2 ** 30, mps_driver_gb=torch.mps.driver_allocated_memory() / 2 ** 30)
    elif device.startswith("cuda"):
        r.update(cuda_alloc_gb=torch.cuda.memory_allocated() / 2 ** 30, cuda_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    return r


def top_procs(n: int = 5) -> list[str]:
    """Top other processes by %CPU."""
    lines = _run(["ps", "-axo", "pid,%cpu,comm", "-r"] if IS_MAC else ["ps", "-eo", "pid,%cpu,comm", "--sort=-%cpu"]).splitlines()[1:n + 2]
    return [ln.strip()[:90] for ln in lines if ln.split()[:1] != [str(os.getpid())]][:n]


def environment() -> dict:
    return dict(load=os.getloadavg(), avail_gb=avail_gb(), top_procs=top_procs(), platform=platform.platform(),
                machine=platform.machine(), python=platform.python_version(), macos=platform.mac_ver()[0] if IS_MAC else None)
