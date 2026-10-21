"""Query-time cost vs corpus size: (1) text-query encoding (batch 1, same embedder) and (2) exact top-k search over N stored vectors.

Search latency depends only on N, dim and dtype, not on content, so the index is filled with random unit vectors. N spans one pool to
~1 year of 5-s screenshots at 8 h/day (2.1M).

example: python scripts/query_bench.py --device mps --lora v1.pt --out query_bench.json --ns 1000,63538,1000000 --search-devices mps,cpu
"""
import argparse
import json
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from screenret.cli import add_model_args, load_from_args, model_meta
from screenret.config import QUERY_INSTRUCTION

QUERIES = ["Which album's release page did I pull up on Discogs to get the tracks?",
           "What price limit did I set when shopping for a yoga mat?",
           "When did I write and save my to-do list in Notepad?",
           "Which flight search site did I use to compare fares to Denver?",
           "What was the case number on the visa application confirmation?"]
DIM = 2048


def sync(dev: str):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


def bench_encode(m, reps: int) -> tuple[dict, torch.Tensor]:
    dev = m.device.type
    enc = lambda q: m.embedder.embed_texts([q], instruction=QUERY_INSTRUCTION)   # noqa: E731
    for q in QUERIES[:3]:
        enc(q)                                                     # warm-up (kernel / shader compile)
    ts = []
    for r in range(reps):
        sync(dev)
        t = time.perf_counter()
        e = enc(QUERIES[r % len(QUERIES)])
        sync(dev)
        ts.append(time.perf_counter() - t)
    return dict(median_ms=float(np.median(ts) * 1000), p90_ms=float(np.percentile(ts, 90) * 1000)), F.normalize(e.float(), dim=-1)[0].cpu()


def bench_search(qv: torch.Tensor, dev: str, n: int, reps: int, topk: int) -> dict:
    dt = torch.float32 if dev == "cpu" else torch.float16
    g = torch.Generator(device="cpu").manual_seed(n)
    X = torch.empty(n, DIM, dtype=dt, device=dev)
    for s in range(0, n, 200000):                                  # fill in slices (bounded host memory)
        blk = torch.randn(min(200000, n - s), DIM, generator=g)
        X[s:s + len(blk)] = F.normalize(blk, dim=-1).to(dt).to(dev)
    q = qv.to(dt).to(dev)
    for _ in range(3):
        torch.topk(X @ q, topk)
        sync(dev)
    ts = []
    for _ in range(reps):
        sync(dev)
        t = time.perf_counter()
        _, i = torch.topk(X @ q, topk)
        i.cpu()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return dict(device=dev, dtype=str(dt).split(".")[1], N=n, index_gb=n * DIM * X.element_size() / 1e9,
                median_ms=float(np.median(ts) * 1000), p90_ms=float(np.percentile(ts, 90) * 1000))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--ns", default="1000,10000,63538,250000,1000000,2100000", help="index sizes")
    p.add_argument("--search-devices", default="mps,cpu")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = load_from_args(a)
    R = dict(machine=platform.machine(), model=model_meta(a, m), dim=DIM, topk=a.topk, reps=a.reps, search=[])
    R["encode"], qv = bench_encode(m, a.reps)
    print("query encode:", R["encode"], flush=True)
    del m
    for dev in a.search_devices.split(","):
        for n in map(int, a.ns.split(",")):
            try:
                row = bench_search(qv, dev, n, a.reps, a.topk)
            except RuntimeError as ex:
                row = dict(device=dev, N=n, error=str(ex)[:200])
            R["search"].append(row)
            print(row, flush=True)
            if dev == "mps":
                torch.mps.empty_cache()
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
