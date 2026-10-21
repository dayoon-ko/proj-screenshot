"""On-device latency of ONE configuration over a stream timing sample (one config per process).

Sample JSON (scripts/make_timing_sample.py): {"chunks": [{"ctx": [{"fid", "rel_path"}...], "frames": [...]}, ...]}. Each chunk is a
window of consecutive frames of the screen stream; `ctx` are the frames right before it.

Timed per frame: image read + processor (+ Token Pruning mask) + ViT + LLM + device sync -> embedding on the CPU.
OCR routes: OCR + text embedding.
Untimed: model load, warm-up (first frames of the first window), the ctx frames before each window (stateful configs only: the caches
reach the state the continuous stream has at the window start), the --gap sleep between frames (5-s capture period is mostly idle).

Output: <out>/<tag>_<name>.json (per-frame rows + summary), <tag>_<name>_emb.npy (fp16), <tag>_<name>_fids.npy.

example: python -m screenret.timing --device mps --preset S+G+T+L --lora adapter.pt --sample sample.json --frames-root FRAMES --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import sysmon
from .cli import add_cpu_speed_args, add_method_args, add_model_args, load_from_args, method_from_args, model_meta
from .engine import Engine, preprocess
from .macs import full_frame_macs
from .ocr import ocr_embed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m screenret.timing", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_method_args(p)
    add_model_args(p)
    add_cpu_speed_args(p)
    g = p.add_argument_group("run")
    g.add_argument("--route", choices=["screenshot", "ocr-rapid", "ocr-vision"], default="screenshot",
                   help="screenshot = the method (component flags apply); ocr-* = OCR baseline + text embedding")
    g.add_argument("--sample", required=True, help="timing sample JSON")
    g.add_argument("--frames-root", default=os.environ.get("SCREENRET_FRAMES"), help="root of rel_path (default $SCREENRET_FRAMES)")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--tag", default="run", help="label of the condition, e.g. busy / idle (prefix of the output files)")
    g.add_argument("--gap", type=float, default=0.3, help="untimed sleep between frames, s (default 0.3)")
    g.add_argument("--n-chunks", type=int, default=0, help="use only the first n windows (0 = all)")
    g.add_argument("--chunk-step", type=int, default=1, help="use every k-th window")
    g.add_argument("--warmup", type=int, default=5, help="untimed warm-up frames (ctx frames of the first window)")
    g.add_argument("--min-avail-gb", type=float, default=0.3, help="stop after two windows with less free memory than this (swapping)")
    return p.parse_args(argv)


class Runner:
    """Embeds one frame with the configured route and returns (embedding, timing/stats row)."""

    def __init__(self, args, m, cfg):
        self.args, self.m, self.cfg = args, m, cfg
        self.ocr = args.route != "screenshot"
        self.stateful = not self.ocr and cfg.temporal_vit

    def engine(self):
        return None if self.ocr else Engine(self.m, self.cfg)

    def one(self, eng, path: str, first: bool):
        tim: dict = {}
        if self.ocr:
            t0 = time.perf_counter()
            e, nch = ocr_embed(self.m, path, self.args.route, tim)
            t1 = time.perf_counter()
            return e.float().cpu(), dict(s=t1 - t0, s_ocr=tim["t_ocr_done"] - t0, s_text_embed=t1 - tim["t_ocr_done"], ocr_chars=nch)
        self.m.sync()
        t0 = time.perf_counter()
        frame = preprocess(self.m, path, self.cfg)
        t1 = time.perf_counter()
        e, row = eng.step(frame, first, tim)
        e = e.float().cpu()
        t2 = time.perf_counter()
        row.update(s=t2 - t0, s_prep=t1 - t0, s_vit=tim["t_vit_done"] - t1, s_llm=t2 - tim["t_vit_done"])
        return e, row


def summarize(rows: list[dict], full_macs: int | None, capture_period: float = 5.0) -> dict:
    s = np.array([r["s"] for r in rows])
    full = np.array([bool(r.get("full", True)) for r in rows])
    mean_of = lambda k: float(np.mean([r[k] for r in rows])) if k in rows[0] else None   # noqa: E731
    has_macs = "vit_macs" in rows[0] and full_macs
    return dict(
        n=len(rows), mean=float(s.mean()), std=float(s.std()), median=float(np.median(s)), p90=float(np.percentile(s, 90)),
        p99=float(np.percentile(s, 99)), duty_cycle_median=float(np.median(s) / capture_period),
        refresh_rate=float(full.mean()) if "full" in rows[0] else None,
        median_refresh=float(np.median(s[full])) if "full" in rows[0] and full.any() else None,
        median_nonrefresh=float(np.median(s[~full])) if "full" in rows[0] and not full.all() else None,
        mean_prep=mean_of("s_prep"), mean_vit=mean_of("s_vit"), mean_llm=mean_of("s_llm"),
        mean_ocr=mean_of("s_ocr"), mean_text_embed=mean_of("s_text_embed"),
        mac_frac_mean=float(np.mean([r["vit_macs"] + r["llm_macs"] for r in rows]) / full_macs) if has_macs else None,
        rss_peak_gb=max(r["rss_peak_gb"] for r in rows),
        device_mem_peak_gb=max(r.get("mps_driver_gb", r.get("cuda_peak_gb", 0)) for r in rows),
        swap_frames_frac=float(np.mean([r["swap_delta"] > 0 for r in rows])), load1_mean=float(np.mean([r["load1"] for r in rows])))


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = method_from_args(args)
    name = cfg.name if args.route == "screenshot" else args.route.upper()
    sample = json.load(open(args.sample))
    chunks = sample["chunks"][::args.chunk_step][: args.n_chunks or None]
    path_of = lambda rel: os.path.join(args.frames_root or "", rel)   # noqa: E731
    os.makedirs(args.out, exist_ok=True)

    t = time.time()
    m = load_from_args(args)
    load_s = time.time() - t
    run = Runner(args, m, cfg)
    need = [f["rel_path"] for ch in chunks for f in (ch["ctx"] if run.stateful else []) + ch["frames"]]
    miss = [p for p in need if not os.path.exists(path_of(p))]
    if miss:
        sys.exit(f"{len(miss)} frames missing under {args.frames_root}, e.g. {miss[:2]}")
    print(f"[{name}] loaded in {load_s:.1f}s | {model_meta(args, m)} | {len(chunks)} windows", flush=True)

    res_path = f"{args.out}/{args.tag}_{name}.json"
    R = dict(name=name, route=args.route, method=vars(cfg) if not run.ocr else None, tag=args.tag, gap_s=args.gap,
             chunk_step=args.chunk_step, n_chunks=len(chunks), sample=os.path.basename(args.sample), load_s=load_s,
             model=model_meta(args, m), env_start=sysmon.environment())
    save = lambda: json.dump(R, open(res_path, "w"), indent=1)   # noqa: E731

    eng = run.engine()                                           # warm-up (untimed): kernels, shaders, OCR models
    for k, f in enumerate(chunks[0]["ctx"][:args.warmup]):
        run.one(eng, path_of(f["rel_path"]), k == 0)

    rows, embs, fids, low = [], [], [], 0
    t_all = time.time()
    for ci, ch in enumerate(chunks):
        eng = run.engine()
        if run.stateful:                                         # untimed history: caches match the continuous stream
            for k, f in enumerate(ch["ctx"]):
                run.one(eng, path_of(f["rel_path"]), k == 0)
                time.sleep(args.gap)
        started = run.stateful and len(ch["ctx"]) > 0
        for k, f in enumerate(ch["frames"]):
            sw0 = sysmon.swaps()
            e, row = run.one(eng, path_of(f["rel_path"]), not started and k == 0)
            row.update(chunk=ci, k=k, fid=f.get("fid", -1), load1=os.getloadavg()[0], swap_delta=sysmon.swaps() - sw0,
                       **sysmon.memory(m.device.type), **sysmon.battery())
            rows.append(row)
            embs.append(e.numpy().astype(np.float16))
            fids.append(f.get("fid", -1))
            time.sleep(args.gap)
        low = low + 1 if sysmon.avail_gb() < args.min_avail_gb else 0
        if low >= 2:
            R["rows"], R["stopped"] = rows, "memory guard"
            save()
            sys.exit(f"MEMORY GUARD: free memory below {args.min_avail_gb} GB twice; stopping")
        if (ci + 1) % 10 == 0 or ci + 1 == len(chunks):
            s = np.array([r["s"] for r in rows])
            print(f"[{args.tag} {name}] window {ci + 1}/{len(chunks)} | {len(rows)} frames | median {np.median(s):.3f} s "
                  f"mean {s.mean():.3f} s"
                  f" | wall {(time.time() - t_all) / 60:.1f} min", flush=True)
            R["rows"] = rows
            save()

    g = None if run.ocr else next(iter(m.grids.values()))
    fm = None if run.ocr else full_frame_macs(m, g)
    R["summary"] = summarize(rows, fm)
    if g is not None:
        R["summary"]["kept_frac_mean"] = float(np.mean([r["n_kept"] for r in rows]) / g.n_blocks)
    R["full_frame_macs"] = fm
    R["wall_s"], R["rows"], R["env_end"] = time.time() - t_all, rows, sysmon.environment()
    save()
    np.save(f"{args.out}/{args.tag}_{name}_emb.npy", np.stack(embs))
    np.save(f"{args.out}/{args.tag}_{name}_fids.npy", np.array(fids))
    print(f"DONE {args.tag} {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["summary"].items()}),
          flush=True)


if __name__ == "__main__":
    main()
