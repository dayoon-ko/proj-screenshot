"""Consistency adapter training: a second LoRA that encodes PRUNED frames so that they embed like the pixel LoRA's full frames.

Teacher = the pixel retrieval LoRA, frozen: its full-frame vectors (--teacher-frames) and its query vectors stay the index.
Student = a LoRA with the same targets and size (r 32 / alpha 32, ViT attn qkv / proj + LLM q / k / v / o_proj, dropout 0.05),
initialised from the pixel LoRA and used only to encode frames under S (lt, tau 8). Differentiable pruned encoding: only the kept
blocks go through the ViT (attention among kept patches, sliced rotary / position embeddings), the LLM sees only the kept visual
tokens at their full-grid mRoPE positions (= the stateless S path of the engine); ViT blocks checkpointed, LLM layers via HF gradient
checkpointing.

Loss per step (frames of --nq training questions: positive + 3 same-site + 1 other hard negatives, plus --nrand random T1-train frames
for coverage; they also join the KL pool):
  L = lam0 * mean(1 - cos(e_P, e_F)) + lam1 * KL(softmax(s_T / tau) || softmax(s_S / tau)) + lam2 * MSE(margin_S, margin_T)
      + lam3 * InfoNCE(teacher query -> pruned positive frame, tau_nce, other frames of the gold recordings masked)
  e_P = student pruned frame, e_F = teacher full frame, s_T = q . e_F, s_S = q . e_P (teacher queries), tau 0.02; margins (positive -
  same-site negative) in logit units (s / tau). --p-lt8 of the steps use tau 8 grey levels, the others one of {4, 12, 16}.
Paper setting: lam0 = lam1 = lam2 = 1, lam3 = 0, lr 1e-5, 300 steps, nq = nrand = 12; the released adapter is step 200
(best T1-val R@10, then cos).

Training data (person A): positive = the question's best-scoring gold frame among the T1-train frames (minus bad frames) under the
zero-shot model; hard negatives = the next best frames of other recordings whose task name is not a near-duplicate (word Jaccard <
0.6 with every gold recording), 3 sharing a site with the gold recordings and 1 other (topped up with other-site frames).
Validation every --eval-every steps on T1-val (315 questions over its frames): mean cos(e_P, e_F), R@1 / R@10 with the teacher queries.

Inputs
  --lora            pixel retrieval LoRA (.pt): teacher and student initialisation
  --train / --val   data/questions_train.jsonl, data/questions_val.jsonl
  --splits          data/splits.json (T1-train / T1-val frames, bad_fids)
  --frames-meta     {"fid", "uid", "rel_path"} per frame; --frames-root its image root
  --diary           data_pipeline s2 diary.jsonl (recording category and site)
  --tasks-meta      PSAI psai_tasks_meta.jsonl (task names, for near-duplicate negatives)
  --zs-frames       id-indexed zero-shot frame vectors:  python -m screenret.embed_stream --preset full --lora none --dtype float16 ...
  --zs-queries      zero-shot vectors of --train in file order:  python -m screenret.embed_text queries --lora none --texts <train> ...
  --teacher-frames  id-indexed full-frame vectors of the pixel LoRA:  embed_stream --preset full --lora <LoRA> --dtype bfloat16 --no-lora-merge
Output: <out-dir>/<tag>.step<k>_lora.pt (every --eval-every steps), <tag>.best_lora.pt, <tag>.json (history); load any of them
with --lora (frames of S / S+G presets). Needs a CUDA GPU with bf16 (~40 GB at the paper setting).

usage: python scripts/train_consistency_adapter.py {check,smoke,train} --lora pixel_lora.pt ... --out-dir runs/adapter
  check  stateless S encoding of this script vs the engine (preset S, no reuse), and grad path vs no-grad path
  smoke  two steps on the same batch (loss must move), peak memory
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup
from transformers.models.qwen3_vl import modeling_qwen3_vl as MQ

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screenret.config import LEVEL, MODEL_ID, PRESETS, QUERY_INSTRUCTION   # noqa: E402
from screenret.engine import Engine, preprocess                             # noqa: E402
from screenret.model import attach_lora, load_model, lora_state             # noqa: E402
from screenret.ocr import embed_texts                                       # noqa: E402
from screenret.spatial import lt_keep_mask                                  # noqa: E402

D8 = 8 * LEVEL
ALT = [4 * LEVEL, 12 * LEVEL, 16 * LEVEL]
N_SAME, N_ANY, SEED = 3, 1, 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "smoke", "train"])
    for k in ("lora", "train", "val", "splits", "frames-meta", "frames-root", "diary", "tasks-meta", "zs-frames", "zs-queries",
              "teacher-frames", "out-dir"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--tag", default="consistency_adapter")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--nq", type=int, default=12)
    p.add_argument("--nrand", type=int, default=12)
    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--lam1", type=float, default=1.0)
    p.add_argument("--lam2", type=float, default=1.0)
    p.add_argument("--lam3", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--tau-nce", type=float, default=0.07)
    p.add_argument("--p-lt8", type=float, default=0.8)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


a = parse_args()
torch.multiprocessing.set_sharing_strategy("file_system")   # many small tensors from DataLoader workers
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
os.makedirs(a.out_dir, exist_ok=True)

# ---------------------------------------------------------------- model (student; starts as the pixel LoRA)
m = load_model("cuda", torch.bfloat16, lora=None, model_id=a.model_id)
model = m.embedder.model
attach_lora(model, a.lora, dropout=a.lora_dropout)
for n, p_ in model.named_parameters():
    p_.requires_grad = n.endswith(("lora_a", "lora_b"))
model.to(m.device)
if a.mode != "check":
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.eval()
M, V, LM, IMG, dev = m.core, m.vit, m.llm, m.image_token_id, m.device
DT = V.dtype
NH, HD, DSI = m.vit_heads, m.vit_head_dim, m.deepstack_layers
light = m.embedder.without_model()
print(f"student LoRA: {sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) / 1e6:.1f}M trainable parameters | "
      f"attn {model.config._attn_implementation} | {torch.cuda.get_device_name(0)}", flush=True)


def prep(path, tau):
    """Processor tensors (CPU) + S keep mask (lt at tau; None = keep every block)."""
    it = light.preprocess([light.format_model_input(image=path)])
    thw = it["image_grid_thw"]
    n = int(thw[0, 1]) // 2 * (int(thw[0, 2]) // 2)
    it["keep"] = torch.from_numpy(lt_keep_mask(it["pixel_values"], thw, tau)) if tau is not None else torch.ones(n, dtype=torch.bool)
    return it


GC = {}


@torch.no_grad()
def grid(it):
    """Per-(resolution, prompt) constants: ViT position embedding / RoPE, prompt embeddings, mRoPE positions (no grad needed)."""
    thw, ids = it["image_grid_thw"], it["input_ids"]
    key = (tuple(thw[0].tolist()), ids.shape[1])
    g = GC.get(key)
    if g is not None:
        assert torch.equal(g["ids"], ids)
        return g
    cv = V.config
    ii, iw = MQ.get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=V.num_grid_per_side, mode=V.interpolation_mode,
                                                             align_corners=V.interpolation_align_corners,
                                                             spatial_merge_size=cv.spatial_merge_size, kwargs={})
    pid = MQ.get_vision_position_ids(thw, V.spatial_merge_size, kwargs={})
    pos = (V.pos_embed(ii) * iw[:, :, None]).sum(1)
    cos, sin = V.rotary_pos_emb(pos, pid)
    mm = it.get("mm_token_type_ids", (ids == IMG).int())
    x0 = M.get_input_embeddings()(ids)
    M.rope_deltas = None
    p3 = M.compute_3d_position_ids(input_ids=ids, inputs_embeds=x0, image_grid_thw=thw, attention_mask=it["attention_mask"],
                                   mm_token_type_ids=mm)
    ipos = (ids[0] == IMG).nonzero()[:, 0]
    first, last, n = int(ipos[0]), int(ipos[-1]), len(ipos)
    assert last - first + 1 == n and int(thw.prod()) == 4 * n
    g = dict(ids=ids, pos=pos, cos=cos, sin=sin, x0=x0[0], p3=p3[:, 0], first=first, last=last, L=ids.shape[1])
    GC[key] = g
    return g


def rows4(b):
    return (b[:, None] * 4 + torch.arange(4, device=b.device)).reshape(-1)


def vblock(blk, X, cos, sin):
    """One ViT block over the kept patches of one image (single-sequence attention)."""
    h = blk.norm1(X)
    q, k, v = blk.attn.qkv(h).reshape(-1, 3, NH, HD).permute(1, 0, 2, 3).unbind(0)
    q, k = MQ.apply_rotary_pos_emb_vision(q, k, cos, sin)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], dropout_p=0.0,
                                       scale=blk.attn.scaling, is_causal=False)
    X = X + blk.attn.proj(o[0].transpose(0, 1).reshape(X.shape[0], -1))
    return X + blk.mlp(blk.norm2(X))


def encode(it, grad=False):
    """Pruned single-image embedding (kept blocks only); differentiable when grad=True (ViT blocks checkpointed)."""
    it = {k: v.to(dev, non_blocking=True) for k, v in it.items()}
    g = grid(it)
    with torch.set_grad_enabled(grad):
        kept = it["keep"].nonzero()[:, 0]
        rows, nK, f = rows4(kept), len(kept), g["first"]
        X = V.patch_embed(it["pixel_values"].float()[rows].to(DT))
        X = X + g["pos"][rows].to(X.dtype)
        cos, sin = g["cos"][rows], g["sin"][rows]
        ds = []
        for layer, blk in enumerate(V.blocks):
            X = checkpoint(vblock, blk, X, cos, sin, use_reentrant=False) if grad else vblock(blk, X, cos, sin)
            if layer in DSI:
                ds.append(V.deepstack_merger_list[DSI.index(layer)](X))
        feat = V.merger(X)
        seq = torch.cat([torch.arange(f, device=dev), f + kept, torch.arange(g["last"] + 1, g["L"], device=dev)])
        Lq = len(seq)
        x = g["x0"][seq].clone()
        x[f:f + nK] = feat.to(x.dtype)
        vm = torch.zeros(1, Lq, dtype=torch.bool, device=dev)
        vm[0, f:f + nK] = True
        hs = LM(input_ids=None, inputs_embeds=x[None], position_ids=g["p3"][:, seq][:, None],
                attention_mask=torch.ones(1, Lq, dtype=torch.long, device=dev), visual_pos_masks=vm,
                deepstack_visual_embeds=[d.to(x.dtype) for d in ds], use_cache=False).last_hidden_state
        return F.normalize(hs[0, -1].float(), p=2, dim=-1)


class DS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items              # (path, tau)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        return prep(*self.items[j])


def loader(items):
    return torch.utils.data.DataLoader(DS(items), batch_size=1, num_workers=a.workers, collate_fn=lambda x: x[0],
                                       prefetch_factor=8 if a.workers else None)


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


S = json.load(open(a.splits))
FR = {r["fid"]: r for r in read_jsonl(a.frames_meta)}
n_rows = max(FR) + 1
uid = np.array([FR[f]["uid"] if f in FR else "" for f in range(n_rows)])
bad = np.zeros(n_rows, bool)
bad[S["bad_fids"]] = True


def path(f):
    return os.path.join(a.frames_root, FR[int(f)]["rel_path"])


TV = np.load(a.teacher_frames, mmap_mode="r")


def fullv(fs):
    return F.normalize(torch.tensor(np.asarray(TV[np.asarray(fs)], np.float32)), dim=1)


# ---------------------------------------------------------------- check: this encoder vs the engine, grad vs no-grad
if a.mode == "check":
    eng, cfg = Engine(m, PRESETS["S"].with_(linear_patch_embed=False)), PRESETS["S"].with_(linear_patch_embed=False)
    vf = [f for f in S["fids"]["T1-val"] if not bad[f]][::300][:6]
    for f in vf:
        e_eng, _ = eng.step(preprocess(m, path(f), cfg), first=True)
        e_me = encode(prep(path(f), D8)).cpu()
        e_full = encode(prep(path(f), None)).cpu()
        print(f"fid {f}: cos(this S encoding, engine preset S) {float(e_me @ F.normalize(e_eng.float().cpu(), dim=0)):.6f} | "
              f"cos(this keep-all encoding, teacher frame) {float(e_full @ fullv([f])[0]):.5f}", flush=True)
    it = prep(path(vf[0]), D8)
    e0, e1 = encode(it), encode(it, grad=True)
    print(f"grad / checkpoint path vs no-grad path: cos {float(e0 @ e1.detach()):.6f}", flush=True)
    print("CHECK DONE")
    sys.exit(0)

# ---------------------------------------------------------------- data: positives and hard negatives from the zero-shot model
words_of = {t["unique_data_id"]: set(re.findall(r"[a-z0-9]+", (t["task_name"] or "").lower())) for t in read_jsonl(a.tasks_meta)}
diary = read_jsonl(a.diary)
CAT = {u: d["category"] for d in diary for u in d["uids"]}
SITE = {u: {x.strip().lower() for x in (d.get("site") or "").split(",") if x.strip()} for d in diary for u in d["uids"]}


def jac(x, y):
    return len(x & y) / max(1, len(x | y))


rows = read_jsonl(a.train)
Q0 = np.load(a.zs_queries, mmap_mode="r")
assert len(Q0) == len(rows), "--zs-queries must hold one row per training question, in file order"
order_rows = sorted(range(len(rows)), key=lambda i: CAT.get(rows[i]["gold"][0]) != "web")   # web questions first, as in training
V0 = np.load(a.zs_frames, mmap_mode="r")
bank = np.array(S["fids"]["T1-train"])
bank = bank[~bad[bank]]
Xb, ub = np.asarray(V0[bank], dtype=np.float32), uid[bank]
train = []
for i in order_rows:
    r = rows[i]
    s, g = Xb @ np.asarray(Q0[i], np.float32), np.isin(ub, r["gold"])
    if not g.any():
        continue
    pos = int(bank[np.where(g)[0][np.argmax(s[g])]])
    gw = [words_of.get(u, set()) for u in r["gold"]]
    gs = set().union(*[SITE.get(u, set()) for u in r["gold"]])
    same, other, used = [], [], set()
    for j in np.argsort(-s):
        u = ub[j]
        if g[j] or u in used or any(jac(words_of.get(u, set()), w) >= 0.6 for w in gw):
            continue
        used.add(u)
        if gs & SITE.get(u, set()):
            if len(same) < N_SAME:
                same.append(int(bank[j]))
        elif len(other) < N_SAME + N_ANY:
            other.append(int(bank[j]))
        if len(same) == N_SAME and len(other) >= N_SAME + N_ANY:
            break
    train.append({"text": r["question"], "pos": pos, "same": same, "other": other[:N_SAME + N_ANY - len(same)], "gold": set(r["gold"])})
val_rows = read_jsonl(a.val)
vpool = np.array(S["fids"]["T1-val"])
vpool = vpool[~bad[vpool]]
vgold = torch.tensor(np.stack([np.isin(uid[vpool], r["gold"]) for r in val_rows]))
print(f"train {len(train)} questions (same-site negatives {np.mean([len(t['same']) for t in train]):.2f}/q) | val {len(val_rows)} "
      f"questions over {len(vpool)} frames", flush=True)
QT = embed_texts(m, [t["text"] for t in train], instruction=QUERY_INSTRUCTION, batch_size=64)   # teacher queries (student == teacher now)
QVv = embed_texts(m, [r["question"] for r in val_rows], instruction=QUERY_INSTRUCTION, batch_size=64)
EFv = fullv(vpool)


def ranks(scores, gold):
    best = scores.masked_fill(~gold, float("-inf")).max(1).values
    return (scores > best[:, None]).sum(1) + 1


def val_score():
    model.eval()
    E = torch.stack([encode(it).cpu() for it in loader([(path(f), D8) for f in vpool])])
    cs, r = (E * EFv).sum(1), ranks(QVv @ E.T, vgold)
    return dict(cos=float(cs.mean()), cos_p5=float(np.percentile(cs.numpy(), 5)), r1=float((r <= 1).float().mean() * 100),
                r10=float((r <= 10).float().mean() * 100))


# ---------------------------------------------------------------- schedule (fixed in advance -> resumable)
NSTEPS = 2 if a.mode == "smoke" else a.steps
order = rng.permutation(len(train)).tolist()
while len(order) < NSTEPS * a.nq:
    order += rng.permutation(len(train)).tolist()
SCHED = []
for s in range(NSTEPS):
    qs = order[s * a.nq:(s + 1) * a.nq] if a.mode != "smoke" else order[:a.nq]
    tau = D8 if rng.random() < a.p_lt8 else float(rng.choice(ALT))
    rnd = [int(x) for x in rng.choice(bank, a.nrand, replace=False)]
    if a.mode == "smoke":
        tau = D8
    fl = list(dict.fromkeys([f for q in qs for f in [train[q]["pos"]] + train[q]["same"] + train[q]["other"]] + rnd))
    SCHED.append(dict(qs=qs, tau=tau, fids=fl))
if a.mode == "smoke":
    SCHED[1] = SCHED[0]

params = [p_ for p_ in model.parameters() if p_.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * NSTEPS), NSTEPS)


def train_step(st, its):
    model.train()
    fi = {f: j for j, f in enumerate(st["fids"])}
    E = torch.stack([encode(it, grad=True) for it in its])
    EF = fullv(st["fids"]).to(dev)
    cons = (1 - (E * EF).sum(1)).mean()
    Qb = QT[st["qs"]].to(dev)
    sT, sS = Qb @ EF.T / a.tau, Qb @ E.T / a.tau
    kl = F.kl_div(F.log_softmax(sS, 1), F.log_softmax(sT, 1), log_target=True, reduction="batchmean")
    pi, ni = [], []
    for i, q in enumerate(st["qs"]):
        for n in train[q]["same"]:
            pi.append((i, fi[train[q]["pos"]]))
            ni.append((i, fi[n]))
    mse = torch.zeros((), device=dev)
    if pi:
        pi, ni = torch.tensor(pi, device=dev), torch.tensor(ni, device=dev)
        mS = sS[pi[:, 0], pi[:, 1]] - sS[ni[:, 0], ni[:, 1]]
        mT = sT[pi[:, 0], pi[:, 1]] - sT[ni[:, 0], ni[:, 1]]
        mse = F.mse_loss(mS, mT)
    nce = torch.zeros((), device=dev)
    if a.lam3 > 0:
        lg = Qb @ E.T / a.tau_nce
        tgt = torch.tensor([fi[train[q]["pos"]] for q in st["qs"]], device=dev)
        msk = torch.zeros_like(lg, dtype=torch.bool)
        for i, q in enumerate(st["qs"]):
            for f, j in fi.items():
                if j != tgt[i] and uid[f] in train[q]["gold"]:
                    msk[i, j] = True
        nce = F.cross_entropy(lg.masked_fill(msk, float("-inf")), tgt)
    loss = a.lam0 * cons + a.lam1 * kl + a.lam2 * mse + a.lam3 * nce
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    opt.step()
    sched.step()
    return dict(loss=loss.item(), cons=cons.item(), kl=kl.item(), mse=mse.item(), nce=nce.item(), gnorm=gn, n=len(its), tau=st["tau"])


def batches(s0):
    it = iter(loader([(path(f), st["tau"]) for st in SCHED[s0:] for f in st["fids"]]))
    for st in SCHED[s0:]:
        yield st, [next(it) for _ in st["fids"]]


def save(obj, path_):
    tmp = path_ + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path_)


if a.mode == "smoke":
    torch.cuda.reset_peak_memory_stats()
    out = []
    for st, its in batches(0):
        t = time.time()
        out.append(train_step(st, its))
        out[-1]["sec"] = round(time.time() - t, 1)
        print("smoke step", out[-1], flush=True)
    print(f"loss moved {out[0]['loss']:.5f} -> {out[1]['loss']:.5f} | peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    print("SMOKE DONE")
    sys.exit(0)

tag, ck = a.tag, os.path.join(a.out_dir, f"{a.tag}.ckpt.pt")
best_file = os.path.join(a.out_dir, f"{tag}.best_lora.pt")


def set_lora(state):
    for n, mod in model.named_modules():
        if hasattr(mod, "lora_a"):
            mod.lora_a.data.copy_(state[f"base_model.model.{n}.lora_A.weight"])
            mod.lora_b.data.copy_(state[f"base_model.model.{n}.lora_B.weight"])


if os.path.exists(ck):
    c = torch.load(ck, weights_only=False)
    set_lora(c["lora"])
    opt.load_state_dict(c["opt"])
    sched.load_state_dict(c["sched"])
    step0, hist, best = c["step"], c["hist"], c["best"]
    print(f"resumed at step {step0}", flush=True)
else:
    v = val_score()
    r = ranks(QVv @ EFv.T, vgold)
    full = dict(r1=float((r <= 1).float().mean() * 100), r10=float((r <= 10).float().mean() * 100))
    hist, step0, best = [dict(step=0, **v, full=full)], 0, None
    print(f"[{tag}] step 0 = teacher: S val cos {v['cos']:.4f} (p5 {v['cos_p5']:.4f}) R@1 {v['r1']:.1f} R@10 {v['r10']:.1f} | "
          f"teacher full frames R@1 {full['r1']:.1f} R@10 {full['r10']:.1f}", flush=True)
logs, t = [], time.time()
for st, its in batches(step0):
    step0 += 1
    step = step0
    logs.append(train_step(st, its))
    if step % 10 == 0:
        print(f"  step {step}: " + " ".join(f"{k} {np.mean([x[k] for x in logs[-10:]]):.4f}" for k in ("loss", "cons", "kl", "mse", "nce", "gnorm"))
              + f" ({time.time() - t:.0f} s)", flush=True)
    if step % a.eval_every == 0 or step == NSTEPS:
        v = val_score()
        h = dict(step=step, **v, **{k: float(np.mean([x[k] for x in logs])) for k in ("loss", "cons", "kl", "mse")}, sec=time.time() - t)
        hist.append(h)
        save(lora_state(model), os.path.join(a.out_dir, f"{tag}.step{step}_lora.pt"))
        if best is None or (v["r10"], v["cos"]) > (best["r10"], best["cos"]):
            best = dict(step=step, r1=v["r1"], r10=v["r10"], cos=v["cos"])
            save(lora_state(model), best_file)
        print(f"[{tag}] step {step} loss {h['loss']:.4f} (cons {h['cons']:.4f} kl {h['kl']:.4f} mse {h['mse']:.4f}) | val S cos {v['cos']:.4f} "
              f"(p5 {v['cos_p5']:.4f}) R@1 {v['r1']:.1f} R@10 {v['r10']:.1f} | best step {best['step']} ({time.time() - t:.0f} s)", flush=True)
        save({"lora": lora_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "hist": hist, "best": best}, ck)
        logs, t = [], time.time()
json.dump(dict(teacher=os.path.basename(a.lora), args=vars(a), n_train=len(train), best=best, history=hist, margin_units="logits (s/tau)"),
          open(os.path.join(a.out_dir, f"{tag}.json"), "w"))
if os.path.exists(ck):
    os.remove(ck)
print(f"[{tag}] DONE best step {best['step']}: R@10 {best['r10']:.1f} cos {best['cos']:.4f}", flush=True)
