# screen-retrieval: screens beat text for private on-device retrieval

Code for the MLSys 2027 submission *"Screens beat text"*: private, on-device retrieval over a personal screenshot stream
(one screenshot every 5 s). We index each screenshot directly with **Qwen3-VL-Embedding-2B** and compare against the usual
**OCR-text index** (OCR, then the same model as a text embedder). To make the screenshot index cheap enough to run continuously on a
laptop, the stream embedder reuses work across space and time:

| component | what it does | exact? |
|---|---|---|
| **S** (spatial, "LT8") | drop a 32x32-px block if every pixel is within 8 grey levels of its LEFT or UPPER neighbour block (order-free, vectorised); dropped blocks enter neither ViT nor LLM | approx. |
| **G** (Grid Keep, "t2") | on top of S: within each flat region (blocks linked by S's left / upper equalities), keep the top-left block of every 2x2-block tile that S left without a kept block, so large flat areas keep a sparse grid of blocks at their own positions | approx. |
| **T_vit** (temporal ViT) | only blocks whose pixels changed by > 8 grey levels since their last recompute run through the ViT; they attend over cached post-RoPE K/V of the unchanged blocks. Full refresh when > 90 % of kept blocks changed (rho 0.9) or every K = 10 frames | approx. |
| exact LLM prefix reuse | (on with T_vit) the LLM is causal: reuse the previous frame's KV cache up to the first changed position | exact |
| **T_llm** (temporal LLM) | after the first change, recompute only changed visual tokens + all following text tokens; unchanged visual tokens reuse cached K/V | approx. |
| **+pe** | ViT Conv3d patch embedding computed as the identical `F.linear` (MPS runs Conv3d at ~0.02 TFLOP/s) | exact (same math) |

The paper measures (1) **retrieval accuracy** of screenshot vs OCR indexes over a 63,538-frame pool: single-hop R@k on 4,265
questions and recall@10 on 846 parallel multi-hop questions (several evidence recordings each), zero-shot and with a retrieval LoRA,
and the accuracy cost of each efficiency component; (2) **on-device cost**: per-frame latency on a 16 GB MacBook (MPS fp16 and CPU,
idle and in the wild), MACs per frame, memory, and query-time cost vs corpus size.

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # on Apple silicon use an arm64 Python (not x86 under Rosetta: no MPS)
pip install -r requirements.txt && pip install -e .
pip install pyobjc-framework-Vision pyobjc-framework-Cocoa   # macOS only, for the Apple Vision OCR baseline
```

Tested with Python 3.12, torch 2.8.0, transformers 5.17.0 (Mac: MPS / CPU; server: CUDA 12.6).

## Model and LoRA

- Base model: `Qwen/Qwen3-VL-Embedding-2B` (downloaded from the Hugging Face Hub on first use; `--model-id` takes a local path).
- Screenshot rows = base model + the **pixel retrieval LoRA** (`mp8_pixel_bal.step50_lora.pt`: PEFT LoRA r 32 / alpha 32 on the
  ViT attention `qkv`/`proj` and the LLM `q/k/v/o_proj`, "v1" recipe on the 1,000 questions of `data/questions_train.jsonl`). It
  encodes the full-screenshot frames and every screenshot row's queries.
- Paper's method (`S+G+T+L`): frames are encoded by the **Consistency adapter** (`mp12_pixel_bal_padapt.step200_lora.pt`), a second
  LoRA with the same targets and size, initialised from the pixel LoRA and trained so that pruned frames embed like the pixel LoRA's
  full frames (loss = (1 - cos) consistency + ranking KL (tau 0.02) + MarginMSE against the frozen pixel LoRA). Queries stay on the
  pixel LoRA, so the method's frame vectors drop into the same index. It loads like any LoRA: `--lora <adapter.pt>` for frames.
- The earlier "v1" pixel LoRA (`stage2_pixelA.best_lora.pt`, all 2,837 T1-train questions) is the LoRA of the timing table and of
  the release check below; timing does not depend on which LoRA is merged.
- OCR rows = base model + the **OCR LoRA** (`stage2_ocrA.best_lora.pt`: same recipe on the LLM `q/k/v/o_proj` only, trained on the
  RapidOCR text of the same screenshots), used both for the OCR-text frame vectors and for that row's query vectors.
- By default a LoRA is **merged into the weights at load**, so the compute graph (and the timing) is the zero-shot model's.
  Download: **`<LORA_URL — to be added at camera-ready>`**.
- Every entry point takes `--lora PATH` (or `$SCREENRET_LORA`) or `--lora none` for the zero-shot model; there is no silent default.
- In **bf16** pass `--no-lora-merge`: rounding `W + B A` to bf16 (8-bit mantissa) drops much of the low-rank update, so the LoRA is then
  applied as a separate fp32 path (`x W^T + (x A^T) B^T`, as PEFT computes it). Merging is fine in fp16 / fp32 (the Mac default).
- LoRA and Consistency adapter training code is not part of this release.

## Quick start

```bash
export SCREENRET_FRAMES=/path/to/frames_5s        # root of the screenshot files (relative paths in the lists below)
export SCREENRET_LORA=/path/to/mp12_pixel_bal_padapt.step200_lora.pt   # Consistency adapter: frames of the paper's method
LORA=/path/to/mp8_pixel_bal.step50_lora.pt                              # pixel LoRA: full frames and queries

# on-device latency of one configuration (one config per process; warm-up and context frames untimed); see "Timing rows" below
python -m screenret.timing --device mps --preset S+G+T+L --sample sample.json --gap 0.3 --out results/ --tag idle
python -m screenret.timing --device cpu --dtype float32 --compile --weight-storage bf16-llm-fp32 --preset S+G+T+L --sample sample.json --out results_cpu/ --tag idle

# accuracy: embed the stream in time order (4 parallel segments), merge, evaluate
python scripts/make_stream_order.py --splits splits.json --frames-meta frames.jsonl --out order.tsv
for i in 0 1 2 3; do python -m screenret.embed_stream --device cuda --dtype bfloat16 --no-lora-merge --preset S+G+T+L --frames-list order.tsv --segment $i/4 --out runs/final_seg$i.npz; done
python -m screenret.merge_segments runs/final_seg*.npz --n-rows 80459 --out runs/final_frames.npy   # rows = frames in frames.jsonl
for i in 0 1 2 3; do python -m screenret.embed_stream --device cuda --dtype bfloat16 --no-lora-merge --preset full --lora $LORA --frames-list order.tsv --segment $i/4 --out runs/full_seg$i.npz; done
python -m screenret.merge_segments runs/full_seg*.npz --n-rows 80459 --out runs/full_frames.npy
python scripts/make_eval_protocol.py --eval-input data/eval_input_test.json --texts data/texts_test.jsonl --parallel data/questions_test_parallel.jsonl \
    --splits data/splits.json --frames-meta frames.jsonl --out protocol.json --queries-out protocol_queries.jsonl   # 4,265 single + 846 parallel
python -m screenret.embed_text queries --device cuda --dtype bfloat16 --no-lora-merge --lora $LORA --texts protocol_queries.jsonl --out q_lora.npy
python -m screenret.evaluate --protocol protocol.json --queries q_lora.npy --frames FULL=runs/full_frames.npy S+G+T+L=runs/final_frames.npy --reference FULL
```

Every command has `--help`.

## Component flags

A `--preset` sets defaults (default `S+G+T+L`); explicit flags override it. The same flags work for `timing` and `embed_stream`
(`--compile` exists only there).

| preset | `--spatial` | `--temporal-vit` | `--approx-llm` | `--linear-patch-embed` | tag in outputs |
|---|---|---|---|---|---|
| `full` | none | off | off | off (stock Conv3d) | `FULL` |
| `full+pe` | none | off | off | on | `FULL+pe` |
| `S` | lt | off | off | on | `LT8+pe` |
| `S+T` | lt | on | off | on | `LT8T+pe` |
| `S+T+L` | lt | on | on | on | `LT8TL+pe` |
| `S+G` | lt-grid | off | off | on | `LT8t2+pe` |
| `S+G+T+L` (paper method) | lt-grid | on | on | on | `LT8t2TL+pe` |

| flag | default | meaning |
|---|---|---|
| `--spatial {none,lt,lt-grid,pixelprune}` | lt-grid | S rule (`lt-grid` = lt + Grid Keep; `pixelprune` = PixelPrune Pred-2D, raster-order, for comparison) |
| `--tau` | 8 | S threshold, 8-bit grey levels |
| `--temporal-vit / --no-temporal-vit` | on | T_vit (+ exact LLM prefix reuse) |
| `--delta` | 8 | T_vit change threshold, grey levels |
| `--refresh-rho` | 0.9 | full refresh if more than this fraction of kept blocks changed |
| `--refresh-every` | 10 | full refresh every K frames |
| `--approx-llm / --no-approx-llm` | on | T_llm (requires T_vit) |
| `--linear-patch-embed / --no-...` | on | +pe (identical math; matters for speed on MPS only) |
| `--route {screenshot,ocr-rapid,ocr-vision}` | screenshot | timing only: OCR baselines (RapidOCR ONNX CPU / Apple Vision accurate) + text embedding |
| `--device`, `--dtype` | auto, fp16 (fp32 on CPU) | `cuda`, `mps`, `cpu` |
| `--weight-storage {native,bf16,bf16-llm-fp32}` | native | CPU + fp32 only. `bf16`: Linear weights stored bf16, upcast per call (~4.3 GB instead of ~8.5 GB, fp32 outputs). `bf16-llm-fp32`: as `bf16`, but the LLM decoder weights are upcast once at load, ViT + input-embedding table stay bf16 (~7 GB, faster, bitwise identical to `bf16`) |
| `--compile` | off | CPU speed option: `torch.compile` (dynamic shapes) of the dense per-layer parts of the ViT and T_llm and of the exact-path LLM forward; attention, K/V cache I/O and host planning stay eager (cos >= 0.99999 vs eager) |
| `--lora PATH\|none` | `$SCREENRET_LORA` | LoRA merged at load, or zero-shot |
| `--lora-merge / --no-lora-merge` | merge | `--no-lora-merge` keeps the LoRA as a separate low-rank path (use in bf16) |

## Paper tables -> commands

See [CODE_MAP.md](CODE_MAP.md) for the full mapping and [CHANGES.md](CHANGES.md) for every change relative to the stock HF Qwen3-VL forward.

Accuracy protocol (the paper's, **unfiltered**): person A, all 4,265 single-hop test questions + all 846 parallel multi-hop
questions (`data/`: T2-T5 3,575 + 415, plus 690 + 431 on the 433 T1-train recordings the LoRA does not train on; no freq2), pool =
63,538 frames (T1-T5 minus bad frames).
- single-hop: rank = 1 + #non-gold frames above the best gold frame; R@k.
- parallel: per evidence group (one or more recordings) rank = 1 + #pool frames above that group's best frame (every pool frame competes,
  including other groups' gold frames); per question recall@k = mean over groups of [rank <= k]; **parallel R@10** = its mean over
  questions (`PR10` in the output). The strict all-groups-within-k number is reported as `PA@k`.
- `make_eval_protocol.py --leak-filter` drops high-leak parallel questions (leak >= 0.65 unless all evidence recordings are desktop
  recordings of person A; 846 -> 725); not used in the paper.

Release check on the earlier protocol (3,575 single-hop + 497 parallel questions incl. freq2; pixel rows: v1 pixel LoRA trained on all
2,837 T1-train questions; OCR row: OCR LoRA). `screenret.evaluate` on the stored frame vectors, with query vectors encoded by
`embed_text queries --no-lora-merge`, reproduces these numbers and the research code's per-question ranks:

| index | single-hop R@1 | parallel R@10 | query encoding |
|---|---|---|---|
| OCR text (RapidOCR, same model, OCR LoRA) | 74.7 | 57.0 | fp16 (bf16: 74.7 / 57.4) |
| screenshot FULL | 77.9 | 62.2 | bf16 (fp16: 77.9 / 62.4) |
| screenshot S+T+L (rho 0.9, final method) | 77.6 | 62.8 | bf16 |

Query vectors are the only precision-sensitive input here (bf16 on Ampere GPUs, fp16 on older ones); frame vectors are the stored runs.

## Timing rows -> commands

One configuration per process (`scripts/run_timing_table.sh` loops over them). Timing is identical for the zero-shot and the (merged)
LoRA model, so the LoRA only matters for the stored embeddings.

| paper row | command |
|---|---|
| Mac GPU (MPS fp16), screenshot rows | `python -m screenret.timing --device mps --lora stage2_pixelA.best_lora.pt --preset <full\|full+pe\|S\|S+T\|S+T+L\|S+G\|S+G+T+L> --sample sample.json --out results/ --tag idle` (no CPU options) |
| CPU idle (32 GB machine), screenshot rows | `... --device cpu --dtype float32 --compile --weight-storage bf16-llm-fp32 --preset ...` |
| CPU in the wild (16 GB laptop), screenshot rows | `... --device cpu --dtype float32 --compile --weight-storage bf16 --preset ...` (fp32 LLM weights swapped on 16 GB) |
| OCR rows | `... --route ocr-rapid` (or `ocr-vision`) `--lora stage2_ocrA.best_lora.pt`, stock settings: no `--compile`; on the CPU `--dtype float32 --weight-storage bf16` |

`--compile` on macOS: if inductor fails with `'algorithm' file not found`, the Command Line Tools install has no libc++ headers next to
clang (only inside the SDK). Point clang at them before running:
`export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1` (`run_timing_table.sh` does this
when the directory exists). The first frames compile (tens of seconds) inside the untimed warm-up; `TORCH_LOGS=recompiles` shows
recompiles.

## Questions (included) and data the scripts expect (not included)

- **Questions** ([`data/`](data/README.md)): retrieval LoRA training / validation questions (1,000 / 315) and the paper's test set
  (4,265 single-hop + 846 parallel multi-hop questions of person A), `splits.json`, `t1_test_uids.json` (the 433 T1-train recordings
  moved to the test set), and the single-hop questions in the input format of `make_eval_protocol.py` (`eval_input_test.json` +
  `texts_test.jsonl`; parallel: `questions_test_parallel.jsonl`).
  How they were built: [`data_pipeline/`](data_pipeline/README.md) (LLM generation, uniqueness judge, filters; reproduces the files
  from the cached LLM outputs).
- `frames.jsonl` (not included): one `{"fid", "uid" (recording), "i" (frame index in the recording), "rel_path"}` per 5-s frame.
- `splits.json` format: `"periods"` (`[{"name", "uids"}]` in time order), `"fids"` per period, `"bad_fids"` (excluded from the pool).
- `eval_input` format: `"new"` = single-hop questions with `qid`, `person`, `row`, `groups`, + a texts JSONL (query text per row);
  parallel questions: `qid`, `person`, `question`, `evidence_groups` [{`uids`, `category`}], `leak`. `main_qids.json` (optional)
  restricts single-hop questions to the older 2,059-question main set.
- Timing samples: `scripts/make_timing_sample.py` (needs `tasks_meta.jsonl` with each recording's machine and start time).
No recordings, frames, embeddings, checkpoints or results are shipped in this repository.

## Busy / idle timing conditions (macOS)

`scripts/run_timing_table.sh` runs every configuration in its own process (`COMPILE=1 WSTORE=...` for the CPU options; OCR rows always
run without `--compile`). `MODE=busy` starts a fixed foreground load
(`scripts/load/load.sh`: Speedometer 3.1 in a separate Chrome profile + a looping 1080p video; assets are downloaded separately, see
the script header). `MODE=idle`: reboot, quit apps, AC power, Low Power Mode off. The MPS allocation cap (`--mps-cap-gb 9.5`) and the
free-memory guard (`--min-avail-gb`) keep a 16 GB Mac from swapping during a run.

## Tests

```bash
pytest tests/                                                       # rules / CLI / LoRA / evaluation (no model)
SCREENRET_TEST_FRAMES=<dir with >= 6 consecutive screenshots> SCREENRET_TEST_DEVICE=mps pytest tests/test_model_consistency.py
SCREENRET_TEST_FRAMES=<dir with >= 14 consecutive screenshots> pytest tests/test_cpu_options.py   # CPU: bf16-llm-fp32 + --compile
```

## Layout

```
screenret/           the one implementation of the method (device-agnostic: cuda / mps / cpu)
  config.py          MethodConfig (component switches + thresholds) and presets
  qwen_embedder.py   minimal Qwen3-VL-Embedding wrapper (processor, chat format, last-token pooling)
  model.py           load + LoRA (merged / unmerged) + bf16 weight storage, linear patch embedding, per-resolution constants (Grid)
  spatial.py         S rules (lt, lt-grid = lt + Grid Keep, pixelprune)
  temporal.py        T_vit: per-stream state, change detection, refresh rule, FramePlan
  vit.py             full / partial ViT forward with cached K/V
  llm.py             LLM sequence, exact prefix reuse, T_llm
  engine.py          preprocess() + Engine.step(): one frame in, one embedding + stats/MAC row out
  cpu_compile.py     --compile: torch.compile of the dense per-layer kernels (vit.KERNELS, llm.KERNELS) + the LLM forward
  macs.py            MAC accounting of the executed work
  ocr.py             OCR baselines (RapidOCR, Apple Vision) + text embedding
  cli.py             shared component / model flags
  timing.py          entry point: on-device latency of one configuration
  embed_stream.py    entry point: stream embedding for accuracy (segmentable)
  merge_segments.py  entry point: segments -> id-indexed frame matrix
  embed_text.py      entry point: OCR-text frame vectors and query vectors
  evaluate.py        entry point: single-hop R@k, parallel recall@k, paired / clustered bootstrap
  sysmon.py          memory / swap / load readings logged with timings
scripts/             sample + protocol builders, query bench, bulk Vision OCR, timing table runner, busy-load scripts
data/                released questions (LoRA train / val, single-hop + parallel test), splits.json, T1 test recordings
data_pipeline/       how data/ was built: splits, diary, LLM question generation, uniqueness judge, filters (see its README)
tests/               unit tests and model consistency tests
```
