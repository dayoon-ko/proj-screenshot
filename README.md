# screen-retrieval: episodic recall over your own screen, from screenshots

Code for the MLSys 2027 submission: private, on-device episodic recall over a personal screen history.

- **Task.** Index: one screenshot every 5 s. Input: a text question about something you did ("Which album's release page did I pull
  up on Discogs?"). Output: the matching frames.
- **Screenshots vs OCR text.** Existing recall tools index the OCR text of each screen. Indexing the screenshot itself with
  **Qwen3-VL-Embedding-2B** retrieves better than the OCR text embedded by the same model, but a full screenshot is slow to encode
  on a laptop.
- **Two redundancies, two fixes.** Much of a screen is blank or flat (backgrounds, margins, whitespace): *spatial* redundancy.
  Consecutive frames are often identical and only small regions change (typing, cursor): *temporal* redundancy. **Token Pruning**
  removes the first within a frame, **KV Reuse** the second across frames.
- **Grid Keep.** Token Pruning shrinks a large flat region (blank page, empty panel) to a single token, so the model loses its size and
  layout; on varied screens (myPCBench) this costs most of the screenshot's advantage. Grid Keep keeps one block per 2x2 tile of each
  flat region instead. No training; it is part of Token Pruning in the paper's method.
- **Consistency adapter.** A second LoRA encodes the pruned frames so that they embed like the full screenshots; questions are still
  encoded by the screenshot LoRA, so both live in one index.

The paper's method = Token Pruning (with Grid Keep) + KV Reuse (ViT + LLM) + Consistency adapter. A *block* is a 32x32-px square
(2x2 ViT patches, merged into one LLM token); a 1280x704 screenshot has 880 blocks. Thresholds compare 8-bit pixel values (8/255).

| step | what it does | exact? |
|---|---|---|
| **Token Pruning** | within a frame, drop a block if every pixel is within 8/255 of its left or upper neighbour block (an order-free version of PixelPrune). Dropped blocks enter neither the ViT nor the LLM | approx. |
| **+ Grid Keep** | a flat region (blocks linked by those left / upper matches) keeps the top-left block of every 2x2-block tile that has no kept block; kept blocks stay at their own positions | approx. |
| **KV Reuse (ViT)** | across frames, a block counts as changed if any pixel differs by more than 8/255 from when it was last computed; the ViT caches each layer's K/V and recomputes only the changed blocks. Every block is recomputed when more than 90 % of the kept blocks changed (a new screen) or every 10 frames | approx. |
| (LLM prefix reuse) | on with KV Reuse (ViT): the LLM is causal, so the previous frame's KV cache is reused up to the first changed token | exact |
| **KV Reuse (LLM)** | after the first change, recompute only the changed visual tokens and the text after them; unchanged visual tokens reuse their cached K/V | approx. |
| **Consistency adapter** | frames are encoded by a second LoRA (same layers and size, initialised from the screenshot LoRA), trained so that pruned frames match the screenshot LoRA's full-frame vectors | trained |
| Linear patch embedding | the ViT's Conv3d patch embedding computed as the identical matrix product (Apple's MPS backend runs that Conv3d very slowly) | exact (same math) |

## Benchmarks and data split

- **PSAI-A**: one person's screen recordings from the PSAI dataset (2,105 recordings of computer tasks, Apr–Sep 2025, one frame every
  5 s). Its questions are in [`data/`](data/README.md). **PSAI-B** (a second person, 946 recordings) and **myPCBench** (787
  computer-use recordings on a VM) are test-only and not part of this release.
- **Questions** are *single* (one recording answers it: where / when / what / how / which value did I type) or *multi* (several
  recordings must be found: which came first, how many times, which took longer, which value was higher). In the code and data
  files multi questions are called *parallel*, because each of their recordings is retrieved on its own.
- **Split by time.** Person A's recordings are sorted by time and cut into five periods with the same number of frames, **T1**
  (earliest) to **T5** (latest). T1 is the training period: 710 recordings for training (*T1-train*) and 79 for model selection
  (*T1-val*). The screenshot LoRA trains on 1,000 questions about T1-train recordings; the 433 T1-train recordings that none of these
  questions points to join the test set. Test = T2–T5 plus those 433 recordings: **4,265 single + 846 multi questions**.
- **Search pool**: every frame of T1–T5 except the frames that show a question's answer as plain text: 63,538 frames. Every question
  searches the whole pool.

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # on Apple silicon use an arm64 Python (not x86 under Rosetta: no MPS)
pip install -r requirements.txt && pip install -e .
pip install pyobjc-framework-Vision pyobjc-framework-Cocoa   # macOS only, for the Apple Vision OCR baseline
```

Tested with Python 3.12, torch 2.8.0, transformers 5.17.0 (Mac: MPS / CPU; server: CUDA 12.6).

## Model and checkpoints

Base model: `Qwen/Qwen3-VL-Embedding-2B` (downloaded from the Hugging Face Hub on first use; `--model-id` takes a local path). All
LoRAs are PEFT LoRAs with r 32 / alpha 32. File names are the names used in our experiments.

| checkpoint | file | layers | trained on | used for |
|---|---|---|---|---|
| **screenshot LoRA** | `mp8_pixel_bal.step50_lora.pt` | ViT attention `qkv` / `proj`, LLM `q/k/v/o_proj` | the 1,000 questions of `data/questions_train.jsonl` (contrastive, with hard negatives) | full-screenshot frame vectors and every question vector of the screenshot index |
| **Consistency adapter** | `mp12_pixel_bal_padapt.step200_lora.pt` | same as the screenshot LoRA | the same questions; teacher = the screenshot LoRA (see "Consistency adapter training") | frame vectors of the paper's method |
| **OCR LoRA** | `stage2_ocrA.best_lora.pt` | LLM `q/k/v/o_proj` | RapidOCR text of the same screenshots | frame and question vectors of the OCR-text index |
| earlier screenshot LoRA | `stage2_pixelA.best_lora.pt` | as the screenshot LoRA | all 2,837 T1-train questions | the timing table and the release check below (timing does not depend on which LoRA is loaded) |

Download: **`<LORA_URL — to be added at camera-ready>`**.

- Every entry point takes `--lora PATH` (or `$SCREENRET_LORA`) or `--lora none` for the zero-shot model; there is no silent default.
- By default a LoRA is **merged into the weights at load**, so the compute graph (and the timing) is the zero-shot model's.
- In **bf16** pass `--no-lora-merge`: rounding `W + B A` to bf16 drops much of the low-rank update, so the LoRA then runs as a
  separate fp32 path (`x W^T + (x A^T) B^T`, as PEFT computes it). Merging is fine in fp16 / fp32 (the Mac default).
- Training code: the Consistency adapter's is included (`scripts/train_consistency_adapter.py`); the screenshot / OCR LoRAs' is not.

## Quick start

```bash
export SCREENRET_FRAMES=/path/to/frames_5s                              # root of the screenshot files
export SCREENRET_LORA=/path/to/mp12_pixel_bal_padapt.step200_lora.pt   # Consistency adapter: frames of the paper's method
LORA=/path/to/mp8_pixel_bal.step50_lora.pt                              # screenshot LoRA: full frames and questions

# on-device latency of one configuration (one per process; warm-up frames untimed); see "Latency" below
python -m screenret.timing --device mps --preset S+G+T+L --sample sample.json --gap 0.3 --out results/ --tag idle
python -m screenret.timing --device cpu --dtype float32 --compile --weight-storage bf16-llm-fp32 --preset S+G+T+L --sample sample.json --out results_cpu/ --tag idle

# accuracy: embed the frames in time order (one continuous stream, cut into 4 segments that run in parallel), merge, evaluate
python scripts/make_stream_order.py --splits data/splits.json --frames-meta frames.jsonl --out order.tsv
for i in 0 1 2 3; do python -m screenret.embed_stream --device cuda --dtype bfloat16 --no-lora-merge --preset S+G+T+L --frames-list order.tsv --segment $i/4 --out runs/method_seg$i.npz; done
python -m screenret.merge_segments runs/method_seg*.npz --n-rows 80459 --out runs/method_frames.npy   # one row per frame id of frames.jsonl
for i in 0 1 2 3; do python -m screenret.embed_stream --device cuda --dtype bfloat16 --no-lora-merge --preset full --lora $LORA --frames-list order.tsv --segment $i/4 --out runs/full_seg$i.npz; done
python -m screenret.merge_segments runs/full_seg*.npz --n-rows 80459 --out runs/full_frames.npy
python scripts/make_eval_protocol.py --eval-input data/eval_input_test.json --texts data/texts_test.jsonl --parallel data/questions_test_parallel.jsonl \
    --splits data/splits.json --frames-meta frames.jsonl --out protocol.json --queries-out protocol_queries.jsonl   # 4,265 single + 846 multi
python -m screenret.embed_text queries --device cuda --dtype bfloat16 --no-lora-merge --lora $LORA --texts protocol_queries.jsonl --out q_lora.npy
python -m screenret.evaluate --protocol protocol.json --queries q_lora.npy --frames FULL=runs/full_frames.npy METHOD=runs/method_frames.npy --reference FULL
```

Every command has `--help`.

## Configurations (presets and flags)

A `--preset` picks a configuration (default: the paper's method); explicit flags override it. The same flags work for `timing` and
`embed_stream`. Output files and logs name a configuration by its *tag*.

| preset | configuration | tag in outputs |
|---|---|---|
| `full` | full screenshot, stock Conv3d patch embedding | `FULL` |
| `full+pe` | full screenshot, linear patch embedding | `FULL+pe` |
| `S` | Token Pruning without Grid Keep | `LT8+pe` |
| `S+T` | + KV Reuse (ViT) | `LT8T+pe` |
| `S+T+L` | + KV Reuse (LLM) | `LT8TL+pe` |
| `S+G` | Token Pruning with Grid Keep | `LT8t2+pe` |
| `S+G+T+L` | Token Pruning with Grid Keep + KV Reuse (ViT + LLM): **the paper's method** (load the Consistency adapter with `--lora`) | `LT8t2TL+pe` |

Tag letters: `LT8` = Token Pruning at threshold 8/255 ("left-or-top" rule), `t2` = Grid Keep (2x2 tiles), `T` = KV Reuse (ViT),
`L` = KV Reuse (LLM), `+pe` = linear patch embedding; a suffix like `_r0.6k10d8` records non-default KV Reuse settings.

| flag | default | meaning |
|---|---|---|
| `--spatial {none,lt,lt-grid,pixelprune}` | lt-grid | Token Pruning rule: `lt` = without Grid Keep, `lt-grid` = with Grid Keep, `pixelprune` = the original PixelPrune rule (raster order), for comparison |
| `--tau` | 8 | Token Pruning threshold in 8-bit grey levels (8 = 8/255) |
| `--temporal-vit / --no-temporal-vit` | on | KV Reuse (ViT) (+ exact LLM prefix reuse) |
| `--delta` | 8 | KV Reuse change threshold in grey levels |
| `--refresh-rho` | 0.9 | recompute every block when more than this fraction of the kept blocks changed |
| `--refresh-every` | 10 | recompute every block every K frames |
| `--approx-llm / --no-approx-llm` | on | KV Reuse (LLM) (needs KV Reuse (ViT)) |
| `--linear-patch-embed / --no-...` | on | linear patch embedding (identical math; matters for speed on MPS only) |
| `--route {screenshot,ocr-rapid,ocr-vision}` | screenshot | `timing` only: OCR-text baselines (RapidOCR on the CPU / Apple Vision OCR) + text embedding |
| `--device`, `--dtype` | auto, fp16 (fp32 on CPU) | `cuda`, `mps`, `cpu` |
| `--weight-storage {native,bf16,bf16-llm-fp32}` | native | CPU + fp32 only. `bf16`: Linear weights stored in bf16 and upcast per call (~4.3 GB instead of ~8.5 GB, same fp32 outputs). `bf16-llm-fp32`: as `bf16`, but the LLM weights are upcast once at load (~7 GB, faster, bitwise identical to `bf16`) |
| `--compile` | off | CPU speed option: `torch.compile` of the dense per-layer parts of the ViT and the LLM; attention, KV cache handling and planning stay eager (cos >= 0.99999 vs eager) |
| `--lora PATH\|none` | `$SCREENRET_LORA` | LoRA to load, or the zero-shot model |
| `--lora-merge / --no-lora-merge` | merge | `--no-lora-merge` keeps the LoRA as a separate low-rank path (use in bf16) |

## Evaluation

See [CODE_MAP.md](CODE_MAP.md) for every paper table -> command and [CHANGES.md](CHANGES.md) for every change relative to the stock
Hugging Face Qwen3-VL forward.

The paper uses all 4,265 single + 846 multi test questions of PSAI-A over the 63,538-frame pool (no question filter). A *gold* frame
is any frame of a recording that answers the question.
- single: rank = 1 + number of non-gold frames scoring above the best gold frame; **R@k** = fraction of questions with rank <= k.
- multi: each needed recording (or set of interchangeable recordings) is ranked separately: rank = 1 + number of pool frames
  scoring above its best frame (every other frame competes, including the other recordings of the same question). Per question,
  recall@k = fraction of its recordings with rank <= k; **multi R@10** = the mean over questions (printed as `PR10`). `PA10` is the
  stricter fraction of questions whose recordings are all within the top 10.
- `evaluate` also prints differences to a reference configuration with paired bootstrap confidence intervals (questions resampled,
  and recordings resampled as clusters).
- `make_eval_protocol.py --leak-filter` (not used in the paper) drops multi questions whose words mostly appear as text on their
  answer screens (846 -> 725).

Release check on an earlier version of the test set (3,575 single + 497 multi questions, including a since-dropped "which did I use
more often" type; the earlier screenshot LoRA): `screenret.evaluate` on the stored frame vectors, with question vectors from
`embed_text queries --no-lora-merge`, reproduces our research code's numbers and per-question ranks:

| index | single R@1 | multi R@10 | question encoding |
|---|---|---|---|
| OCR text (RapidOCR, same model, OCR LoRA) | 74.7 | 57.0 | fp16 (bf16: 74.7 / 57.4) |
| Full screenshot | 77.9 | 62.2 | bf16 (fp16: 77.9 / 62.4) |
| Token Pruning + KV Reuse (without Grid Keep) | 77.6 | 62.8 | bf16 |

Question vectors are the only precision-sensitive input here (bf16 on Ampere GPUs, fp16 on older ones); frame vectors are stored.

## Consistency adapter training

`scripts/train_consistency_adapter.py` trains the adapter from the screenshot LoRA, which is frozen as the teacher and also the
starting point of the adapter. In each step, the frames of 12 training questions (the correct frame and 4 hard negatives each) and
12 random T1-train frames are encoded with Token Pruning by the adapter and pulled towards the teacher:
- consistency: 1 - cos(adapter vector of the pruned frame, teacher vector of the full frame);
- ranking: KL divergence between the teacher's and the adapter's question-to-frame score distributions (temperature 0.02);
- margins: MarginMSE on (correct frame - hard negative from the same website) score margins.
Token Pruning uses threshold 8/255 on 80 % of the steps and 4, 12 or 16 on the rest. Paper setting = the script's defaults (lr 1e-5,
300 steps); the released adapter is step 200, the best T1-val R@10. One CUDA GPU with bf16; ~14 GB peak, ~20 min per 50 steps on
an RTX A6000.

```bash
# inputs: zero-shot frame vectors (to pick each question's correct frame and hard negatives), the teacher's full-frame vectors,
# zero-shot vectors of the training questions
python -m screenret.embed_stream --device cuda --dtype float16 --no-linear-patch-embed --preset full --lora none \
    --frames-list order.tsv --out runs/zs.npz && python -m screenret.merge_segments runs/zs.npz --n-rows 80459 --out zs_frames.npy
python -m screenret.embed_stream --device cuda --dtype bfloat16 --no-lora-merge --no-linear-patch-embed --preset full --lora $LORA \
    --frames-list order.tsv --out runs/teacher.npz && python -m screenret.merge_segments runs/teacher.npz --n-rows 80459 --out teacher_frames.npy
python -c "import json; [print(json.dumps({'text': json.loads(l)['question']})) for l in open('data/questions_train.jsonl')]" > train_texts.jsonl
python -m screenret.embed_text queries --device cuda --dtype float16 --lora none --texts train_texts.jsonl --out zs_queries.npy
# run `check` and `smoke` first (same arguments): the script's pruned encoding must equal the engine's, cos 1.000000
python scripts/train_consistency_adapter.py train --lora $LORA --train data/questions_train.jsonl --val data/questions_val.jsonl \
    --splits data/splits.json --frames-meta frames.jsonl --frames-root $SCREENRET_FRAMES --diary work/diary.jsonl \
    --tasks-meta psai_tasks_meta.jsonl --zs-frames zs_frames.npy --zs-queries zs_queries.npy --teacher-frames teacher_frames.npy \
    --out-dir runs/adapter   # -> runs/adapter/consistency_adapter.step200_lora.pt, .best_lora.pt, .json
```

`work/diary.jsonl` (each recording's website / app and category) comes from step 2 of [`data_pipeline/`](data_pipeline/README.md);
`psai_tasks_meta.jsonl` is PSAI's per-recording metadata (task names are used to skip near-duplicate tasks as negatives).

Release check (RTX A6000, bf16, the same input vectors as our research run): `check` gives cos 1.000000 between the script's pruned
encoding and the engine's, and between the gradient-checkpointed and plain paths. Re-running the paper setting follows the research
run on T1-val (315 questions over the T1-val frames, pruned frames, teacher question vectors; one question = 0.3 points):

| step | research run: mean cos to teacher / R@1 / R@10 | this script |
|---|---|---|
| 0 (= screenshot LoRA) | 0.9599 / 80.3 / 88.3 | 0.9598 / 80.0 / 88.6 |
| 50 | 0.9648 / 80.0 / 87.9 | 0.9647 / 78.4 / 88.9 |
| 100 | 0.9653 / 79.4 / 88.9 | 0.9642 / 78.7 / 88.6 |

## Latency

One configuration per process (`scripts/run_timing_table.sh` loops over them). Timing does not depend on which LoRA is loaded.

| paper row | command |
|---|---|
| Mac GPU (MPS fp16), screenshot | `python -m screenret.timing --device mps --lora stage2_pixelA.best_lora.pt --preset <full\|full+pe\|S\|S+T\|S+T+L\|S+G\|S+G+T+L> --sample sample.json --out results/ --tag idle` |
| Mac CPU, idle (32 GB machine), screenshot | `... --device cpu --dtype float32 --compile --weight-storage bf16-llm-fp32 --preset ...` |
| Mac CPU, in use (16 GB laptop), screenshot | `... --device cpu --dtype float32 --compile --weight-storage bf16 --preset ...` (fp32 LLM weights would swap on 16 GB) |
| OCR text | `... --route ocr-rapid` (or `ocr-vision`) `--lora stage2_ocrA.best_lora.pt`, stock settings: no `--compile`; on the CPU `--dtype float32 --weight-storage bf16` |

*Idle*: reboot, quit apps, AC power, Low Power Mode off (`MODE=idle`). *In use*: `MODE=busy` starts a fixed foreground load
(`scripts/load/load.sh`: the Speedometer 3.1 browser benchmark in a separate Chrome profile + a looping 1080p video; assets are
downloaded separately, see the script header). The MPS memory cap (`--mps-cap-gb 9.5`) and the free-memory guard (`--min-avail-gb`)
keep a 16 GB Mac from swapping. Timing samples: `scripts/make_timing_sample.py` (100 windows of 20 consecutive frames, each after 10
untimed frames).

`--compile` on macOS: if it fails with `'algorithm' file not found`, the Command Line Tools have no libc++ headers next to clang.
Point clang at them: `export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1`
(`run_timing_table.sh` does this). The first frames compile (tens of seconds) inside the untimed warm-up.

## Data

Included ([`data/`](data/README.md); built by [`data_pipeline/`](data_pipeline/README.md) with an LLM, a uniqueness judge and
filters):
- training / validation questions (1,000 / 315) and the test questions (4,265 single + 846 multi);
- `splits.json`: the periods T1–T5 (`periods`, `fids` = frame ids per period), the T1-train / T1-val recordings and `bad_fids` (the
  answer-revealing frames left out of the pool); `t1_test_uids.json`: the 433 T1-train recordings that are test recordings;
- `eval_input_test.json` + `texts_test.jsonl`: the single questions in the input format of `make_eval_protocol.py` (multi questions
  are read from `questions_test_parallel.jsonl`).

Not included (from PSAI): the screenshots and `frames.jsonl`, one `{"fid", "uid" (recording id), "i" (frame index in the recording),
"rel_path"}` per 5-s frame. No recordings, frames, embeddings, checkpoints or results are in this repository.

## Tests

```bash
pytest tests/                                                       # pruning rules, flags, LoRA, evaluation (no model)
SCREENRET_TEST_FRAMES=<dir with >= 6 consecutive screenshots> SCREENRET_TEST_DEVICE=mps pytest tests/test_model_consistency.py
SCREENRET_TEST_FRAMES=<dir with >= 14 consecutive screenshots> pytest tests/test_cpu_options.py   # CPU: bf16-llm-fp32 + --compile
```

## Layout

```
screenret/           the one implementation of the method (cuda / mps / cpu)
  config.py          configuration (which steps are on, thresholds) and presets
  qwen_embedder.py   minimal Qwen3-VL-Embedding wrapper (processor, chat format, last-token pooling)
  model.py           loading, LoRA (merged / separate), bf16 weight storage, linear patch embedding, per-resolution constants
  spatial.py         Token Pruning rules (without / with Grid Keep, PixelPrune)
  temporal.py        KV Reuse: per-stream state, change detection, when to recompute everything
  vit.py             ViT forward over the kept blocks, with cached K/V for unchanged ones
  llm.py             LLM input sequence, exact prefix reuse, KV Reuse (LLM)
  engine.py          one frame in, one vector + work / MAC counts out
  cpu_compile.py     --compile
  macs.py            MAC counting of the work actually done
  ocr.py             OCR-text baselines (RapidOCR, Apple Vision) + text embedding
  cli.py             shared flags
  timing.py          entry point: on-device latency of one configuration
  embed_stream.py    entry point: frame vectors of a stream, for accuracy
  merge_segments.py  entry point: stream segments -> one matrix indexed by frame id
  embed_text.py      entry point: OCR-text frame vectors and question vectors
  evaluate.py        entry point: single R@k, multi recall@k, bootstrap intervals
  sysmon.py          memory / swap / load readings logged with timings
scripts/             stream order, test protocol, timing samples and runner, load scripts, question-time benchmark,
                     Apple Vision OCR, Consistency adapter training
data/                released questions and splits
data_pipeline/       how data/ was built (see its README)
tests/               unit tests and model consistency tests
```
