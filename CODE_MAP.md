# Code map: paper -> files -> commands

Section names follow the submission's structure; adjust numbering at camera-ready.

## Method

| paper | file(s) | notes |
|---|---|---|
| Embedding model, last-token pooling, instructions | `screenret/qwen_embedder.py`, `screenret/config.py` (`QUERY_INSTRUCTION`) | frames / OCR text: default instruction; queries: retrieval instruction |
| Blocks (32x32 px = 2x2 patches = 1 LLM token), per-resolution constants | `screenret/model.py` (`Grid`, `grid`) | 1280x704 -> 880 blocks |
| S: spatial pruning (LT8 left-or-upper rule; PixelPrune for comparison) | `screenret/spatial.py` (`lt_keep_mask`) | `--spatial lt --tau 8` |
| Grid Keep (part of Token Pruning in the paper's method: one block per flat region per 2x2-block tile kept) | `screenret/spatial.py` (`gridkeep_keep_mask`) | `--spatial lt-grid` (presets `S+G`, `S+G+T+L`) |
| T_vit: change detection, refresh rule (rho 0.9, K 10), per-stream caches | `screenret/temporal.py` | `--temporal-vit --delta 8 --refresh-rho 0.9 --refresh-every 10` |
| T_vit: partial ViT with cached K/V, per-block merger / DeepStack cache | `screenret/vit.py` | |
| Exact LLM prefix reuse, T_llm approximate K/V reuse | `screenret/llm.py` | `--approx-llm` |
| +pe (MPS Conv3d fix) | `screenret/model.py` (`embed_patches`) | `--linear-patch-embed` |
| Frame pipeline | `screenret/engine.py` (`preprocess`, `Engine.step`) | |
| MAC accounting ("MAC fraction") | `screenret/macs.py`; per-frame rows from `engine.stats_row` | FULL 1280x704 frame = 3.084e12 MACs |
| Retrieval LoRAs (pixel "v1" recipe: r 32 / alpha 32, ViT qkv/proj + LLM q/k/v/o; OCR: LLM q/k/v/o) | `screenret/model.py` (`merge_lora`, `attach_lora`) | merged by default; `--no-lora-merge` = PEFT's unmerged fp32 path (bf16); training code not released; the paper's pixel LoRA is trained on the 1,000 questions of `data/questions_train.jsonl` |
| Consistency adapter (a second LoRA, same targets / size, initialised from the pixel LoRA, that encodes the pruned frames; queries stay on the pixel LoRA) | no new code: frames `--lora <adapter.pt>` in `embed_stream` / `timing`, queries `--lora <pixel LoRA.pt>` in `embed_text queries` | training code not released |
| CPU speed options | `screenret/cpu_compile.py` (`--compile`; kernels in `vit.KERNELS`, `llm.KERNELS`), `screenret/model.py` (`--weight-storage bf16 / bf16-llm-fp32`) | off by default |
| OCR baselines (RapidOCR, Apple Vision accurate) + same-model text embedding | `screenret/ocr.py` | |

## Accuracy (main table, component ablation)

Protocol (unfiltered): person A, 4,265 single-hop test questions + 846 parallel multi-hop questions (T2-T5 + the 433 T1-train
recordings the LoRA does not train on; no freq2) x 63,538-frame pool (T1-T5 minus bad frames). Single-hop: gold = any frame of the question's recording; rank = 1 + #non-gold frames above the best gold frame; R@k.
Parallel: per evidence group rank = 1 + #pool frames above the group's best frame (all frames compete, incl. other groups' gold);
parallel R@10 = mean over questions of the per-question mean over groups of [rank <= 10] (`PR10`; strict all-groups `PA10`).
Paired bootstrap (B = 10,000, seed 0) and recording-clustered bootstrap (2,000), per question set.

| step | command |
|---|---|
| stream order (63,644 frames, 2,105 recordings, continuous) | `python scripts/make_stream_order.py --splits splits.json --frames-meta frames.jsonl --out order.tsv` |
| screenshot vectors, one config, 4 parallel segments | `python -m screenret.embed_stream --device cuda --preset S+T+L --lora v1.pt --frames-list order.tsv --segment i/4 --out runs/stl_seg<i>.npz` |
| zero-shot arm (paper: fp16, Conv3d on CUDA) | add `--lora none --dtype float16 --no-linear-patch-embed` |
| LoRA arm (paper: bf16, PEFT unmerged) | `--lora v1.pt --dtype bfloat16 --no-lora-merge --no-linear-patch-embed` |
| ablation rows | `--preset full` / `S` / `S+T` / `S+T+L`; rho 0.6 rows: `--refresh-rho 0.6` |
| merge | `python -m screenret.merge_segments runs/stl_seg*.npz --n-rows <#frames> --out runs/stl_frames.npy` |
| OCR-text vectors | `python scripts/vision_ocr_extract.py ...` (or any `{"id","text"}` JSONL), then `python -m screenret.embed_text ocr --lora <OCR LoRA \| none> --ocr-jsonl ocr.jsonl --n-rows <#frames> --out ocr_frames.npy` |
| test questions (4,265 single + 846 parallel) and LoRA training questions (1,000) | `data/` (built by `data_pipeline/`, see its README; final split = `s9_final_split.py`) |
| protocol (4,265 single + 846 parallel) | `python scripts/make_eval_protocol.py --eval-input data/eval_input_test.json --texts data/texts_test.jsonl --parallel data/questions_test_parallel.jsonl --splits data/splits.json --frames-meta frames.jsonl --out protocol.json --queries-out protocol_queries.jsonl` (`--leak-filter`: 725 parallel; `--main-qids main_qids.json`: old 2,059 single-hop set) |
| query vectors (per arm: pixel LoRA for screenshot rows, OCR LoRA for the OCR row) | `python -m screenret.embed_text queries --device cuda --dtype bfloat16 --no-lora-merge --lora <LoRA> --texts protocol_queries.jsonl --out q.npy` |
| single-hop R@1/5/10, parallel PR@k / PA@k + deltas vs FULL | `python -m screenret.evaluate --protocol protocol.json --queries q.npy --frames FULL=... S=... S+T=... S+T+L=... --reference FULL --out table.json` |

Main-table cells: R@k and dR@k [95 % CI] = `evaluate` output; MAC fraction / refresh rate of each row = `summary` in the
`merge_segments` JSON (`mac_frac`, `refresh_rate`).

## On-device cost (Mac)

| paper | command |
|---|---|
| timing samples (100 windows x 20 frames + 10 untimed context frames each) | `python scripts/make_timing_sample.py --frames-meta frames.jsonl --tasks-meta tasks_meta.jsonl --node <PC A> [--test-fids test_fids.json --seed 20261015] --out sample.json` |
| latency table, MPS (all rows) | `SAMPLE=sample.json FRAMES=... LORA=stage2_pixelA.best_lora.pt OCR_LORA=stage2_ocrA.best_lora.pt MODE=idle scripts/run_timing_table.sh` |
| one GPU row (no CPU options) | `python -m screenret.timing --device mps --lora stage2_pixelA.best_lora.pt --preset S+T+L --sample sample.json --out results/ --tag idle` |
| CPU idle (32 GB) | `... --device cpu --dtype float32 --compile --weight-storage bf16-llm-fp32` (script: `DEVICE=cpu DTYPE=float32 COMPILE=1 WSTORE=bf16-llm-fp32`) |
| CPU in the wild (16 GB laptop) | `... --device cpu --dtype float32 --compile --weight-storage bf16` (fp32 LLM weights swapped on 16 GB) |
| OCR rows (stock forward, OCR LoRA) | `python -m screenret.timing --route ocr-rapid` / `--route ocr-vision` `--lora stage2_ocrA.best_lora.pt` (no `--compile`; CPU: `--dtype float32 --weight-storage bf16`) |
| `--compile` on a Mac whose Command Line Tools lack libc++ headers (`'algorithm' file not found`) | `export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1` |
| Conv3d vs linear patch embedding (+pe) | rows `full` vs `full+pe` |
| latency columns: mean / median / p90 / p99, refresh vs non-refresh median, prep / ViT / LLM split, MAC fraction, peak RSS / device memory, swap | `summary` of `results/<tag>_<name>.json` |
| query cost vs corpus size (encode + exact top-k up to 2.1M vectors) | `python scripts/query_bench.py --device mps --out query_bench.json` |

## Verification of this release

| check | how |
|---|---|
| embeddings / stats / MACs identical to the reference implementation | `tests/test_model_consistency.py` (self-consistency); the bitwise comparison against the original research code is described in the release notes |
| CPU options | `tests/test_cpu_options.py`: `bf16-llm-fp32` bitwise equal to `bf16`; `--compile` cos >= 0.99999 vs eager (FULL+pe, S, S+T, S+T+L) |
| unmerged LoRA = PEFT | `tests/test_lora.py` (toy layers, bitwise vs `peft` when installed) |
| evaluation rules | `tests/test_evaluate.py` (toy pool); on the stored vectors the per-question ranks equal the research code's rank rule |
