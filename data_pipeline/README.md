# Question pipeline

How the released question files in [`../data`](../data) were built from the PSAI screen recordings: training questions for the
retrieval LoRA, the single-hop test set and the parallel multi-hop test set, and the final train / test split (step 9). Every step is a script with `--help`; LLM steps
cache their raw outputs (resumable files given as arguments), so a re-run with the paper's caches reproduces the released files
byte for byte without new LLM calls.

LLM calls go through the `claude` CLI (`claude -p`, no tools; the paper used Claude Opus 4.8 for the training questions and Claude
Opus 5.5 for everything else). Prompts used verbatim are in [`prompts/`](prompts) (the training-question prompt is inline in
`s3_train_questions.py`).

## Inputs (PSAI, not shipped)

| file | content |
|---|---|
| `psai_tasks_meta.jsonl` | one row per recording: `unique_data_id`, `taskId`, `task_name` (instruction), `application_website`, `category`, `reasoning_steps`, `node` (machine), `start_time`, `stop_time` |
| `frames.jsonl` | one `{"fid", "uid", "i", "rel_path"}` per 5-s frame |
| `ocr_all.jsonl` | `{"id": fid, "text"}` RapidOCR text per frame (answer-revealing frames, parallel leak score) |

Person A = machine `DESKTOP-GFL3B1N` (test person); person B = machine `Learno` (its questions are generated with A's but not released).

## Steps

| step | script | output |
|---|---|---|
| 1. periods T1..T5 (equal frame counts), T1-train / T1-val, twins, answer-revealing frames | `s1_splits.py` | `splits.json` |
| 2. diary: one event per recording (cleaned instruction, steps, time) | `s2_diary.py` | `diary.jsonl` |
| 3. training questions (v1 recipe, per task definition) + split assignment | `s3_train_questions.py` | `questions_train_all.jsonl` (2,837), `questions_val.jsonl` |
| 4. task-text vectors for the competitor search | `s4_task_embeddings.py` + `python -m screenret.embed_text ocr --lora none` | `task_emb.npy` |
| 5. candidate test questions per event (where / when / what / how / anchor + input values) | `s5_test_generate.py` | `scope.json`, `candidates.jsonl` |
| 6. uniqueness judge (+ input-leak judge, what rule) | `s6_uniqueness.py` | `judged.jsonl` |
| 7. single-hop test set (T2..T5) | `s7_build_single.py --person A` | `questions_test_single_T2T5.jsonl`, `single_all.jsonl` |
| P. parallel multi-hop test set (T2..T5) | `p1_action_labels.py` ... `p4_parallel_set.py` (see below) | `questions_test_parallel_T2T5.jsonl` |
| 9. final split: 1,000 training questions, the 433 unused T1-train recordings + their questions join the test set, freq2 dropped | `s9_final_split.py` | `questions_train.jsonl`, `t1_test_uids.json`, `questions_test_single.jsonl`, `questions_test_parallel.jsonl` |
| 8. evaluation input for `scripts/make_eval_protocol.py` (run after step 9) | `s8_eval_input.py` | `eval_input_test.json`, `texts_test.jsonl` |

### Training questions (steps 3) For every unique task definition an LLM sees only the instruction, the app/website and the step list
and writes four first-person questions: **where** (which app/site/tool, without naming it), **when**, **how** (which control, menu or
sequence) and **which** (one detail only readable from the screen). Gold = all recordings of the task (twins merged); identical texts
across tasks share their recordings. A question goes to T1-train / T1-val only if all its gold recordings lie in that part of T1;
where-questions that name the app/site are dropped. 2,837 train / 315 val questions (person A); step 9 keeps 1,000 of the 2,837.

### Single-hop test questions (steps 5-7)

Scope: person A's events in T2..T5 with at least two usable frames (1,315 events; person B's events go through the same steps).
Per event the LLM writes (`prompts/test_task_questions.txt`) where / when / what / how questions and a short cue that refers to the
task without site or time, and (`prompts/test_input_questions.txt`) up to two questions whose answer is a value the person typed,
chose or set (with the verbatim instruction span that holds it). No how-question without recorded steps, no where-question for a
generic site. The ground truth comes from the diary (site, time) or from the LLM (what summary, how procedure, input value).

Filters, in order:
1. **rules**: a where-question naming the site, a when-question stating a date or time, an input answer not found in the instruction;
2. **uniqueness** (`prompts/judge_uniqueness.txt`): for every competitor task of the same person (repeats, same site on the same day,
   similar tasks; at most 30) the judge says which questions it also fits and whether it gives the same answer. A match with a
   different answer deletes the question; a match with the same answer adds that recording to the gold. "what" questions (cue = site +
   time) are deleted if another task of that site happened in the same part of the day;
3. **input leak** (`prompts/judge_input_leak.txt`): input questions whose wording gives the answer away.

Result (person A, T2..T5): 3,575 questions = where 1,088, when 1,139, input 1,214, what 132, how 2 (step 9 adds 690 T1 questions).

### Parallel multi-hop test questions (step P)

Hops are verified single-hop facts (kept anchors and input values from `single_all.jsonl`) or diary records; links are chosen by
rules and the LLM only words and checks the question (shared code in `parallel_lib.py`, prompts `prompts/parallel_*.txt`).
- `p1_action_labels.py`: one action-family label per event ("create trip plan"), all events of a person on one site in the same call,
  so that counts over (site, action) are exact.
- `p2_multihop.py`: **order** (two anchored tasks on the same day, >= 15 min apart), **compare** (two anchored tasks with comparable
  numeric input values), **aggregate** ("how many times did I ... on <site> in <month>", 2-5 events, exact). Sequential specs are
  still drawn (they fix the random state and the spec ids) but not worded or released (`--with-sequential` turns them on).
- `p3_parallel_extra.py`, three rounds: label / site aggregates by month and week + typed compares; day-window aggregates; **freq2**
  ("did I use X or Y more often in <month>") and **duration** ("which of tasks X and Y took longer").
- Checks: answer string not in the question; drop-one (with all evidence + same-day distractors the LLM must recover the answer,
  without any one hop it must not); aggregates and freq2 by two independent recounts that must return exactly the group.
- `p4_parallel_set.py`: concatenates the kept questions and adds a lexical `leak` score (question words found in the gold
  recordings' OCR text; reported, not used to filter).

497 person-A questions: aggregate 191, order 100, duration 100, freq2 82, compare 24 (step 9 drops freq2 -> 415).

### Final split (step 9)

T1-train holds 66 % desktop questions (1,877 of 2,837), the test periods ~4 % desktop frames. The retrieval LoRA therefore trains on
every web question (960) + 40 desktop questions sampled to that 4 % share (numpy `default_rng(1)` over the desktop questions in s3
file order; category = diary category of the first gold recording) = `questions_train.jsonl`. T1-val is unchanged. The 433 T1-train
recordings that are no training question's first gold recording (all desktop) join the test set (`t1_test_uids.json`) with the
questions of the T1 question run below: single-hop questions with a content cue (`cue_type` content / mixed) whose gold recordings
are all in T1-train and whose first one is in the 433 (690 questions on 344 recordings), and parallel questions whose evidence
recordings are all in the 433 (431). T1 qids get the prefix `T1:` (68 T1 parallel qids equal T2..T5 qids). freq2 questions are
dropped everywhere. Test set: 4,265 single-hop + 846 parallel questions; the pool stays T1..T5 minus `bad_fids` (63,538 frames).

T1 question run (input of step 9, not shipped): the research versions of steps 5-7 and P1-P3 on person A's T1-train / T1-val events
(single-hop generation and the p1 / p2 labels and wording with Claude Sonnet, uniqueness judge with Claude Opus 5.5), then
`p4_parallel_set.py` on its p2 + p3 outputs for the `leak` score. Its outputs are 1,647 single-hop and 987 parallel questions;
step 9 keeps the subsets above. The released command block does not regenerate this run.

## Reproduce the released files

```bash
P=data_pipeline; W=work; D=/path/to/psai; T1=/path/to/t1_question_run
python $P/s1_splits.py --meta $D/psai_tasks_meta.jsonl --frames-meta $D/frames.jsonl --ocr $D/ocr_all.jsonl --out $W/splits.json
python $P/s2_diary.py --meta $D/psai_tasks_meta.jsonl --frames-meta $D/frames.jsonl --splits $W/splits.json --out $W/diary.jsonl
python $P/s3_train_questions.py --meta $D/psai_tasks_meta.jsonl --splits $W/splits.json --raw $W/slot_questions_opus_raw.jsonl $W/slot_questions_opus.jsonl \
    --out $W/questions_v1.jsonl --train-out $W/questions_train_all.jsonl --val-out data/questions_val.jsonl
python $P/s4_task_embeddings.py --diary $W/diary.jsonl --out $W/task_texts.jsonl
python -m screenret.embed_text ocr --lora none --device cuda --ocr-jsonl $W/task_texts.jsonl --n-rows 3051 --out $W/task_emb.npy
python $P/s5_test_generate.py --diary $W/diary.jsonl --raw-task $W/gen_task_raw.jsonl --raw-input $W/gen_input_raw.jsonl --scope-out $W/scope.json --out $W/candidates.jsonl
python $P/s6_uniqueness.py --candidates $W/candidates.jsonl --scope $W/scope.json --diary $W/diary.jsonl --task-emb $W/task_emb.npy \
    --raw $W/judge_all.jsonl --raw-leak $W/leak_all.jsonl --out $W/judged.jsonl
python $P/s7_build_single.py --judged $W/judged.jsonl --single-all $W/single_all.jsonl --person A --out $W/questions_test_single_T2T5.jsonl
# parallel
python $P/p1_action_labels.py --diary $W/diary.jsonl --out $W/action_labels.jsonl
python $P/p2_multihop.py --diary $W/diary.jsonl --single-all $W/single_all.jsonl --action-labels $W/action_labels.jsonl --raw $W/multihop_raw.jsonl \
    --checks $W/multihop_checks.jsonl --recount $W/multihop_recount.jsonl --out-specs $W/multihop_specs.jsonl --out-all $W/multihop_all.jsonl \
    --out $W/questions_multihop.jsonl
C="--diary $W/diary.jsonl --single-all $W/single_all.jsonl --action-labels $W/action_labels.jsonl --p2-all $W/multihop_all.jsonl"
for r in extra extra2 extra3; do
  case $r in
    extra)  X="--exclude-specs $W/multihop_specs.jsonl --mid-prefix MX" ;;
    extra2) X="--exclude-specs $W/multihop_specs.jsonl $W/extra_specs.jsonl --mid-prefix MX2 --windows day --no-compare" ;;
    extra3) X="--exclude-specs $W/multihop_specs.jsonl $W/extra_specs.jsonl $W/extra2_specs.jsonl --mid-prefix MX3 --windows '' --no-compare --kinds freq2,dur2" ;;
  esac
  eval python $P/p3_parallel_extra.py $C $X --raw $W/multihop_${r}_raw.jsonl --checks $W/multihop_${r}_checks.jsonl --recount $W/multihop_${r}_recount.jsonl \
      --out-specs $W/${r}_specs.jsonl --out-all $W/${r}_all.jsonl --out $W/questions_multihop_${r}.jsonl
done
python $P/p4_parallel_set.py --multihop $W/questions_multihop.jsonl --extra $W/questions_multihop_extra.jsonl $W/questions_multihop_extra2.jsonl \
    $W/questions_multihop_extra3.jsonl --frames-meta $D/frames.jsonl --ocr $D/ocr_all.jsonl --person A --out $W/questions_test_parallel_T2T5.jsonl
# final split (T1 = outputs of the T1 question run; its parallel file = p4 on that run's p2 + p3 outputs)
python $P/p4_parallel_set.py --multihop $T1/questions_multihop.jsonl --extra $T1/questions_multihop_extra.jsonl --frames-meta $D/frames.jsonl \
    --ocr $D/ocr_all.jsonl --person A --out $T1/questions_parallel.jsonl
python $P/s9_final_split.py --train-all $W/questions_train_all.jsonl --diary $W/diary.jsonl --splits $W/splits.json \
    --single $W/questions_test_single_T2T5.jsonl --parallel $W/questions_test_parallel_T2T5.jsonl \
    --t1-single $T1/questions_single.jsonl --t1-parallel $T1/questions_parallel.jsonl --train-out data/questions_train.jsonl \
    --t1-test-out data/t1_test_uids.json --single-out data/questions_test_single.jsonl --parallel-out data/questions_test_parallel.jsonl
python $P/s8_eval_input.py --single data/questions_test_single.jsonl --out-eval-input data/eval_input_test.json --out-texts data/texts_test.jsonl
```

Verified for this release: with the paper's raw caches, steps 1-3 reproduce the training-period split and all 8,659 person-A
training/test v1 questions; steps 5-7 reproduce `scope.json`, `candidates.jsonl`, `judged.jsonl`, `single_all.jsonl` and the
T2..T5 single-hop file (3,575) byte for byte, with zero LLM calls; P1-P4 reproduce the p2 specs, every round's specs / candidates /
kept questions and the T2..T5 parallel file (497) byte for byte (all persons: the 1,004-question research file), also without LLM
calls; p4 on the T1 run recomputes the same `leak` scores as the 497 released ones from the same OCR. Step 9 gives exactly the
paper's training set (the 1,000 questions of the research training run), the 433 T1 test recordings and the 1,121 T1 test qids,
gold groups and query texts of the paper's evaluation; `make_eval_protocol.py` on the released files gives 4,265 single + 846
parallel questions, pool 63,538. The paper's `task_emb.npy` is the zero-shot embedding with the
default instruction (re-embedding a sample with the research code: cosine 0.9999), which is what `embed_text ocr` computes.
