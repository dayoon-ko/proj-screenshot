# Question files

Questions about person A's screen history in PSAI (the recordings and frames themselves are not included). Built by
[`../data_pipeline`](../data_pipeline); recording ids (`uid`) are PSAI `unique_data_id`s.

Split: the retrieval LoRA trains on 1,000 T1-train questions and is selected on T1-val. The 433 T1-train recordings that no training
question points to join the test set, so the test set covers T2..T5 and those T1 recordings. The pool is unchanged.

| file | rows | content |
|---|---|---|
| `questions_train.jsonl` | 1,000 | training questions of the retrieval LoRA (T1-train): all 960 web questions + 40 desktop questions sampled to the test periods' desktop frame share (4 %); `qid`, `split`, `asks` (where / when / how / which), `question`, `taskId`, `gold` (recording ids) |
| `questions_val.jsonl` | 315 | model-selection questions (T1-val), same fields |
| `questions_test_single.jsonl` | 4,265 | single test questions (one recording answers each): T2..T5 3,575 + T1 690 (qids `T1:S-...`, on 344 of the 433 T1 test recordings); `qid`, `type` (where / when / input / what / how), `slot`, `period`, `question`, `answer`, `evidence_groups` [{`uids`, ...}], `cue_parts`, `cue_type`, `n_gold_recordings`, ... |
| `questions_test_parallel.jsonl` | 846 | multi test questions (several recordings each; "parallel" in the code): T2..T5 415 + T1 431 (qids `T1:M-...`); `qid`, `type` (order / aggregate / duration / compare; no freq2), `question`, `answer`, `evidence_groups` (one per evidence recording set), `leak` |
| `eval_input_test.json`, `texts_test.jsonl` | 4,265 | single questions in the input format of `scripts/make_eval_protocol.py` |
| `splits.json` | | periods T1..T5 (`periods`, `fids`), T1-train / T1-val recordings, twins, `bad_fids` (answer-revealing frames, excluded from the pool) |
| `t1_test_uids.json` | 433 | T1-train recordings that are no training question's first gold recording (all desktop) = the T1 part of the test set |

Test set by type: single when 1,431, where 1,348, input 1,285, what 132, how 69 (web 3,516 / desktop 749); multi order 426,
aggregate 243, duration 153, compare 24.

Evaluation protocol of the paper (4,265 single + 846 multi questions, pool = T1..T5 frames minus `bad_fids` = 63,538 frames):

```bash
python scripts/make_eval_protocol.py --eval-input data/eval_input_test.json --texts data/texts_test.jsonl \
    --parallel data/questions_test_parallel.jsonl --splits data/splits.json --frames-meta frames.jsonl \
    --out protocol.json --queries-out protocol_queries.jsonl
```
