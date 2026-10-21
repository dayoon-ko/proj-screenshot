"""Step 4: task texts for the competitor search of the uniqueness judge (s6) -> task_texts.jsonl, embedded with screenret.

One text per diary event (its cleaned instruction), row = diary order. The vectors are zero-shot Qwen3-VL-Embedding-2B embeddings with
the model's default instruction (the one used for screenshots / OCR text), so `screenret.embed_text ocr` embeds them; the competitor
search only uses their cosine similarities.

example: python data_pipeline/s4_task_embeddings.py --diary diary.jsonl --out task_texts.jsonl
         python -m screenret.embed_text ocr --lora none --device cuda --ocr-jsonl task_texts.jsonl --n-rows <#events> --out task_emb.npy
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diary", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = 0
    with open(a.out, "w") as f:
        for i, l in enumerate(open(a.diary)):
            e = json.loads(l)
            f.write(json.dumps({"id": i, "event_id": e["event_id"], "text": e["instruction"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} task texts -> {a.out} (embed with --n-rows {n})")


if __name__ == "__main__":
    main()
