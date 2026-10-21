"""Accuracy-side text embeddings with the same model (stock HF forward, batched): OCR-text frame vectors and query vectors.

  ocr      --ocr-jsonl lines {"id": frame id, "text": OCR text} -> id-indexed (n_rows, 2048) fp16 matrix. Default instruction;
           text cut to 12,000 characters, empty -> "(no text)" (same input as the timed OCR route).
  queries  --texts JSON list of strings, or JSONL with a "text" field -> (n, 2048) fp32 rows in file order, with the retrieval
           query instruction ("Retrieve the computer screen that matches the user's description.").

example: python -m screenret.embed_text ocr --lora none --ocr-jsonl ocr_vision.jsonl --n-rows 80459 --out ocr_frames.npy
         python -m screenret.embed_text queries --lora v1.pt --texts questions.jsonl --out q_lora.npy
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .cli import add_model_args, load_from_args
from .config import QUERY_INSTRUCTION
from .ocr import embed_texts, ocr_input


def read_texts(path: str) -> list[str]:
    if path.endswith(".json"):
        return [t if isinstance(t, str) else t["text"] for t in json.load(open(path))]
    return [json.loads(line)["text"] for line in open(path) if line.strip()]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m screenret.embed_text", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["ocr", "queries"])
    add_model_args(p)
    p.add_argument("--ocr-jsonl", help="ocr: {'id', 'text'} per line")
    p.add_argument("--n-rows", type=int, help="ocr: rows of the id-indexed output (> max id)")
    p.add_argument("--texts", help="queries: JSON list or JSONL with 'text'")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True, help="output .npy")
    a = p.parse_args(argv)
    m = load_from_args(a)
    if a.what == "ocr":
        recs = [json.loads(line) for line in open(a.ocr_jsonl) if line.strip()]
        E = embed_texts(m, [ocr_input(r["text"]) for r in recs], batch_size=a.batch_size).numpy()
        X = np.zeros((a.n_rows, E.shape[1]), np.float16)
        X[[int(r["id"]) for r in recs]] = E
    else:
        X = embed_texts(m, read_texts(a.texts), instruction=QUERY_INSTRUCTION, batch_size=a.batch_size).numpy()
    np.save(a.out, X)
    print(f"{a.what}: {len(X)} rows -> {a.out}")


if __name__ == "__main__":
    main()
