"""Bulk Apple Vision OCR (accurate, language correction on) of a frame list -> JSONL {"id", "text"} (input of `embed_text ocr`).

Same request and line ordering as the timed ocr-vision route (screenret/ocr.py). Resumable: ids already in --out are skipped. macOS only.
example: python scripts/vision_ocr_extract.py --frames-list order.tsv --frames-root FRAMES --out ocr_vision.jsonl
"""
import argparse
import json
import os
import time

from screenret.embed_stream import read_frames_list
from screenret.ocr import vision_text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-list", required=True, help="'<id>\\t<path>' per line")
    p.add_argument("--frames-root", default=os.environ.get("SCREENRET_FRAMES", ""))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    ids, paths = read_frames_list(a.frames_list, a.frames_root)
    done = {json.loads(line)["id"] for line in open(a.out)} if os.path.exists(a.out) else set()
    todo = [(int(i), pth) for i, pth in zip(ids, paths) if int(i) not in done][: a.limit or None]
    print(f"{len(ids)} frames, {len(done)} done, {len(todo)} to do -> {a.out}", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fo:
        for k, (i, pth) in enumerate(todo):
            fo.write(json.dumps({"id": i, "text": vision_text(pth)}, ensure_ascii=False) + "\n")
            fo.flush()
            if (k + 1) % 500 == 0 or k + 1 == len(todo):
                rate = (time.time() - t0) / (k + 1)
                print(f"{k + 1}/{len(todo)} | {rate:.3f} s/frame | ETA {(len(todo) - k - 1) * rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
