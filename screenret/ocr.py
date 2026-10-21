"""OCR baselines: extract text from the screenshot, then embed the text with the SAME model (text-only input, default instruction).

ocr-rapid   RapidOCR (rapidocr_onnxruntime, ONNX Runtime CPU execution provider)       pip install rapidocr_onnxruntime
ocr-vision  Apple Vision VNRecognizeTextRequest, accurate level, language correction on  pip install pyobjc-framework-Vision (macOS)

Both order lines the same way (top-to-bottom in 12-px bands, then left-to-right) so the text embedder sees comparable input.
Text is cut to MAX_OCR_CHARS characters; an empty result is embedded as "(no text)".
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from PIL import Image

from .config import MAX_OCR_CHARS
from .model import Model

_rapid = None


def rapidocr_text(path: str) -> str:
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid = RapidOCR()
    res, _ = _rapid(path)
    if not res:
        return ""
    lines = sorted(res, key=lambda r: (round(min(p[1] for p in r[0]) / 12), min(p[0] for p in r[0])))   # (band of top y, left x)
    return "\n".join(r[1] for r in lines if r[1] and r[1].strip())


def vision_text(path: str, level: int = 0) -> str:
    """Apple Vision OCR. level 0 = accurate, 1 = fast."""
    import objc
    import Vision
    from Foundation import NSURL
    W, H = Image.open(path).size
    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(level)
        req.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), {})
        handler.performRequests_error_([req], None)
        out = []
        for o in (req.results() or []):
            cands = o.topCandidates_(1)
            if not cands:
                continue
            bb = o.boundingBox()                       # normalised, origin bottom-left -> pixel top / left
            out.append(((1 - bb.origin.y - bb.size.height) * H, bb.origin.x * W, str(cands[0].string())))
    out.sort(key=lambda r: (round(r[0] / 12), r[1]))
    return "\n".join(t for _, _, t in out if t.strip())


OCR_ENGINES = {"ocr-rapid": rapidocr_text, "ocr-vision": vision_text}


def ocr_input(text: str) -> str:
    return (text or "(no text)")[:MAX_OCR_CHARS]


def ocr_embed(m: Model, path: str, route: str, timings: dict) -> tuple[torch.Tensor, int]:
    """Timing route: OCR + text embedding (batch 1, stock HF forward). Sets timings['t_ocr_done']. Returns (embedding, n OCR chars)."""
    txt = OCR_ENGINES[route](path)
    timings["t_ocr_done"] = time.perf_counter()
    e = m.embedder.process([{"text": ocr_input(txt)}])
    m.sync()
    return e[0].float(), len(txt)


def embed_texts(m: Model, texts: list[str], instruction: str | None = None, batch_size: int = 16) -> torch.Tensor:
    """Accuracy route: batched text embeddings (right padding, last-token pooling), normalised in fp32. Returns CPU float32 (N, 2048)."""
    out = []
    for b in range(0, len(texts), batch_size):
        e = m.embedder.embed([m.embedder.format_model_input(text=t, instruction=instruction) for t in texts[b:b + batch_size]],
                             normalize=False)
        out.append(F.normalize(e.float(), p=2, dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 2048)
