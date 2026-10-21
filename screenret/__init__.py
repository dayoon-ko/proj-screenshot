"""screenret: efficient continuous-stream screenshot embedding with Qwen3-VL-Embedding-2B (Token Pruning with Grid Keep + KV Reuse)
and its evaluation.

Entry points: python -m screenret.timing | embed_stream | merge_segments | embed_text | evaluate   (each has --help)
"""
from .config import PRESETS, MethodConfig            # noqa: F401
from .engine import Engine, preprocess               # noqa: F401
from .model import load_model                         # noqa: F401
