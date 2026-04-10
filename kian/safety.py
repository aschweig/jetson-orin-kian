"""Content safety classifier using Granite Guardian HAP 38M (ONNX).

Uses a purpose-built 38M-parameter hate/abuse/profanity detector instead of
a general-purpose LLM.  Runs entirely on CPU via ONNX Runtime — no GPU memory
needed, no CUDA context created.

Model: ibm-granite/granite-guardian-hap-38m
ONNX:  KantiArumilli/granite-guardian-hap-38m-onnx (INT8 quantized)
"""

from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = PROJECT_ROOT / "models" / "granite-guardian-hap"
_ONNX_PATH = _MODEL_DIR / "guardian_model_quantized.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"

_session: onnxruntime.InferenceSession | None = None
_tokenizer: Tokenizer | None = None

# Confidence threshold: flag as unsafe when P(toxic) >= this value.
_THRESHOLD = 0.5


def _get_model() -> tuple[onnxruntime.InferenceSession, Tokenizer] | None:
    global _session, _tokenizer
    if _session is not None:
        return _session, _tokenizer
    if not _ONNX_PATH.exists() or not _TOKENIZER_PATH.exists():
        return None

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    _session = onnxruntime.InferenceSession(
        str(_ONNX_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
    _tokenizer.enable_truncation(max_length=128)
    _tokenizer.enable_padding(length=128)
    return _session, _tokenizer


def load() -> None:
    """Eagerly load the safety model at startup."""
    _get_model()


def classify(text: str) -> bool:
    """Return True if text is safe, False if unsafe."""
    pair = _get_model()
    if pair is None:
        return True  # fail open if model unavailable

    session, tokenizer = pair
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    logits = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]  # shape: [1, 2]

    # softmax → P(toxic)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    toxic_prob = float(probs[0, 1])

    return toxic_prob < _THRESHOLD


def available() -> bool:
    """Check if the safety model is available."""
    return _ONNX_PATH.exists() and _TOKENIZER_PATH.exists()
