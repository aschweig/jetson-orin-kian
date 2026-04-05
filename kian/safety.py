"""LLM-based content safety classifier using a small model via llama-cpp-python.

Runs entirely on CPU to avoid competing with the primary LLM for GPU memory.
"""

from pathlib import Path

from llama_cpp import Llama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct-q2_k.gguf"

SYSTEM_PROMPT = (
    "You are a content safety filter for a childrens educational assistant. "
    "Flag ANY of the following: profanity, insults, violence, weapons, "
    "sexual content, self harm, drugs, or adult topics. "
    "Reply SAFE or UNSAFE only."
)

_llm: Llama | None = None


def _get_llm() -> Llama | None:
    global _llm
    if _llm is not None:
        return _llm
    if not MODEL_PATH.exists():
        return None
    _llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )
    return _llm


def classify(text: str) -> bool:
    """Return True if text is safe, False if unsafe."""
    llm = _get_llm()
    if llm is None:
        return True  # fail open if model unavailable

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=4,
    )
    result = resp["choices"][0]["message"]["content"].strip().upper()
    return "UNSAFE" not in result


def available() -> bool:
    """Check if the safety model is available."""
    return MODEL_PATH.exists()
