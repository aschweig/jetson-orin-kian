"""LLM inference via llama-cpp-python."""

import asyncio
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

from llama_cpp import Llama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf")


class LLM:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_gpu_layers: int = -1):
        self._llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
            n_batch=256,
            flash_attn=True,
            verbose=False,
        )
        self._history: list[dict] = [
            {"role": "system", "content": "You are Kian, a helpful voice assistant. You are talking to an imaginative and curious 7 year old. Keep responses concise and conversational. Your output will be spoken aloud, so never use markdown, asterisks, bullet points, emojis, or any formatting. Use plain spoken English only."}
        ]

    def _trim_history(self):
        """Drop oldest message pairs to stay within context limit."""
        max_tokens = 1500  # leave room for response
        while len(self._history) > 3:  # keep system + at least 1 exchange
            total = sum(len(m["content"]) // 3 for m in self._history)  # rough token estimate
            if total < max_tokens:
                break
            # Remove oldest user+assistant pair (keep system prompt at index 0)
            del self._history[1:3]

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens."""
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                self._llm.create_chat_completion,
                messages=self._history,
                stream=True,
            ),
        )

        full_response = []
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                full_response.append(token)
                yield token

        self._history.append({"role": "assistant", "content": "".join(full_response)})

    def reset(self):
        self._history = self._history[:1]
