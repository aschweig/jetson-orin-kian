"""LLM inference via llama-cpp-python."""

import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

from llama_cpp import Llama

from kian.llm import system_prompt, update_system_prompt

_SENTINEL = object()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen3-4b-instruct-2507-q4_k_m.gguf"


class LlamaLLM:
    def __init__(self, model_path: str | None = None, n_gpu_layers: int = -1):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / DEFAULT_MODEL)
        self.model_name = Path(model_path).name
        self._llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
            n_batch=256,
            flash_attn=True,
            use_mmap=False,
            verbose=False,
        )
        self._history: list[dict] = [{"role": "system", "content": system_prompt()}]
        self._wiki_context: str | None = None
        self._wiki_titles: list[str | None] = [None]  # parallel to _history
        self._on_evict_title: callable | None = None

    def set_on_evict_title(self, callback: callable):
        """Set a callback invoked with a wiki title when a turn is dropped."""
        self._on_evict_title = callback

    def set_wiki_context(self, context: str | None, title: str | None = None):
        """Set reference material for the next turn, with its title for eviction tracking."""
        self._wiki_context = context
        self._pending_wiki_title = title

    def _trim_history(self):
        """Drop oldest message pairs to stay within context limit."""
        max_tokens = 1500
        while len(self._history) > 3:
            total = sum(len(m["content"]) // 3 for m in self._history)
            if total < max_tokens:
                break
            # Evict wiki titles from the dropped pair
            for wiki_title in self._wiki_titles[1:3]:
                if wiki_title and self._on_evict_title:
                    self._on_evict_title(wiki_title)
            del self._history[1:3]
            del self._wiki_titles[1:3]

    def _generate_sync(self, token_queue: queue.Queue):
        """Run LLM inference in a thread, pushing tokens to a queue."""
        try:
            response = self._llm.create_chat_completion(
                messages=self._history,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content", "")
                if token:
                    token_queue.put(token)
        finally:
            token_queue.put(_SENTINEL)

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens."""
        self._history.append({"role": "user", "content": user_text})
        self._wiki_titles.append(getattr(self, "_pending_wiki_title", None))
        self._pending_wiki_title = None
        self._trim_history()
        update_system_prompt(self._history)

        # Inject wiki context into system prompt for this turn
        if self._wiki_context:
            base = self._history[0]["content"]
            self._history[0]["content"] = (
                base + "\n\nYou have the following reference material. "
                "Use it only if relevant to what the child is asking — "
                "do not mention that you have it or where it came from.\n\n"
                + self._wiki_context
            )

        token_queue: queue.Queue = queue.Queue()
        thread = threading.Thread(target=self._generate_sync, args=(token_queue,), daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        full_response = []
        while True:
            token = await loop.run_in_executor(None, token_queue.get)
            if token is _SENTINEL:
                break
            full_response.append(token)
            yield token

        # Restore system prompt (remove wiki context)
        if self._wiki_context:
            self._history[0]["content"] = system_prompt()
            self._wiki_context = None

        self._history.append({"role": "assistant", "content": "".join(full_response)})
        self._wiki_titles.append(None)  # assistant turns don't have wiki titles

    def reset(self):
        self._history = [{"role": "system", "content": system_prompt()}]
        self._wiki_titles = [None]
