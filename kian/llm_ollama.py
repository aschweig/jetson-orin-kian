"""LLM inference via Ollama running as a local server.

Ollama provides efficient GPU inference with a native streaming API.
Install with: curl -fsSL https://ollama.com/install.sh | sh

Usage:
    uv run kian --backend ollama
"""

import asyncio
import json
from collections.abc import AsyncIterator
from urllib.error import URLError
from urllib.request import Request, urlopen

from kian.llm import system_prompt, update_system_prompt

DEFAULT_MODEL = "qwen3:4b-q4_K_M"
HOST = "127.0.0.1"
PORT = 11434
NUM_CTX = 2048


def _server_ready() -> bool:
    try:
        urlopen(Request(f"http://{HOST}:{PORT}/api/tags"), timeout=2)
        return True
    except (URLError, OSError):
        return False


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self._model = model or DEFAULT_MODEL
        self._url = f"http://{HOST}:{PORT}/api/chat"
        self._history: list[dict] = [{"role": "system", "content": system_prompt()}]
        self._wiki_context = None

        if not _server_ready():
            raise ConnectionError(
                f"Ollama is not running on {HOST}:{PORT}. "
                "Start it with: systemctl start ollama"
            )

    def _trim_history(self):
        trim_high = NUM_CTX * 9 // 10
        trim_low = NUM_CTX * 7 // 10
        total = sum(len(m["content"]) // 3 for m in self._history)
        if total < trim_high:
            return
        while len(self._history) > 3:
            total = sum(len(m["content"]) // 3 for m in self._history)
            if total < trim_low:
                break
            del self._history[1:3]

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens via Ollama's native API."""
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()
        update_system_prompt(self._history, user_text=user_text, has_wiki=bool(self._wiki_context))
        if self._wiki_context:
            self._history[0]["content"] += f"\n\n{self._wiki_context}"

        body = json.dumps({
            "model": self._model,
            "messages": self._history,
            "stream": True,
            "think": False,
            "options": {"num_ctx": NUM_CTX},
        }).encode()

        req = Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
        )

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urlopen(req, timeout=60))

        full_response: list[str] = []
        try:
            while True:
                raw_line = await loop.run_in_executor(None, resp.readline)
                if not raw_line:
                    break
                line = raw_line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response.append(token)
                    yield token
                if chunk.get("done"):
                    break
        finally:
            resp.close()

        self._history.append({"role": "assistant", "content": "".join(full_response)})

    def set_wiki_context(self, context: str | None, title: str | None = None):
        """Accept wiki context for compatibility; injected into the next system prompt."""
        self._wiki_context = context

    def reset(self):
        self._history = [{"role": "system", "content": system_prompt()}]
        self._wiki_context = None
