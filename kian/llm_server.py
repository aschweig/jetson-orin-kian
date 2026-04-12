"""LLM inference via llama.cpp server (OpenAI-compatible API).

Auto-starts a llama-server child process if one isn't already running.
The child is terminated when Kian exits.

Usage:
    uv run kian --backend server
    uv run kian --backend server --model Bonsai-8B.gguf
"""

import asyncio
import atexit
import json
import signal
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from kian.llm import system_prompt, update_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_SERVER = PROJECT_ROOT / "vendor" / "prismml-llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_MODEL = "Bonsai-8B.gguf"
HOST = "127.0.0.1"
PORT = 8080
NUM_CTX = 2048
SERVER_STARTUP_TIMEOUT = 60


def _server_ready() -> bool:
    try:
        urlopen(Request(f"http://{HOST}:{PORT}/health"), timeout=2)
        return True
    except (URLError, OSError):
        return False


def _start_server(model_path: str) -> subprocess.Popen:
    """Launch llama-server as a child process."""
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER}. "
            "Build it with: ./scripts/build-prismml-llama.sh"
        )
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-c", str(NUM_CTX),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--threads", "6",
        "--host", HOST,
        "--port", str(PORT),
    ]
    print(f"[SERVER] starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def _kill_server():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_kill_server)

    # Wait for the server to become ready
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"llama-server exited during startup (code {proc.returncode}):\n{stderr}")
        if _server_ready():
            print(f"[SERVER] ready on {HOST}:{PORT}")
            return proc
        time.sleep(1)

    proc.kill()
    raise TimeoutError(f"llama-server did not become ready within {SERVER_STARTUP_TIMEOUT}s")


def _resolve_model_path(model: str | None) -> str:
    """Resolve model name to a full path under models/."""
    name = model or DEFAULT_MODEL
    if not name.endswith(".gguf"):
        name += ".gguf"
    path = PROJECT_ROOT / "models" / name
    if path.exists():
        return str(path)
    # If it's already an absolute path, use it directly
    if Path(name).is_absolute() and Path(name).exists():
        return name
    raise FileNotFoundError(
        f"Model not found: {path}\n"
        "Download it with: ./scripts/download-bonsai.sh"
    )


class ServerLLM:
    def __init__(self, model: str | None = None):
        model_path = _resolve_model_path(model)
        self.model_name = Path(model_path).stem
        self._url = f"http://{HOST}:{PORT}/v1/chat/completions"
        self._history: list[dict] = [{"role": "system", "content": system_prompt()}]
        self._wiki_context: str | None = None
        self._wiki_titles: list[str | None] = [None]
        self._on_evict_title: callable | None = None
        self._server_proc: subprocess.Popen | None = None

        if not _server_ready():
            self._server_proc = _start_server(model_path)

    def set_on_evict_title(self, callback: callable):
        self._on_evict_title = callback

    def set_wiki_context(self, context: str | None, title: str | None = None):
        self._wiki_context = context
        self._pending_wiki_title = title

    def _trim_history(self):
        max_tokens = 1500
        while len(self._history) > 3:
            total = sum(len(m["content"]) // 3 for m in self._history)
            if total < max_tokens:
                break
            for wiki_title in self._wiki_titles[1:3]:
                if wiki_title and self._on_evict_title:
                    self._on_evict_title(wiki_title)
            del self._history[1:3]
            del self._wiki_titles[1:3]

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens via OpenAI-compatible API."""
        self._history.append({"role": "user", "content": user_text})
        self._wiki_titles.append(getattr(self, "_pending_wiki_title", None))
        self._pending_wiki_title = None
        self._trim_history()
        update_system_prompt(self._history, user_text=user_text, has_wiki=bool(self._wiki_context))

        if self._wiki_context:
            base = self._history[0]["content"]
            self._history[0]["content"] = (
                base + "\n\nYou have the following reference material. "
                "Use it only if relevant to what the child is asking — "
                "do not mention that you have it or where it came from.\n\n"
                + self._wiki_context
            )

        body = json.dumps({
            "model": self.model_name,
            "messages": self._history,
            "stream": True,
            "temperature": 0.5,
            "top_p": 0.85,
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
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_response.append(token)
                    yield token
        finally:
            resp.close()

        # Restore system prompt (remove wiki context)
        if self._wiki_context:
            self._history[0]["content"] = system_prompt()
            self._wiki_context = None

        self._history.append({"role": "assistant", "content": "".join(full_response)})
        self._wiki_titles.append(None)

    def reset(self):
        self._history = [{"role": "system", "content": system_prompt()}]
        self._wiki_titles = [None]
        self._wiki_context = None
