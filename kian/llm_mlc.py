"""LLM inference via MLC LLM running in a Docker container.

Launches dustynv/mlc as a local OpenAI-compatible server on Jetson Orin.
The container is started automatically and stopped on exit.

First run pulls the image (~7GB) and the model (~2.3GB).

Usage:
    uv run kian --backend mlc
"""

import asyncio
import atexit
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from kian.llm import system_prompt, update_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONTAINER_IMAGE = "dustynv/mlc:0.20.0-r36.4.0"
CONTAINER_NAME = "kian-mlc"
DEFAULT_MODEL = "HF://mlc-ai/Qwen3-4B-q4f16_1-MLC"
HOST = "127.0.0.1"
PORT = 8400


def _container_running() -> bool:
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == "true"


def _stop_container():
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True, timeout=15)


def _start_server(model: str):
    """Start the MLC serve container and wait for it to be ready."""
    if _container_running():
        return

    # Clean up any stopped container with the same name
    _stop_container()

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    docker = shutil.which("docker") or "docker"
    cmd = [
        "sudo", docker, "run", "-d",
        "--runtime", "nvidia",
        "--name", CONTAINER_NAME,
        "--network", "host",
        "-v", f"{models_dir}:/data/models",
        "-e", f"HF_HOME=/data/models/hf-cache",
        CONTAINER_IMAGE,
        "python3", "-m", "mlc_llm", "serve", model,
        "--mode", "interactive",
        "--host", HOST,
        "--port", str(PORT),
    ]
    print(f"[MLC] Starting container ({CONTAINER_IMAGE})...")
    subprocess.run(cmd, check=True, timeout=60)
    atexit.register(_stop_container)

    # Wait for server to become ready
    url = f"http://{HOST}:{PORT}/v1/models"
    deadline = time.monotonic() + 120  # model load can take a while
    while time.monotonic() < deadline:
        try:
            urlopen(Request(url), timeout=2)
            print(f"[MLC] Server ready on port {PORT}")
            return
        except (URLError, OSError):
            time.sleep(2)
    raise TimeoutError("MLC server did not start within 120s")


class MlcLLM:
    def __init__(self, model: str | None = None):
        import json
        from urllib.request import Request, urlopen  # noqa: F811

        self._model = model or DEFAULT_MODEL
        self._url = f"http://{HOST}:{PORT}/v1/chat/completions"
        self._history: list[dict] = [{"role": "system", "content": system_prompt()}]

        _start_server(self._model)

    def _trim_history(self):
        max_tokens = 1500
        while len(self._history) > 3:
            total = sum(len(m["content"]) // 3 for m in self._history)
            if total < max_tokens:
                break
            del self._history[1:3]

    async def chat_stream(self, user_text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens via the container's OpenAI-compatible API."""
        import json

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()
        update_system_prompt(self._history)

        body = json.dumps({
            "model": self._model,
            "messages": self._history,
            "stream": True,
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
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_response.append(token)
                    yield token
        finally:
            resp.close()

        self._history.append({"role": "assistant", "content": "".join(full_response)})

    def reset(self):
        self._history = [{"role": "system", "content": system_prompt()}]
