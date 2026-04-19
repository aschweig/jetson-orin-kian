#!/usr/bin/env python3
"""Benchmark TTFT and tok/s across LLM backends.

Sends a fixed sequence of prompts to each engine, measuring time-to-first-token
and generation speed.  Prompts build on each other to simulate a growing
conversation (context window).

Usage:
    uv run python scripts/benchmark-llm.py
    uv run python scripts/benchmark-llm.py --engines llamacpp,ollama:qwen3:4b-q4_K_M
    uv run python scripts/benchmark-llm.py --csv results.csv
    uv run python scripts/benchmark-llm.py --prompts scripts/my-prompts.txt
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = PROJECT_ROOT / "scripts" / "benchmark-prompts.txt"
LOG_DIR = PROJECT_ROOT / "benchmark-logs"

# Dummy prompts that simulate cache-warming inputs; cap at 1 token output.
SINGLE_TOKEN_PROMPTS = {"Ummm", "Okay"}

SYSTEM_PROMPT = (
    "You are Kian, a helpful educating cartoon animal -- but it's a secret. "
    "You are talking to an imaginative and curious child in grade 3. "
    "Reply in three to five sentences. Be detailed and enthusiastic. "
    "Your output will be spoken aloud, so never use markdown, asterisks, bullet points, emojis, "
    "or any formatting. Use plain spoken English only."
)


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a text file (one per line, blank lines ignored)."""
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


def read_meminfo() -> tuple[int, int]:
    """Read swap usage and page cache from /proc/meminfo. Returns (swap_used_mb, cached_mb)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
    swap_used = (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) // 1024
    cached = info.get("Cached", 0) // 1024
    return swap_used, cached


def append_log(log_path: Path, engine_name: str, prompts: list[str],
               responses: list[str], results: list[dict]):
    """Append a conversation log for one engine to the shared run log."""
    lines = [f"{'='*60}", f"Engine: {engine_name}", f"{'='*60}", ""]
    for i, (prompt, response, row) in enumerate(zip(prompts, responses, results)):
        lines.append(f"--- Prompt {i+1} ---")
        lines.append(f"User: {prompt}")
        lines.append(f"Assistant: {response}")
        lines.append(f"  TTFT={row['ttft']:.3f}s  tokens={row['tokens']}  "
                     f"total={row['total_time']:.1f}s  tok/s={row['tok_per_sec']:.1f}")
        lines.append("")

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"


def ollama_unload(model: str):
    """Unload a model from Ollama to free GPU memory."""
    try:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = Request(
            f"{OLLAMA_URL}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urlopen(req, timeout=10)
    except (URLError, OSError):
        pass
    time.sleep(1)


def drop_caches():
    """Drop kernel page cache to free mmap'd GGUF pages (requires sudo)."""
    import subprocess as sp
    try:
        sp.run(["sudo", "tee", "/proc/sys/vm/drop_caches"],
               input=b"3", stdout=sp.DEVNULL, timeout=5)
    except (sp.TimeoutExpired, OSError):
        pass


def ollama_unload_all():
    """Unload all Ollama models and wait until GPU memory is freed."""
    try:
        req = Request(f"{OLLAMA_URL}/api/ps", method="GET")
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        for name in models:
            print(f"  Unloading Ollama model: {name}")
            ollama_unload(name)
        # Wait until all models are actually unloaded
        if models:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                req = Request(f"{OLLAMA_URL}/api/ps", method="GET")
                resp = urlopen(req, timeout=5)
                data = json.loads(resp.read().decode())
                if not data.get("models"):
                    print(f"  All models unloaded.")
                    break
                time.sleep(1)
    except (URLError, OSError):
        pass
    drop_caches()
    time.sleep(2)


def ollama_check_offload(model: str) -> tuple[int | None, float | None, float | None]:
    """Check and report GPU offload status after warmup.

    Returns (gpu_pct, vram_gb, total_gb).
    """
    try:
        req = Request(f"{OLLAMA_URL}/api/ps", method="GET")
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        for m in data.get("models", []):
            if m.get("name", "") == model:
                total_gb = m.get("size", 0) / 1e9
                vram_gb = m.get("size_vram", 0) / 1e9
                pct = round((m.get("size_vram", 0) / m.get("size", 1)) * 100)
                print(f"  Offload: {pct}% GPU  "
                      f"(VRAM: {vram_gb:.1f}GB / Total: {total_gb:.1f}GB)")
                if pct < 100:
                    print(f"  WARNING: Model not fully offloaded to GPU!")
                return pct, round(vram_gb, 2), round(total_gb, 2)
    except (URLError, OSError):
        pass
    return None, None, None


def ollama_warmup(model: str):
    """Send a throwaway request so the model is loaded in GPU memory."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": 2048},
    }).encode()
    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(req, timeout=120)
    except (URLError, OSError) as e:
        print(f"  Warmup failed: {e}", file=sys.stderr)
        sys.exit(1)


def bench_ollama(model: str, prompts: list[str], log_path: Path) -> list[dict]:
    """Benchmark an Ollama model, returning per-prompt metrics."""
    engine_name = f"ollama:{model}"
    print(f"\n{'='*60}")
    print(f"Engine: {engine_name}")
    print(f"{'='*60}")

    # Unload any existing model, then warm up this one
    ollama_unload_all()
    print(f"  Warming up {model} ...")
    ollama_warmup(model)
    print(f"  Model loaded.")
    gpu_pct, vram_gb, total_gb = ollama_check_offload(model)
    time.sleep(2)

    trim_high = 2048 * 9 // 10
    trim_low = 2048 * 7 // 10

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    responses = []

    for i, prompt in enumerate(prompts):
        history.append({"role": "user", "content": prompt})

        # Trim: when we exceed 90% of context, drop turns until we're at 70%
        total_est = sum(len(m["content"]) // 3 for m in history)
        trimmed = 0
        if total_est >= trim_high:
            while len(history) > 3:
                total_est = sum(len(m["content"]) // 3 for m in history)
                if total_est < trim_low:
                    break
                del history[1:3]
                trimmed += 1
            total_est = sum(len(m["content"]) // 3 for m in history)
        if trimmed:
            print(f"  [TRIM] dropped {trimmed} turn(s), "
                  f"~{total_est} tokens remain after trim (trigger at {trim_high}, target {trim_low}), "
                  f"{len(history)} messages", file=sys.stderr)

        options = {"num_ctx": 2048}
        if prompt in SINGLE_TOKEN_PROMPTS:
            options["num_predict"] = 1

        body = json.dumps({
            "model": model,
            "messages": history,
            "stream": True,
            "think": False,
            "options": options,
        }).encode()

        req = Request(
            f"{OLLAMA_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.monotonic()
        resp = urlopen(req, timeout=120)

        ttft = None
        token_count = 0
        full_response = []

        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                if ttft is None:
                    ttft = time.monotonic() - t0
                token_count += 1
                full_response.append(token)
            if chunk.get("done"):
                break
        resp.close()

        total = time.monotonic() - t0
        gen_time = total - (ttft or total)
        tps = token_count / gen_time if gen_time > 0 else 0

        row = {
            "engine": engine_name,
            "prompt_no": i + 1,
            "ttft": round(ttft or 0, 3),
            "total_time": round(total, 3),
            "tokens": token_count,
            "tok_per_sec": round(tps, 1),
            "gpu_pct": gpu_pct or 0,
            "vram_gb": vram_gb or 0,
            "total_ram_gb": total_gb or 0,
            "trimmed": trimmed,
        }
        row["swap_mb"], row["cache_mb"] = read_meminfo()
        results.append(row)
        response_text = "".join(full_response)
        responses.append(response_text)
        print(f"  Prompt {i+1}: TTFT={row['ttft']:.3f}s  "
              f"tokens={token_count}  "
              f"total={row['total_time']:.1f}s  "
              f"tok/s={row['tok_per_sec']:.1f}")
        snippet = response_text.replace("\n", " ")[:120]
        if len(response_text) > 120:
            snippet += "..."
        print(f"    > {snippet}")

        history.append({"role": "assistant", "content": response_text})

    append_log(log_path, engine_name, prompts, responses, results)
    return results


# ---------------------------------------------------------------------------
# llama-server engine (PrismML fork — for Bonsai-8B etc.)
# ---------------------------------------------------------------------------

LLAMA_SERVER = PROJECT_ROOT / "vendor" / "prismml-llama.cpp" / "build" / "bin" / "llama-server"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 120


def _server_ready() -> bool:
    try:
        urlopen(Request(f"{SERVER_URL}/health"), timeout=2)
        return True
    except (URLError, OSError):
        return False


def _start_llama_server(model_path: str, n_ctx: int = 2048) -> "subprocess.Popen":
    import subprocess as sp
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-c", str(n_ctx),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "q4_0",
        "-ctv", "q4_0",
        "--threads", "6",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
    ]
    print(f"  Starting llama-server: {' '.join(cmd)}")
    proc = sp.Popen(cmd, stdout=sp.DEVNULL, stderr=sp.PIPE)

    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"llama-server exited during startup (code {proc.returncode}):\n{stderr}")
        if _server_ready():
            print(f"  llama-server ready on {SERVER_HOST}:{SERVER_PORT}")
            return proc
        time.sleep(1)

    proc.kill()
    raise TimeoutError(f"llama-server did not become ready within {SERVER_STARTUP_TIMEOUT}s")


def _stop_llama_server(proc: "subprocess.Popen"):
    import signal
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    time.sleep(3)


def bench_server(model_file: str, prompts: list[str], log_path: Path) -> list[dict]:
    """Benchmark a model via the PrismML llama-server (OpenAI-compatible API).

    A `-c<NUM>` suffix on the model_file (before .gguf) overrides the default
    2048-token context (e.g. `granite-4.0-h-micro-Q4_K_M-c1024.gguf`).
    """
    model_name = Path(model_file).stem
    engine_name = f"server:{model_name}"

    # Parse optional "-c<NUM>" suffix to set context size, then strip it
    # from the actual model file path on disk.
    n_ctx = 2048
    actual_file = model_file
    m = re.search(r"-c(\d+)(\.gguf)?$", model_name)
    if m:
        n_ctx = int(m.group(1))
        base = model_name[: m.start()]
        actual_file = base + (".gguf" if model_file.endswith(".gguf") else "")
    model_path = str(PROJECT_ROOT / "models" / actual_file)

    print(f"\n{'='*60}")
    print(f"Engine: {engine_name}  (n_ctx={n_ctx})")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER}. "
            "Build it with: ./scripts/build-prismml-llama.sh"
        )

    # Unload any Ollama models to free GPU
    ollama_unload_all()

    proc = _start_llama_server(model_path, n_ctx=n_ctx)
    try:
        model_size_gb = round(Path(model_path).stat().st_size / 1e9, 2)

        # Warmup
        print(f"  Warming up ...")
        warmup_body = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }).encode()
        req = Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=warmup_body,
            headers={"Content-Type": "application/json"},
        )
        urlopen(req, timeout=120)
        print(f"  Model loaded.")
        time.sleep(2)

        trim_high = n_ctx * 9 // 10
        trim_low = n_ctx * 7 // 10

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        results = []
        responses = []

        for i, prompt in enumerate(prompts):
            history.append({"role": "user", "content": prompt})

            # Trim: when we exceed 90% of context, drop turns until we're at 70%
            total_est = sum(len(m["content"]) // 3 for m in history)
            trimmed = 0
            if total_est >= trim_high:
                while len(history) > 3:
                    total_est = sum(len(m["content"]) // 3 for m in history)
                    if total_est < trim_low:
                        break
                    del history[1:3]
                    trimmed += 1
                total_est = sum(len(m["content"]) // 3 for m in history)
            if trimmed:
                print(f"  [TRIM] dropped {trimmed} turn(s), "
                      f"~{total_est} tokens remain after trim (trigger at {trim_high}, target {trim_low}), "
                      f"{len(history)} messages", file=sys.stderr)

            body_dict = {
                "model": model_name,
                "messages": history,
                "stream": True,
                "temperature": 0.5,
                "top_p": 0.85,
            }
            if prompt in SINGLE_TOKEN_PROMPTS:
                body_dict["max_tokens"] = 1
            body = json.dumps(body_dict).encode()

            req = Request(
                f"{SERVER_URL}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )

            t0 = time.monotonic()
            resp = urlopen(req, timeout=120)

            ttft = None
            token_count = 0
            full_response = []

            for raw_line in resp:
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
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    token_count += 1
                    full_response.append(token)
            resp.close()

            total = time.monotonic() - t0
            gen_time = total - (ttft or total)
            tps = token_count / gen_time if gen_time > 0 else 0

            row = {
                "engine": engine_name,
                "prompt_no": i + 1,
                "ttft": round(ttft or 0, 3),
                "total_time": round(total, 3),
                "tokens": token_count,
                "tok_per_sec": round(tps, 1),
                "gpu_pct": 100,
                "vram_gb": model_size_gb,
                "total_ram_gb": model_size_gb,
                "trimmed": trimmed,
            }
            row["swap_mb"], row["cache_mb"] = read_meminfo()
            results.append(row)
            response_text = "".join(full_response)
            responses.append(response_text)
            print(f"  Prompt {i+1}: TTFT={row['ttft']:.3f}s  "
                  f"tokens={token_count}  "
                  f"total={row['total_time']:.1f}s  "
                  f"tok/s={row['tok_per_sec']:.1f}")

            history.append({"role": "assistant", "content": response_text})

        append_log(log_path, engine_name, prompts, responses, results)
        return results
    finally:
        print(f"  Stopping llama-server ...")
        _stop_llama_server(proc)


# ---------------------------------------------------------------------------
# llama.cpp engine
# ---------------------------------------------------------------------------

LLAMACPP_MODELS = {
    "Qwen3.5-2B-Q4_K_M": {"file": "Qwen3.5-2B-Q4_K_M.gguf"},
    "granite-3.3-2b-instruct-Q4_K_M": {"file": "granite-3.3-2b-instruct-Q4_K_M.gguf"},
    "ibm-granite_granite-4.0-micro-IQ4_XS": {"file": "ibm-granite_granite-4.0-micro-IQ4_XS.gguf"},
    "granite-4.0-micro-Q4_K_M": {"file": "granite-4.0-micro-Q4_K_M.gguf"},
    "granite-4.0-h-micro-Q4_K_M": {"file": "granite-4.0-h-micro-Q4_K_M.gguf"},
    "granite-4.0-h-micro-Q4_K_M-c1024": {"file": "granite-4.0-h-micro-Q4_K_M.gguf", "n_ctx": 1024},
    "qwen3-4b-instruct-2507-q4_k_m": {"file": "qwen3-4b-instruct-2507-q4_k_m.gguf"},
}


def _llamacpp_subprocess(model_name: str, prompts: list[str], log_path: str):
    """Run in a subprocess — benchmark one llamacpp model, print JSON to stdout."""
    import gc
    from llama_cpp import Llama

    model_info = LLAMACPP_MODELS[model_name]
    gguf_file = model_info["file"]
    engine_name = f"llamacpp:{model_name}"

    model_path = str(PROJECT_ROOT / "models" / gguf_file)
    n_gpu_layers = model_info.get("n_gpu_layers", -1)
    gpu_pct = 100 if n_gpu_layers == -1 else None
    model_size_gb = round(Path(model_path).stat().st_size / 1e9, 2)
    vram_gb = model_size_gb if n_gpu_layers == -1 else 0
    total_ram_gb = model_size_gb

    print(f"  Loading model (n_gpu_layers={n_gpu_layers}) ...", file=sys.stderr)
    llm = Llama(
        model_path=model_path,
        n_ctx=model_info.get("n_ctx", 2048),
        n_gpu_layers=n_gpu_layers,
        n_batch=256,
        flash_attn=True,
        use_mmap=False,
        verbose=False,
    )
    print(f"  Model loaded.", file=sys.stderr)

    print(f"  Warming up ...", file=sys.stderr)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
    )
    time.sleep(2)

    n_ctx = model_info.get("n_ctx", 2048)
    trim_high = n_ctx * 9 // 10   # start trimming at 90%
    trim_low = n_ctx * 7 // 10    # trim down to 70%

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    responses = []

    for i, prompt in enumerate(prompts):
        history.append({"role": "user", "content": prompt})

        # Trim: when we exceed 90% of context, drop turns until we're at 70%
        total_est = sum(len(m["content"]) // 3 for m in history)
        trimmed = 0
        if total_est >= trim_high:
            while len(history) > 3:
                total_est = sum(len(m["content"]) // 3 for m in history)
                if total_est < trim_low:
                    break
                del history[1:3]
                trimmed += 1
            total_est = sum(len(m["content"]) // 3 for m in history)
        if trimmed:
            print(f"  [TRIM] dropped {trimmed} turn(s), "
                  f"~{total_est} tokens remain after trim (trigger at {trim_high}, target {trim_low}), "
                  f"{len(history)} messages", file=sys.stderr)

        t0 = time.monotonic()
        ttft = None
        token_count = 0
        full_response = []

        chat_kwargs = {"messages": history, "stream": True}
        if prompt in SINGLE_TOKEN_PROMPTS:
            chat_kwargs["max_tokens"] = 1
        response = llm.create_chat_completion(**chat_kwargs)
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                if ttft is None:
                    ttft = time.monotonic() - t0
                token_count += 1
                full_response.append(token)

        total = time.monotonic() - t0
        gen_time = total - (ttft or total)
        tps = token_count / gen_time if gen_time > 0 else 0

        row = {
            "engine": engine_name,
            "prompt_no": i + 1,
            "ttft": round(ttft or 0, 3),
            "total_time": round(total, 3),
            "tokens": token_count,
            "tok_per_sec": round(tps, 1),
            "gpu_pct": gpu_pct or 0,
            "vram_gb": vram_gb,
            "total_ram_gb": total_ram_gb,
            "trimmed": trimmed,
        }
        row["swap_mb"], row["cache_mb"] = read_meminfo()
        results.append(row)
        response_text = "".join(full_response)
        responses.append(response_text)
        print(f"  Prompt {i+1}: TTFT={row['ttft']:.3f}s  "
              f"tokens={token_count}  "
              f"total={row['total_time']:.1f}s  "
              f"tok/s={row['tok_per_sec']:.1f}", file=sys.stderr)
        snippet = response_text.replace("\n", " ")[:120]
        if len(response_text) > 120:
            snippet += "..."
        print(f"    > {snippet}", file=sys.stderr)

        history.append({"role": "assistant", "content": response_text})

    append_log(Path(log_path), engine_name, prompts, responses, results)

    # Output results as JSON on stdout for the parent to collect
    json.dump(results, sys.stdout)


def bench_llamacpp(model_name: str, prompts: list[str], log_path: Path) -> list[dict]:
    """Run llamacpp benchmark in a subprocess to ensure full CUDA cleanup."""
    import subprocess as sp

    engine_name = f"llamacpp:{model_name}"
    model_info = LLAMACPP_MODELS[model_name]
    gguf_file = model_info["file"]
    model_path = PROJECT_ROOT / "models" / gguf_file

    print(f"\n{'='*60}")
    print(f"Engine: {engine_name}")
    print(f"{'='*60}")

    # Free GPU memory from any loaded Ollama models
    ollama_unload_all()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Launch subprocess: run this same script with --_subprocess flag
    cmd = [
        sys.executable, "-u", __file__,  # -u: unbuffered stdio
        "--_subprocess", model_name,
        "--prompts", str(DEFAULT_PROMPTS),
        "--_log_path", str(log_path),
    ]
    # Capture stdout (JSON results) while streaming stderr live to the terminal.
    # Two threads, two pipes — never let two readers touch the same pipe (which
    # is what caused earlier ValueError: I/O operation on closed file).
    import threading
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, bufsize=1)

    stdout_chunks: list[str] = []
    stderr_done = threading.Event()
    stdout_done = threading.Event()

    def _forward_stderr():
        try:
            for line in proc.stderr:
                print(line.rstrip(), file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass
        finally:
            stderr_done.set()

    def _read_stdout():
        try:
            stdout_chunks.append(proc.stdout.read())
        except (ValueError, OSError):
            pass
        finally:
            stdout_done.set()

    stderr_thread = threading.Thread(target=_forward_stderr, daemon=True)
    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()

    try:
        proc.wait(timeout=1800)
    except sp.TimeoutExpired:
        proc.kill()
        stderr_done.wait(timeout=2)
        stdout_done.wait(timeout=2)
        raise

    stderr_done.wait(timeout=5)
    stdout_done.wait(timeout=5)

    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed (exit {proc.returncode})")

    # Replace result-like object so rest of function works unchanged
    class _R:
        pass
    result = _R()
    result.stdout = "".join(stdout_chunks)
    result.returncode = proc.returncode

    # Show memory state before GPU reclaim
    try:
        mem = sp.run(["free", "-h"], capture_output=True, text=True)
        for line in mem.stdout.strip().splitlines():
            print(f"  [MEM] {line}")
    except Exception:
        pass

    # Wait for Jetson unified memory to be reclaimed after subprocess exits.
    print("  Waiting for GPU memory reclaim ...")
    time.sleep(5)

    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_ENGINES = [
    # Ollama first — reliable unload via API between models
    "ollama:qwen3.5:2b-q4_K_M",
    "ollama:qwen3:4b-q4_K_M",
    "ollama:llama3.2:3b-instruct-q4_K_M",
    "ollama:ministral-3:3b",
    "ollama:granite3.3:2b",
    "ollama:granite4:3b",
    "ollama:nemotron-3-nano:4b",
    "ollama:gemma3:4b",
    # llama-server (PrismML fork) — manages its own process lifecycle.
    # Replaces the deprecated llamacpp engines (llama-cpp-python had
    # pathologically slow post-trim TTFT on Jetson; see kian/llm_llamacpp.py).
    # A "-c<NUM>" suffix sets a non-default context size.
    "server:Bonsai-8B.gguf",
    "server:Qwen3.5-2B-Q4_K_M.gguf",
    "server:granite-3.3-2b-instruct-Q4_K_M.gguf",
    "server:ibm-granite_granite-4.0-micro-IQ4_XS.gguf",
    "server:granite-4.0-micro-Q4_K_M.gguf",
    "server:granite-4.0-h-micro-Q4_K_M.gguf",
    "server:granite-4.0-h-micro-Q4_K_M-c1024.gguf",
    "server:qwen3-4b-instruct-2507-q4_k_m.gguf",
]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Benchmark LLM backends")
    parser.add_argument(
        "--engines",
        help=f"Comma-separated list of engines (default: all). Available: {','.join(ALL_ENGINES)}",
    )
    parser.add_argument(
        "--csv", metavar="FILE",
        help="Write results to CSV file",
    )
    parser.add_argument(
        "--prompts", metavar="FILE", default=str(DEFAULT_PROMPTS),
        help=f"Prompts file, one per line (default: {DEFAULT_PROMPTS.name})",
    )
    # Hidden flag: run a single llamacpp model in a subprocess
    parser.add_argument("--_subprocess", metavar="MODEL", help=argparse.SUPPRESS)
    parser.add_argument("--_log_path", metavar="PATH", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess mode: benchmark one llamacpp model and exit
    if args._subprocess:
        prompts = load_prompts(Path(args.prompts))
        _llamacpp_subprocess(args._subprocess, prompts, args._log_path)
        return

    engines = args.engines.split(",") if args.engines else ALL_ENGINES
    prompts = load_prompts(Path(args.prompts))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{timestamp}.log"

    print(f"Prompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")

    all_results = []

    for engine in engines:
        try:
            if engine.startswith("ollama:"):
                model = engine[len("ollama:"):]
                all_results.extend(bench_ollama(model, prompts, log_path))
            elif engine.startswith("server:"):
                model_file = engine[len("server:"):]
                all_results.extend(bench_server(model_file, prompts, log_path))
            elif engine.startswith("llamacpp:"):
                model_name = engine[len("llamacpp:"):]
                print(f"WARNING: 'llamacpp:' engine is deprecated; use 'server:' instead. "
                      f"Falling back to bench_llamacpp.", file=sys.stderr)
                all_results.extend(bench_llamacpp(model_name, prompts, log_path))
            else:
                print(f"Unknown engine: {engine}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: {engine} failed: {e}")
            print(f"  Skipping to next engine.\n")

        print("", file=sys.stderr)

    # Print CSV summary
    header = "Engine,PromptNo,TTFT,TotalTime,Tokens,TokPerSec,GPU%,VRAM_GB,TotalRAM_GB,Trimmed,SwapMB,CacheMB"
    lines = [header]
    for r in all_results:
        lines.append(f"{r['engine']},{r['prompt_no']},{r['ttft']},{r['total_time']},{r['tokens']},{r['tok_per_sec']},{r['gpu_pct']},{r['vram_gb']},{r['total_ram_gb']},{r['trimmed']},{r['swap_mb']},{r['cache_mb']}")

    print(f"\n{'='*60}")
    print("CSV Results:")
    print(f"{'='*60}")
    for line in lines:
        print(line)

    if args.csv:
        Path(args.csv).write_text("\n".join(lines) + "\n")
        print(f"\nSaved to {args.csv}")

    print(f"\nConversation log: {log_path}")


if __name__ == "__main__":
    main()
