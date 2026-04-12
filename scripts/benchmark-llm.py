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
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = PROJECT_ROOT / "scripts" / "benchmark-prompts.txt"
LOG_DIR = PROJECT_ROOT / "benchmark-logs"

SYSTEM_PROMPT = (
    "You are Kian, a helpful educating cartoon animal -- but it's a secret. "
    "You are talking to an imaginative and curious child in grade 3. "
    "Unless you are explaining, teaching, or telling a story, reply in one or two short sentences. "
    "Your output will be spoken aloud, so never use markdown, asterisks, bullet points, emojis, "
    "or any formatting. Use plain spoken English only."
)


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a text file (one per line, blank lines ignored)."""
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


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

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    responses = []

    for i, prompt in enumerate(prompts):
        history.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": model,
            "messages": history,
            "stream": True,
            "think": False,
            "options": {"num_ctx": 2048},
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
        }
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


def _start_llama_server(model_path: str) -> "subprocess.Popen":
    import subprocess as sp
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-c", "2048",
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
    """Benchmark a model via the PrismML llama-server (OpenAI-compatible API)."""
    model_name = Path(model_file).stem
    engine_name = f"server:{model_name}"
    model_path = str(PROJECT_ROOT / "models" / model_file)

    print(f"\n{'='*60}")
    print(f"Engine: {engine_name}")
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

    proc = _start_llama_server(model_path)
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

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        results = []
        responses = []

        for i, prompt in enumerate(prompts):
            history.append({"role": "user", "content": prompt})

            body = json.dumps({
                "model": model_name,
                "messages": history,
                "stream": True,
                "temperature": 0.5,
                "top_p": 0.85,
            }).encode()

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
            }
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
        n_ctx=2048,
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

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    responses = []

    for i, prompt in enumerate(prompts):
        history.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        ttft = None
        token_count = 0
        full_response = []

        response = llm.create_chat_completion(messages=history, stream=True)
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
        }
        results.append(row)
        response_text = "".join(full_response)
        responses.append(response_text)
        print(f"  Prompt {i+1}: TTFT={row['ttft']:.3f}s  "
              f"tokens={token_count}  "
              f"total={row['total_time']:.1f}s  "
              f"tok/s={row['tok_per_sec']:.1f}", file=sys.stderr)

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
        sys.executable, __file__,
        "--_subprocess", model_name,
        "--prompts", str(DEFAULT_PROMPTS),
        "--_log_path", str(log_path),
    ]
    # Pass prompts via the prompts file (already the default)
    result = sp.run(cmd, capture_output=True, text=True, timeout=600)

    # Print stderr (progress messages) to our stderr
    if result.stderr:
        for line in result.stderr.splitlines():
            print(line)

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed (exit {result.returncode})")

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
    # llama-server (PrismML fork) — manages its own process lifecycle
    "server:Bonsai-8B.gguf",
    # llamacpp last — CUDA memory not reliably freed on Jetson after exit
    "llamacpp:Qwen3.5-2B-Q4_K_M",
    "llamacpp:granite-3.3-2b-instruct-Q4_K_M",
    "llamacpp:ibm-granite_granite-4.0-micro-IQ4_XS",
    "llamacpp:granite-4.0-micro-Q4_K_M",
    "llamacpp:granite-4.0-h-micro-Q4_K_M",
    "llamacpp:qwen3-4b-instruct-2507-q4_k_m",
]


def main():
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
            if engine.startswith("llamacpp:"):
                model_name = engine[len("llamacpp:"):]
                all_results.extend(bench_llamacpp(model_name, prompts, log_path))
            elif engine.startswith("ollama:"):
                model = engine[len("ollama:"):]
                all_results.extend(bench_ollama(model, prompts, log_path))
            elif engine.startswith("server:"):
                model_file = engine[len("server:"):]
                all_results.extend(bench_server(model_file, prompts, log_path))
            else:
                print(f"Unknown engine: {engine}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: {engine} failed: {e}")
            print(f"  Skipping to next engine.\n")

    # Print CSV summary
    header = "Engine,PromptNo,TTFT,TotalTime,Tokens,TokPerSec,GPU%,VRAM_GB,TotalRAM_GB"
    lines = [header]
    for r in all_results:
        lines.append(f"{r['engine']},{r['prompt_no']},{r['ttft']},{r['total_time']},{r['tokens']},{r['tok_per_sec']},{r['gpu_pct']},{r['vram_gb']},{r['total_ram_gb']}")

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
