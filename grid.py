"""Experiment matrix: launch each engine config, replay every trace against it, save the results.

Run it inside the venv of the engine under test (vLLM and SGLang pin different torch builds):

    python grid.py --engine vllm --hw rtx4060
"""

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

import replay

ROOT = Path(__file__).resolve().parent
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
LAUNCH_TIMEOUT_S = 900

NGRAM = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}
SUFFIX = {"method": "suffix", "num_speculative_tokens": 16}  # needs `pip install arctic-inference`

# (name, engine, extra launch args). Structured on/off and --prime are replay flags, not launch args.
# The cache-off config of each engine is the baseline report.py diffs against.
CONFIGS = [
    ("vllm-cache-on", "vllm", []),
    ("vllm-cache-off", "vllm", ["--no-enable-prefix-caching"]),
    ("vllm-ngram", "vllm", ["--speculative-config", json.dumps(NGRAM)]),
    ("vllm-suffix", "vllm", ["--speculative-config", json.dumps(SUFFIX)]),
    ("sglang-radix-on", "sglang", []),
    ("sglang-radix-off", "sglang", ["--disable-radix-cache"]),
    ("sglang-ngram", "sglang", ["--speculative-algorithm", "NGRAM"]),  # CUDA only; disables overlap scheduler
]

# The configs that also replay the structured-off and prime arms; the rest run only the plain arm.
FULL_ARMS = {"vllm-cache-on", "sglang-radix-on"}

# Speculation that learns from earlier requests: at temperature 0 a replayed session regenerates
# the same text, so a later run would partly predict outputs the server already produced.
# These configs get a fresh server for every measured run.
STATEFUL = {"vllm-suffix", "sglang-ngram"}

# Cache reset between runs, so a run never starts with the previous run's prompts cached.
FLUSH = {"vllm": "/reset_prefix_cache", "sglang": "/flush_cache"}


def launch_cmd(engine, model, gpu_mem, max_len, extra):
    if engine == "vllm":
        base = ["vllm", "serve", model, "--gpu-memory-utilization", str(gpu_mem), "--max-model-len", str(max_len)]
    else:
        base = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--enable-metrics",
                "--mem-fraction-static", str(gpu_mem), "--context-length", str(max_len)]
    return [*base, "--port", str(PORT), *extra]


def arms(name, trace):
    """(structured, prime) combinations to replay for this config and trace."""
    out = [(True, False)]
    if name in FULL_ARMS:
        if replay.has_structured(trace):
            out.append((False, False))
        if replay.has_primeable(trace):
            out.append((True, True))
    return out


def agent_of(trace):
    return Path(trace).stem.split("-")[0]


def tail(path, n=40):
    return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])


def wait_ready(proc, log_path):
    deadline = time.time() + LAUNCH_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}:\n{tail(log_path)}")
        try:
            if httpx.get(f"{BASE}/v1/models", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready after {LAUNCH_TIMEOUT_S}s:\n{tail(log_path)}")


@contextlib.contextmanager
def server(cmd, log_path):
    # VLLM_SERVER_DEV_MODE exposes POST /reset_prefix_cache; SGLang ignores it.
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            wait_ready(proc, log_path)
            yield
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)  # the whole group: vLLM spawns an engine process
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            time.sleep(5)  # let the driver release VRAM before the next launch


def flush_cache(engine):
    r = httpx.post(BASE + FLUSH[engine], timeout=60)
    if not r.is_success:
        print(f"warning: {engine} cache flush returned {r.status_code}", file=sys.stderr)


def iter_results(engine, model, traces, sessions, concurrency, gpu_mem, max_len, log_dir, only=None):
    """Yield (config name, result) as each config finishes; a config that fails to launch is skipped.

    Protocol: each server is warmed up, untimed, on a trace of a different agent than the one about
    to be measured, so no measured session has been seen before. Stateful configs get a fresh server
    per measured run; the others keep one server and flush the prefix cache before every run.
    """
    if len({agent_of(t) for t in traces}) < 2:
        raise ValueError("need traces from at least two agents: the warm-up replays another agent's trace")
    for name, config_engine, extra in CONFIGS:
        if config_engine != engine or (only and name not in only):
            continue
        cmd = launch_cmd(engine, model, gpu_mem, max_len, extra)
        measured = [(t, structured, prime, c) for t in traces for structured, prime in arms(name, t) for c in concurrency]
        launches = [[m] for m in measured] if name in STATEFUL else [measured]
        runs = []
        try:
            for launch in launches:
                with server(cmd, Path(log_dir) / f"{name}.log"):
                    warmed = None
                    for trace, structured, prime, c in launch:
                        if agent_of(trace) != warmed:
                            warm_up = next(t for t in traces if agent_of(t) != agent_of(trace))
                            asyncio.run(replay.run(warm_up, BASE, 2, max(concurrency), True))
                            warmed = agent_of(trace)
                        flush_cache(engine)
                        print(f"[grid] {name} {Path(trace).stem} structured={structured} prime={prime} concurrency={c}", flush=True)
                        runs.append(asyncio.run(replay.run(trace, BASE, sessions, c, structured, prime)))
        except RuntimeError as e:
            print(f"[grid] {name} failed, skipping:\n{e}", file=sys.stderr)
            continue
        yield name, {
            "config": name,
            "engine": engine,
            "engine_version": importlib.metadata.version(engine),
            "model": model,
            "launch_cmd": cmd,
            "runs": runs,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    p.add_argument("--hw", required=True, help="label for results/<hw>/, e.g. rtx4060")
    p.add_argument("--traces", nargs="+", default=[str(ROOT / f"traces/{t}.jsonl") for t in ("accord-stage_first", "accord-shared_first", "deltafin")])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    p.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM --gpu-memory-utilization / SGLang --mem-fraction-static")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--only", nargs="+", help="run just these config names, e.g. to retry one that failed")
    args = p.parse_args()

    out = ROOT / "results" / args.hw
    out.mkdir(parents=True, exist_ok=True)
    for name, result in iter_results(args.engine, args.model, args.traces, args.sessions, args.concurrency,
                                     args.gpu_mem, args.max_model_len, out, args.only):
        (out / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"[grid] wrote {out / name}.json")


if __name__ == "__main__":
    main()
