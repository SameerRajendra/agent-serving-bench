"""Run the grid on one H100 with Qwen2.5-7B-Instruct, the model Accord was built on.

    modal run modal_app.py --engine vllm   --traces traces/accord.jsonl,traces/deltafin.jsonl
    modal run modal_app.py --engine sglang --traces traces/accord.jsonl,traces/deltafin.jsonl

One engine per invocation, since each needs its own container image. Traces are sent as an
argument; each config's result streams back as it finishes and is written to
results/h100/<config>.json, so a timeout loses only the config in progress. The hard cap is
45 minutes per invocation (about $3 at $3.95/h).
"""

import json
from pathlib import Path

import modal

# Same pinned SGLang image family as Accord's benchmarks/bench_serving.py (full image, not
# `-runtime`: NGRAM speculation may need a compiler).
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.16-cu129"
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"
# Qwen2.5-7B is already cached in Accord's volume; HF_HOME keeps its layout.
HF_CACHE_DIR = "/cache/huggingface"
TIMEOUT_S = 45 * 60

app = modal.App("agent-serving-bench")
hf_cache = modal.Volume.from_name("accord-hf-cache", create_if_missing=True)


def with_bench_code(image):
    # add_local_* must come last: Modal forbids build steps after it.
    return (
        image.entrypoint([])  # the base images ship their own ENTRYPOINT
        .env({"HF_HOME": HF_CACHE_DIR, "PYTHONPATH": "/app"})
        .workdir("/app")
        .add_local_file("grid.py", "/app/grid.py")
        .add_local_file("replay.py", "/app/replay.py")
    )


sglang_image = with_bench_code(modal.Image.from_registry(SGLANG_IMAGE).pip_install("httpx>=0.27"))

vllm_image = with_bench_code(
    modal.Image.from_registry(VLLM_IMAGE)
    # arctic-inference provides the `suffix` speculative method; keep the image's torch/vllm.
    .run_commands("pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|tokenizers|vllm)==' > /tmp/constraints.txt")
    .pip_install("arctic-inference", "httpx>=0.27", extra_options="-c /tmp/constraints.txt")
)


def run_grid(engine, model, traces, sessions, concurrency):
    import grid

    trace_dir, log_dir = Path("/tmp/traces"), Path("/tmp/logs")
    trace_dir.mkdir()
    log_dir.mkdir()
    paths = []
    for name, text in traces.items():
        (trace_dir / f"{name}.jsonl").write_text(text)
        paths.append(str(trace_dir / f"{name}.jsonl"))
    yield from grid.iter_results(engine, model, paths, sessions, concurrency, 0.9, 8192, log_dir)


@app.function(image=vllm_image, gpu="H100", volumes={HF_CACHE_DIR: hf_cache}, timeout=TIMEOUT_S)
def run_vllm(model: str, traces: dict, sessions: int, concurrency: list):
    yield from run_grid("vllm", model, traces, sessions, concurrency)


@app.function(image=sglang_image, gpu="H100", volumes={HF_CACHE_DIR: hf_cache}, timeout=TIMEOUT_S)
def run_sglang(model: str, traces: dict, sessions: int, concurrency: list):
    yield from run_grid("sglang", model, traces, sessions, concurrency)


@app.local_entrypoint()
def main(
    engine: str,
    traces: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    sessions: int = 20,
    concurrency: str = "1,8",
):
    fn = {"vllm": run_vllm, "sglang": run_sglang}[engine]
    texts = {Path(p).stem: Path(p).read_text() for p in traces.split(",")}
    out = Path("results/h100")
    out.mkdir(parents=True, exist_ok=True)
    for name, result in fn.remote_gen(model, texts, sessions, [int(c) for c in concurrency.split(",")]):
        (out / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {out / name}.json")
