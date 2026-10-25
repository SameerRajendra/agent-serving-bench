# agent-serving-bench

**Which LLM-serving optimisations actually shorten an agent's end-to-end time?** This repo records real
multi-agent traffic through a proxy, replays it against a serving engine with the agent's dependency waves
intact, and measures TTFT, agent wall time and whether the answers change. Agent frameworks lay prompts out
stage-first, so parallel stages that read the same long document never share KV cache; the repo also tests
the layout fix for that.

**Headline results** (Qwen2.5-1.5B-Instruct, vLLM, RTX 4060 laptop GPU; two agents: Accord with 20 sessions,
DeltaFin with 10; full tables in [`results/REPORT.md`](results/REPORT.md))

| Optimisation | First-wave TTFT | Agent wall time (concurrency 1 / 8) | Answers |
|---|---|---|---|
| Prefix caching | -30% to -35% | -0% to +3% / -7% to -11% (Accord) | not compared (exact by design) |
| + shared-first layout + priming | -44% to -49% | -0% / -10% (Accord) | **changed**: 50-57% agreement vs an 84-89% noise floor |
| N-gram speculative decoding | -34% to -41% | **-24% to -32% / -10% to -16% (Accord); -19% / -6% (DeltaFin)** | 86-90% agreement, at the noise floor |
| Suffix decoding (fresh server per run) | -2% to +58% | +6% to +7% / +24% to +49% (Accord) | 83-89% agreement, at the noise floor |

Prefix caching shortens TTFT but not the agent: each session generates about 930 tokens, so it is
decode-bound, and speculation, which attacks decode, is what pays. An early 2x "win" for suffix decoding was a
benchmark leak (the server had memorised identical replayed sessions); the measurement protocol below now
prevents it. SGLang configs and the Modal H100 run are implemented but have not been run yet.
## The problem

Agent traffic is not chat traffic. One task fans out into concurrent LLM calls that read the same document, later calls wait on earlier ones, and outputs are schema-bound JSON. Serving engines have features aimed at this: prefix caching (vLLM automatic prefix caching, SGLang RadixAttention), speculative decoding without a draft model (n-gram prompt lookup; vLLM's `suffix` method) and grammar-constrained output. Prefix caching only helps if the shared text really is a prefix, and the natural way to write a stage is instructions first, then the document.

Accord (negotiation analysis) does exactly that. Its sentiment, behaviors and stance stages all render the transcript identically (`[turn_index=N] speaker: text`) but put it after different system prompts, so no two stages share a prefix. The only reuse available is each stage's own system prompt across sessions.

The fix is a prompt layout, not an engine change: lead every stage with one shared system message (a preamble plus the transcript, identical bytes) and move the stage's own instructions into the user message.

## What was changed in Accord

One new helper, `analysis/prompting.py::stage_messages`, used by `sentiment.py`, `behaviors.py` and `stance.py`. The env var `ACCORD_PROMPT_LAYOUT` selects the layout:

| Value | Messages |
|---|---|
| `stage_first` (default) | the original messages, byte-identical: system = stage instructions, user = instruction line + transcript |
| `shared_first` | system = shared preamble + transcript; user = the stage's instructions + its instruction line |

Checked by capturing the request bodies before and after the change: with `stage_first` they are byte-identical (the whole body, schema included). The `recommend` call is untouched because it carries analysis summaries, not the transcript. In `shared_first` the instruction lines still say "the following turns"; the preamble tells the model that phrase means the transcript above it.

`with_structured_output` is sent as `response_format` (`json_schema`), not as `tools`, with a current `langchain-openai` (checked on 1.7.2). Tools would be rendered by the chat template into the system prompt ahead of the transcript and break sharing too. Older `langchain-openai` releases default to function calling, so install a current one in the recording venv.

DeltaFin is recorded as a second, differently shaped agent (sequential pipeline, plain-text JSON, no engine-side structured output).

| | Accord | DeltaFin (variance explanation, "flux") |
|---|---|---|
| Shape | 3 stage calls in parallel (sentiment, behaviors, stance), then `recommend` after all three | 1 `explain_drivers` call per drilled-down account in sequence, then `synthesize_brief` |
| Output | schema-bound (`response_format`) | plain text asked to be JSON |
| Recorded with | `agent.graph.run(transcript, use_rag=False)`, the repo's no-RAG arm | `process_period(..., isolated=True)`, the repo's mode that reads and writes no memory or artifacts |

## Method

**Recording** (`record/`). `proxy.py` is a logging reverse proxy in front of any OpenAI-compatible server. For every `/v1/chat/completions` request it stores the exact body, the session (set by the recorder through `POST /session/<id>`) and `deps_done`: how many responses of that session had completed when the request arrived. Parallel calls share a value (Accord: 0, 0, 0, then 3); a sequential pipeline counts 0, 1, 2, and so on. `record_accord.py --layout stage_first|shared_first` records the same K transcripts with the same model and the repo's own temperature; each layout gets its own trace file.

**Replay** (`replay.py`). Replays recorded bodies at temperature 0 with the recorded `max_tokens`, streaming so TTFT is measured. A request with `deps_done = d` starts once d responses of its session have completed. `--sessions N --concurrency C` replays the first N sessions with C in flight. Per request it records TTFT, latency, output tokens and the response text; per session, agent wall time. It scrapes `/metrics` before and after for the prefix-cache hit rate and speculative acceptance rate.

`--prime`: the three wave-0 requests arrive at one instant, so an engine may compute the shared prefix three times before any of it is cached. With `--prime`, each session first sends the messages all wave-0 requests share as a 1-token request and only then releases wave 0. Its cost is inside the session wall time and is also reported as `prime p50`. Sessions with no shared leading message (`stage_first`, DeltaFin) send no prime.

**Arms** (`grid.py`). Each config launches a server and writes `results/<hw>/<config>.json`.

| Config | Engine flag |
|---|---|
| `vllm-cache-off` (baseline) | `--no-enable-prefix-caching` |
| `vllm-cache-on` | default |
| `vllm-ngram` | `--speculative-config` ngram, 4 tokens, lookup 2 to 5 |
| `vllm-suffix` | `--speculative-config` suffix, up to 16 tokens (needs `arctic-inference`) |
| `sglang-radix-off` (baseline) | `--disable-radix-cache` |
| `sglang-radix-on` | default |
| `sglang-ngram` | `--speculative-algorithm NGRAM` (CUDA only; disables the overlap scheduler and mixed chunked prefill) |

The SGLang configs are implemented but were not part of the measured grid. Every config replays all three traces (`accord-stage_first`, `accord-shared_first`, `deltafin`). The default cache-on configs (`vllm-cache-on`, `sglang-radix-on`) also replay the structured-output-off arm (`--no-structured` strips `response_format` and tool fields) and the `--prime` arm. So the layout comparison is: `stage_first`, `shared_first` and `shared_first +prime`, each against the same trace on the cache-off baseline. 

**Measurement protocol.** Before measuring an agent's traces, every server replays 2 sessions of the other agent's trace (untimed), so no measured session has been seen by the server; the prefix cache is flushed before every run. Configs whose speculation learns from earlier requests (`vllm-suffix`, `sglang-ngram`) get a fresh server for every measured run, because at temperature 0 a replayed session regenerates the same text and a warmed-up drafter would predict it from history. Within one run, sessions still learn from each other, which is how an agent in production would see it.

**Read the structured-output arm together with the `out tok/session` and `out tok/wave-0 call` columns.** Stripping `response_format` does not only remove constrained decoding: the model writes shorter, unconstrained outputs, so the arm is faster partly because it generates fewer tokens. Compare latency only between runs with matching token counts.

**Answer equivalence** (`compare_outputs.py`, shown in the report). Stage outputs are matched by (session, schema name) and compared at temperature 0 between layouts: categorical fields (emotion per turn, behavior `present` flags, stance mood and flexibility, trajectory direction) must match exactly; numeric scores (escalation, confidences) are compared by absolute difference. It also reports the share of sessions whose three stage outputs are byte-identical. As a noise floor it compares each trace with itself at the lowest and highest concurrency, since batch shape alone can flip a token. The same table compares each speculative config (`vllm-ngram`, `vllm-suffix`, `sglang-ngram`) with the plain config of its engine (`vllm-cache-on`, `sglang-radix-on`) on the same trace and concurrency: speculative decoding should be lossless, so agreement should sit at the noise floor. DeltaFin has no engine-side schema, so there only the share of byte-identical sessions is reported. `recommend` is not compared: its prompt embeds the stage outputs.

### Limits to keep in mind

- Prompts are fixed at recording time. Replay does not feed outputs forward, so each layout's `recommend` prompt contains that layout's recorded stage outputs, and output lengths under replay can differ from the recording.
- `deps_done` is measured, not declared. If a parallel call arrives after a sibling has already finished (a very fast server with slow client-side setup), it gets a higher value than its siblings. Check the traces (see below).
- DeltaFin's `isolated=True` recording has no institutional-memory text in its prompts.
- vLLM exposes acceptance counters; SGLang exposes only per-batch gauges, so its acceptance rate is `-` and the effect shows in latency instead.
- One run per cell and a small number of sessions: read p95 as indicative and small deltas as noise.

Metrics read from `/metrics` (names from the vLLM and SGLang docs, October 2026; a missing counter yields `-`):

| Quantity | vLLM | SGLang (`--enable-metrics`) |
|---|---|---|
| Prefix-cache hit rate | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` | `realtime_tokens_total{mode="prefill_cache"}` over cache plus `mode="prefill_compute"` |
| Speculative acceptance | `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total` | not available |

## How to run

Use a separate venv per engine, since vLLM and SGLang pin different torch builds. On WSL2 Ubuntu with an NVIDIA GPU, check `nvidia-smi` first.

```bash
python3 -m venv ~/venvs/vllm   && ~/venvs/vllm/bin/pip   install -r requirements-vllm.txt
python3 -m venv ~/venvs/sglang && ~/venvs/sglang/bin/pip install -r requirements-sglang.txt
python3 -m venv ~/venvs/accord    && ~/venvs/accord/bin/pip    install -r /path/to/Accord/requirements.txt
python3 -m venv ~/venvs/deltafin  && ~/venvs/deltafin/bin/pip  install -r /path/to/Money_hackathon/requirements.txt
```

**1. Record** (once; traces are portable to Modal). Each trace gets its own proxy run; the proxy overwrites its `--out` file.

```bash
# terminal 1: model server; the model name must match --model below
source ~/venvs/vllm/bin/activate
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.85 --port 8000

# terminal 2, then terminal 3, once per layout (stop the proxy with Ctrl-C between runs)
python3 -m record.proxy --upstream http://127.0.0.1:8000 --out traces/accord-stage_first.jsonl
source ~/venvs/accord/bin/activate
python -m record.record_accord --repo /path/to/Accord --model Qwen/Qwen2.5-1.5B-Instruct --layout stage_first --k 20
# repeat both with shared_first and traces/accord-shared_first.jsonl

# and for DeltaFin
python3 -m record.proxy --upstream http://127.0.0.1:8000 --out traces/deltafin.jsonl
source ~/venvs/deltafin/bin/activate
python -m record.record_deltafin --repo /path/to/Money_hackathon --model Qwen/Qwen2.5-1.5B-Instruct --k 10
```

The proxy warns on any non-200 upstream response; fix those before using the trace. Sanity-check each trace: `wc -l traces/*.jsonl` (Accord: 4 lines per session; DeltaFin: one per drilled-down account plus 1), and `deps_done` should be 0, 0, 0, 3 within an Accord session.

**2. Laptop grid** (defaults: `Qwen/Qwen2.5-1.5B-Instruct`, `--gpu-mem 0.85`, `--max-model-len 8192`, concurrency 1 and 8, 20 sessions).

```bash
source ~/venvs/vllm/bin/activate   && python grid.py --engine vllm   --hw rtx4060
source ~/venvs/sglang/bin/activate && python grid.py --engine sglang --hw rtx4060
```

A config that fails to launch is skipped with the server log tail printed; rerun it with `--only <config>`. Server logs land next to the results.

**3. Modal H100** (not run yet; Qwen2.5-7B-Instruct, the model Accord was built on; one engine per invocation, 45-minute cap each).

```bash
pip install modal && modal setup
T=traces/accord-stage_first.jsonl,traces/accord-shared_first.jsonl,traces/deltafin.jsonl
modal run modal_app.py --engine vllm   --traces $T
modal run modal_app.py --engine sglang --traces $T
```

**4. Report**

```bash
pip install -r requirements-report.txt && python report.py   # results/REPORT.md and a chart per hardware
```

## Results

Measured on an RTX 4060 laptop GPU with Qwen2.5-1.5B-Instruct under vLLM; full tables in `results/REPORT.md`.

- **Prefix caching** cuts Accord's first-wave TTFT 30–35% (44–49% with the shared-first layout and a priming
  request) but leaves agent wall time within -11% to +3%: each session generates about 930 tokens, so it is decode-bound.
- **N-gram speculative decoding** cuts agent wall time 24–32% on Accord and 19% on DeltaFin at concurrency 1
  (10–16% and 6% at concurrency 8), with outputs inside the run-to-run noise floor.
- **Suffix decoding** accepts more draft tokens (42–54%) but is 6–49% slower here. An earlier 2x "win" came from
  replaying identical sessions on one server; stateful configs now get a fresh server per run.
- **The shared-first layout** raises Accord's cache hit rate from 34.0% to 41.9% but changes answers: 50–57%
  agreement with the original layout against an 84–89% noise floor.

## Layout

```
record/proxy.py            logging reverse proxy (stdlib)
record/record_accord.py    record Accord through the proxy, per prompt layout
record/record_deltafin.py  record DeltaFin through the proxy
replay.py                  async replay: deps_done waves, --prime, metrics scrape, one JSON per run
compare_outputs.py         stage-output agreement between two runs
grid.py                    engine configs x traces x concurrency x arms
modal_app.py               the same grid on one H100
report.py                  results -> REPORT.md + chart
```
