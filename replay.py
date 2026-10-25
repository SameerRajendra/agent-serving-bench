"""Replay a recorded agent trace against an OpenAI-compatible server and measure it.

Requests go out with their recorded bodies, so every arm sees identical prompts. Within a
session, a request with `deps_done = d` is released once d responses of that session have
completed, which reproduces the agent's parallel fan-out and sequential steps.

`--prime` first sends, per session, the message prefix all wave-0 requests share as a 1-token
request and only then releases wave 0. A fan-out arrives at one instant, so an engine may
otherwise compute the shared prefix once per request before any of it is cached. The prime
request is part of the session's wall time.

    python replay.py traces/accord.jsonl --base-url http://127.0.0.1:8000 --sessions 20 --concurrency 4
"""

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx

# Fields that make the server constrain output to a schema; dropped by --no-structured.
STRUCTURED_KEYS = ("response_format", "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call")
DEFAULT_MAX_TOKENS = 1024  # for recorded requests that set no limit (DeltaFin's client sets none)
SAMPLE = re.compile(r"^([A-Za-z_:][\w:]*)(\{[^}]*\})? (\S+)", re.M)


def load_trace(path):
    """Requests grouped by session, in recording order."""
    sessions = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        sessions[row["session"]].append(row)
    return list(sessions.values())


def has_structured(path):
    return any(k in r["body"] for s in load_trace(path) for r in s for k in STRUCTURED_KEYS)


def prime_body(requests, model):
    """1-token request carrying the leading messages every wave-0 request shares, or None."""
    wave0 = [r["body"]["messages"] for r in requests if r["deps_done"] == 0]
    shared = []
    for messages in zip(*wave0):
        if any(m != messages[0] for m in messages):
            break
        shared.append(messages[0])
    if len(wave0) < 2 or not shared:
        return None
    body = prepare(requests[0]["body"], model, False)
    body.pop("max_completion_tokens", None)
    return {**body, "messages": shared, "max_tokens": 1}


def has_primeable(path):
    return any(prime_body(s, "") for s in load_trace(path))


def stage_of(body):
    """The structured-output schema name, which identifies a stage in any prompt layout."""
    return (body.get("response_format") or {}).get("json_schema", {}).get("name")


def prepare(body, model, structured):
    body = {k: v for k, v in body.items() if structured or k not in STRUCTURED_KEYS}
    if not {"max_tokens", "max_completion_tokens"} & body.keys():
        body["max_tokens"] = DEFAULT_MAX_TOKENS
    return {**body, "model": model, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}


async def send(client, body):
    """Stream one request; TTFT is the first chunk carrying content (or a tool call)."""
    start = time.perf_counter()
    ttft, tokens, text = None, 0, []
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            delta = (chunk["choices"] or [{}])[0].get("delta", {})
            text.append(delta.get("content") or "")
            if ttft is None and (delta.get("content") or delta.get("tool_calls")):
                ttft = time.perf_counter() - start
    e2e = time.perf_counter() - start
    return {
        "ttft_ms": 1000 * (e2e if ttft is None else ttft),
        "e2e_ms": 1000 * e2e,
        "out_tokens": tokens,
        "text": "".join(text),
    }


async def attempt(client, body):
    try:
        return await send(client, body)
    except httpx.HTTPError as e:
        return {"error": repr(e)}


async def run_session(client, requests, model, structured, prime):
    done, cv = 0, asyncio.Condition()
    session = requests[0]["session"]

    async def one(index, req):
        nonlocal done
        async with cv:
            await cv.wait_for(lambda: done >= req["deps_done"])
        row = await attempt(client, prepare(req["body"], model, structured))
        async with cv:  # a failed request still counts, or later waves would never start
            done += 1
            cv.notify_all()
        return {"session": session, "index": index, "deps_done": req["deps_done"], "stage": stage_of(req["body"]), **row}

    start = time.perf_counter()
    rows = []
    primer = prime_body(requests, model) if prime else None
    if primer:
        rows.append({"session": session, "index": -1, "deps_done": -1, "prime": True, **await attempt(client, primer)})
    rows += await asyncio.gather(*(one(i, r) for i, r in enumerate(requests)))
    return time.perf_counter() - start, rows


def parse_metrics(text):
    return [(name, labels, float(value)) for name, labels, value in SAMPLE.findall(text)]


async def scrape(client):
    """Prometheus samples as (name, labels, value); empty if the server exposes no /metrics."""
    r = await client.get("/metrics")
    return parse_metrics(r.text) if r.status_code == 200 else []


def total(samples, name, label=""):
    values = [v for n, labels, v in samples if n == name and label in labels]
    return sum(values) if values else None


def delta(before, after, name, label=""):
    end = total(after, name, label)
    return None if end is None else end - (total(before, name, label) or 0.0)


def ratio(num, den):
    return num / den if num is not None and den else None


def engine_stats(before, after):
    """Prefix-cache hit rate and speculative acceptance rate over the run, None if not exposed.

    vLLM: hit/query token counters and accepted/draft token counters. SGLang: cached vs
    computed prefill tokens (its `cache_hit_rate` gauge is overwritten per batch, so it is
    unusable for a run-level rate) and no acceptance counter, only per-batch gauges.
    """

    def d(name, label=""):
        return delta(before, after, name, label)

    cached = d("sglang:realtime_tokens_total", 'mode="prefill_cache"') or 0.0
    computed = d("sglang:realtime_tokens_total", 'mode="prefill_compute"')
    sglang_hit = ratio(cached, cached + computed) if computed is not None else None
    vllm_hit = ratio(d("vllm:prefix_cache_hits_total"), d("vllm:prefix_cache_queries_total"))
    accept = ratio(d("vllm:spec_decode_num_accepted_tokens_total"), d("vllm:spec_decode_num_draft_tokens_total"))
    return {
        "prefix_cache_hit_rate": vllm_hit if vllm_hit is not None else sglang_hit,
        "spec_acceptance_rate": accept,
    }


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))] if xs else None


def spread(xs):
    return {"p50": pct(xs, 0.50), "p95": pct(xs, 0.95)}


async def run(trace, base_url, sessions, concurrency, structured, prime=False):
    traced = load_trace(trace)[:sessions]
    limits = httpx.Limits(max_connections=None)
    async with httpx.AsyncClient(base_url=base_url, timeout=600, limits=limits) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        before = await scrape(client)
        slots = asyncio.Semaphore(concurrency)

        async def guarded(requests):
            async with slots:
                return await run_session(client, requests, model, structured, prime)

        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(s) for s in traced))
        wall = time.perf_counter() - start
        after = await scrape(client)

    rows = [r for _, session_rows in results for r in session_rows]
    primes = [r for r in rows if r.get("prime") and "error" not in r]
    ok = [r for r in rows if "error" not in r and not r.get("prime")]
    wave0 = [r for r in ok if r["deps_done"] == 0]
    return {
        "trace": Path(trace).stem,
        "model": model,
        "sessions": len(results),
        "concurrency": concurrency,
        "structured": structured,
        "prime": prime,
        "errors": sum("error" in r for r in rows),
        "wall_s": wall,
        "output_tok_s": sum(r["out_tokens"] for r in ok) / wall,
        "ttft_ms": spread([r["ttft_ms"] for r in ok]),
        "e2e_ms": spread([r["e2e_ms"] for r in ok]),
        "wave0_ttft_ms": spread([r["ttft_ms"] for r in wave0]),
        "prime_ms": spread([r["e2e_ms"] for r in primes]),
        "agent_wall_s": spread([w for w, _ in results]),
        **engine_stats(before, after),
        "requests": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--sessions", type=int, default=20, help="replay the first N recorded sessions")
    p.add_argument("--concurrency", type=int, default=1, help="sessions in flight")
    p.add_argument("--no-structured", action="store_true", help="strip response_format / tool schemas")
    p.add_argument("--prime", action="store_true", help="send the shared wave-0 prefix first, per session")
    p.add_argument("--out", help="write the JSON result here instead of stdout")
    args = p.parse_args()
    result = asyncio.run(run(args.trace, args.base_url, args.sessions, args.concurrency, not args.no_structured, args.prime))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
