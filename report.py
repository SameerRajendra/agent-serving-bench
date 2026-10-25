"""Turn results/<hw>/<config>.json into results/REPORT.md and one chart per hardware.

    python report.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from compare_outputs import agreement

RESULTS = Path(__file__).resolve().parent / "results"
# Each engine's no-prefix-reuse config is what the deltas compare against.
BASELINE = {"vllm": "vllm-cache-off", "sglang": "sglang-radix-off"}
# Speculative configs must give the same answers as the plain engine, so agreement compares them with it.
SPECULATIVE = {"vllm-ngram", "vllm-suffix", "sglang-ngram"}
REFERENCE = {"vllm": "vllm-cache-on", "sglang": "sglang-radix-on"}
HEADER = ["config", "layout", "conc", "structured", "wave-0 TTFT p50 ms", "wave-0 TTFT p95 ms",
          "agent wall p50 s", "agent wall p95 s", "out tok/s", "out tok/session", "out tok/wave-0 call", "cache hit", "spec accept", "prime p50 ms",
          "errors", "Δ wave-0 TTFT p50", "Δ wall p50"]
AGREEMENT_HEADER = ["config", "conc", "comparison", "sessions", "identical sessions", "categorical agreement",
                    "numeric mean abs diff", "unparsed"]


def load():
    """{(hw, agent): rows}; traces are named <agent>[-<layout>], one row per replay run."""
    tables = defaultdict(list)
    for path in sorted(RESULTS.glob("*/*.json")):
        data = json.loads(path.read_text())
        for run in data["runs"]:
            agent, _, layout = run["trace"].partition("-")
            tables[path.parent.name, agent].append({
                "config": data["config"], "engine": data["engine"], "trace": run["trace"],
                "layout": layout or "-", "conc": run["concurrency"], "structured": run["structured"],
                "prime": run.get("prime", False), "run": run,
            })
    return tables


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def change(value, base):
    return "-" if value is None or not base else f"{100 * (value / base - 1):+.0f}%"


def mean_out_tokens(run):
    """(per session, per wave-0 call). A run that produces shorter outputs does less work and looks faster."""
    ok = [r for r in run["requests"] if "error" not in r and not r.get("prime")]
    wave0 = [r["out_tokens"] for r in ok if r["deps_done"] == 0]
    sessions = {r["session"] for r in ok}
    return (sum(r["out_tokens"] for r in ok) / len(sessions) if sessions else None,
            sum(wave0) / len(wave0) if wave0 else None)


def table(rows):
    by_key = {(r["config"], r["trace"], r["structured"], r["conc"], r["prime"]): r["run"] for r in rows}
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for r in sorted(rows, key=lambda r: (r["config"], r["trace"], r["prime"], not r["structured"], r["conc"])):
        run = r["run"]
        per_session, per_wave0 = mean_out_tokens(run)
        base = by_key.get((BASELINE[r["engine"]], r["trace"], r["structured"], r["conc"], False))
        cells = [
            r["config"], r["layout"] + (" +prime" if r["prime"] else ""), r["conc"], "on" if r["structured"] else "off",
            fmt(run["wave0_ttft_ms"]["p50"], ".0f"), fmt(run["wave0_ttft_ms"]["p95"], ".0f"),
            fmt(run["agent_wall_s"]["p50"], ".2f"), fmt(run["agent_wall_s"]["p95"], ".2f"),
            fmt(run["output_tok_s"], ".0f"), fmt(per_session, ".0f"), fmt(per_wave0, ".0f"),
            fmt(run["prefix_cache_hit_rate"], ".1%"),
            fmt(run["spec_acceptance_rate"], ".1%"), fmt(run["prime_ms"]["p50"], ".0f"), run["errors"],
            change(run["wave0_ttft_ms"]["p50"], base and base["wave0_ttft_ms"]["p50"]),
            change(run["agent_wall_s"]["p50"], base and base["agent_wall_s"]["p50"]),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def agreement_table(rows):
    """Layout runs against stage_first, speculative runs against the plain engine, and each trace at two
    concurrencies against itself as the noise floor (batching alone changes some outputs)."""
    runs = {(r["config"], r["layout"], r["prime"], r["conc"]): r["run"] for r in rows if r["structured"]}
    engine = {r["config"]: r["engine"] for r in rows}
    pairs = []
    for (config, layout, prime, conc), run in sorted(runs.items()):
        if layout not in ("-", "stage_first") and (config, "stage_first", False, conc) in runs:
            pairs.append((config, conc, f"{layout}{' +prime' if prime else ''} vs stage_first",
                          run, runs[config, "stage_first", False, conc]))
        ref = REFERENCE[engine[config]]
        if config in SPECULATIVE and not prime and (ref, layout, False, conc) in runs:
            pairs.append((config, conc, f"{run['trace']} vs {ref}", run, runs[ref, layout, False, conc]))
    for config in sorted({k[0] for k in runs}):
        for layout in ("-", "stage_first"):
            concs = sorted(c for (cfg, lay, prime, c) in runs if cfg == config and lay == layout and not prime)
            if len(concs) > 1:
                low, high = runs[config, layout, False, concs[0]], runs[config, layout, False, concs[-1]]
                pairs.append((config, f"{concs[0]} vs {concs[-1]}", f"{low['trace']} vs itself (noise floor)", low, high))
    if not pairs:
        return None
    lines = ["| " + " | ".join(AGREEMENT_HEADER) + " |", "|" + "---|" * len(AGREEMENT_HEADER)]
    for config, conc, label, a, b in pairs:
        ag = agreement(a, b)
        cells = [config, conc, label, ag["sessions"], fmt(ag["identical_sessions"], ".0%"),
                 fmt(ag["categorical_agreement"], ".1%"), fmt(ag["numeric_mean_abs_diff"], ".3f"), ag["unparsed_stages"]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def chart(hw, agents):
    """Agent wall p50 per config and layout at the highest concurrency, structured output on."""
    fig, axes = plt.subplots(1, len(agents), figsize=(7 * len(agents), 5), squeeze=False)
    for ax, (agent, rows) in zip(axes[0], agents.items()):
        top = max(r["conc"] for r in rows)
        picked = [r for r in rows if r["conc"] == top and r["structured"]]
        ax.bar([f"{r['config']} {r['layout']}{' +prime' if r['prime'] else ''}" for r in picked],
               [r["run"]["agent_wall_s"]["p50"] for r in picked])
        ax.set(title=f"{agent}, concurrency {top}", ylabel="agent wall p50 (s)")
        ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(RESULTS / hw / "agent_wall_p50.png", dpi=120)


def main():
    out = ["# Results", "",
           "Deltas compare each run with the same engine's cache-off config on the same trace, concurrency and "
           "structured-output arm; negative is faster. `wave-0` calls are the first requests of a session (for "
           "Accord, the parallel stage calls). `prime` is the shared-prefix request sent before wave 0; it is "
           "inside the agent wall time. `out tok/session` and `out tok/wave-0 call` are mean generated tokens: "
           "compare latency only between runs where they match (the structured-off arm does not). `cache hit` "
           "and `spec accept` are `-` where the engine exposes no usable counter.", ""]
    by_hw = defaultdict(dict)
    for (hw, agent), rows in load().items():
        by_hw[hw][agent] = rows
    for hw, agents in sorted(by_hw.items()):
        chart(hw, agents)
        out += [f"## {hw}", "", f"![agent wall p50]({hw}/agent_wall_p50.png)", ""]
        for agent, rows in sorted(agents.items()):
            out += [f"### {agent}", "", table(rows), ""]
            agreed = agreement_table(rows)
            if agreed:
                out += [f"#### {agent}: output agreement", "", agreed, ""]
    (RESULTS / "REPORT.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
