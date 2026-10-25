"""Record Accord's LangGraph agent (parallel stage calls, then a barrier call) through the proxy.

Runs the repo's own no-RAG arm (`use_rag=False`), so no Postgres is needed. `--layout` sets
Accord's `ACCORD_PROMPT_LAYOUT`; record both layouts from the same transcripts and model, each
through its own proxy run. Install Accord's requirements.txt first and start the model server, then:

    python -m record.proxy --upstream http://127.0.0.1:8000 --out traces/accord-shared_first.jsonl
    python -m record.record_accord --repo ~/mnt/Accord --model Qwen/Qwen2.5-7B-Instruct --layout shared_first
"""

import argparse
import os
import sys
from pathlib import Path

from record.proxy import set_session


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="path to the Accord checkout")
    p.add_argument("--model", required=True, help="model name served by the upstream server")
    p.add_argument("--proxy", default="http://127.0.0.1:9000")
    p.add_argument("--k", type=int, default=20, help="number of transcripts to record")
    p.add_argument("--layout", choices=["stage_first", "shared_first"], default="stage_first")
    args = p.parse_args()

    # agent/llm.py reads these when the first LLM call builds its client.
    os.environ["SGLANG_BASE_URL"] = f"{args.proxy}/v1"
    os.environ["SGLANG_MODEL"] = args.model
    os.environ["ACCORD_PROMPT_LAYOUT"] = args.layout
    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # the outcome model is loaded from a repo-relative path

    from agent.graph import run
    from data.schema import Transcript

    lines = (repo / "data/processed/casino.jsonl").read_text().splitlines()[: args.k]
    for i, line in enumerate(lines):
        set_session(args.proxy, f"accord-{i:03d}")
        run(Transcript.model_validate_json(line), use_rag=False)
        print(f"recorded {i + 1}/{len(lines)}")


if __name__ == "__main__":
    main()
