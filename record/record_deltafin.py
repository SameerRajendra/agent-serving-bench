"""Record DeltaFin's variance-explanation agent (sequential LLM calls) through the proxy.

Each period comparison is one session: one `explain_drivers` call per drilled-down account,
then one `synthesize_brief` call, so `deps_done` counts 0, 1, 2, ... Runs with `isolated=True`,
the repo's own mode that reads no memory and writes no memory or artifacts, leaving the repo
untouched. Install DeltaFin's requirements.txt first, start the model server and
`python -m record.proxy`, then:

    python -m record.record_deltafin --repo ~/mnt/Money_hackathon --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

from record.proxy import set_session


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="path to the DeltaFin (Money_hackathon) checkout")
    p.add_argument("--model", required=True, help="model name served by the upstream server")
    p.add_argument("--proxy", default="http://127.0.0.1:9000")
    p.add_argument("--k", type=int, default=10, help="number of most recent period comparisons")
    args = p.parse_args()

    # app/config.py reads these at import; app/llm.py appends /v1 to the URL.
    os.environ["MODAL_QWEN_URL"] = args.proxy
    # The agent falls back to its deterministic path on a timeout, which would shorten the trace.
    os.environ["LLM_TIMEOUT_S"] = "600"
    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from app import config
    from app.flux import graph, ingest

    config.QWEN_MODEL_NAME = args.model  # hard-coded to "qwen-reasoner" in the repo

    conn = ingest.connect()
    periods = ingest.list_periods(conn)[1:][-args.k :]  # the first period has no prior to compare
    for i, period in enumerate(periods):
        set_session(args.proxy, f"deltafin-{period}")
        graph.process_period(period=period, conn=conn, isolated=True)
        print(f"recorded {i + 1}/{len(periods)}")


if __name__ == "__main__":
    main()
