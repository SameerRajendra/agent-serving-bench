"""Do two replay runs give the same answers? Used to show a prompt layout or speculative decoding
did not change outputs.

Stage outputs are matched by (session, schema name), so the comparison does not depend on request
order or prompt layout. Categorical fields must match exactly; numeric scores are compared by
absolute difference. Of Accord, only the three transcript stages are compared: the `recommend`
prompt embeds the stage outputs, so it differs between layouts by construction. Requests with no
schema (DeltaFin) are matched by position, which is only meaningful when both runs replay the same
trace; they count towards identical sessions but have no fields to compare.
"""

import json

NUMERIC = {"escalation", "confidence", "trajectory_confidence"}
STAGES = ("SentimentBatch", "BehaviorFlags", "StanceReport")


def flatten(stage, data):
    """One stage's parsed JSON as {(field, item): value}."""
    out = {}
    if stage == "SentimentBatch":
        for t in data["turns"]:
            out["emotion", t["turn_index"]] = t["emotion"]
            out["escalation", t["turn_index"]] = t["escalation"]
    elif stage == "BehaviorFlags":
        for f in data["flags"]:
            out["present", f["name"]] = f["present"]
            out["confidence", f["name"]] = f["confidence"]
    elif stage == "StanceReport":
        for p in data["parties"]:
            out["mood", p["party"]] = p["mood"]
            out["flexibility", p["party"]] = p["flexibility"]
        out["direction", ""] = data["trajectory"]["direction"]
        out["trajectory_confidence", ""] = data["trajectory"]["confidence"]
    return out


def texts(run, by_index):
    rows = [r for r in run["requests"] if "error" not in r and not r.get("prime")]
    return {(r["session"], r["stage"] or r["index"]): r["text"] for r in rows
            if r["stage"] in STAGES or (by_index and not r["stage"])}


def agreement(a, b):
    """Compare two runs of the same trace, or of two traces recorded from the same transcripts."""
    same_trace = a["trace"] == b["trace"]
    ta, tb = texts(a, same_trace), texts(b, same_trace)
    keys = [k for k in ta if k in tb]
    equal_fields = total_fields = unparsed = 0
    diffs = []
    for key in keys:
        if key[1] not in STAGES:
            continue
        try:
            fa, fb = flatten(key[1], json.loads(ta[key])), flatten(key[1], json.loads(tb[key]))
        except (ValueError, KeyError, TypeError):
            unparsed += 1
            continue
        for field in fa.keys() | fb.keys():
            if field[0] in NUMERIC:
                if field in fa and field in fb:
                    diffs.append(abs(fa[field] - fb[field]))
            else:
                total_fields += 1
                equal_fields += fa.get(field) == fb.get(field) and field in fa and field in fb
    sessions = {s for s, _ in keys}
    identical = [s for s in sessions if all(ta[k] == tb[k] for k in keys if k[0] == s)]
    return {
        "sessions": len(sessions),
        "identical_sessions": len(identical) / len(sessions) if sessions else None,
        "categorical_agreement": equal_fields / total_fields if total_fields else None,
        "numeric_mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
        "unparsed_stages": unparsed,
    }
