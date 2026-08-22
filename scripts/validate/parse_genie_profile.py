#!/usr/bin/env python
"""Probe C -- turn Genie `--profile` JSONs into the metrics table.

The v4 device session reported "~11.1 s" for the pipeline and nothing else, so
we still have no init / TTFT / decode split for a 4B two-shard W8A16 tower.
Those numbers are worth capturing even while the output is garbage: decode rate
is the same compute either way, and it is still the first such measurement on
this silicon.

This parser is deliberately SCHEMA-TOLERANT. The exact key layout Genie writes
is not documented in the SDK tree we have, and guessing key names is how a
harness silently reports zeros. So it walks the whole JSON, reports every
numeric leaf whose path looks like a timing or rate, and prints the raw
document underneath. If it labels something wrongly, the raw dump is right
there to correct it -- better than a confident wrong table.

  $PY_DEPLOY scripts/validate/parse_genie_profile.py v1_short_profile.json ...
"""
import json
import sys
from pathlib import Path

# substrings -> the metric name in OPERATOR_GUIDE.md §5. Matched against the
# lower-cased JSON path, longest first so "time-to-first-token" wins over "time".
HINTS = [
    ("time-to-first-token", "TTFT (ms)"),
    ("time_to_first_token", "TTFT (ms)"),
    ("ttft", "TTFT (ms)"),
    ("prompt-processing-rate", "prefill rate (tok/s)"),
    ("prompt_processing_rate", "prefill rate (tok/s)"),
    ("token-generation-rate", "decode rate (tok/s)"),
    ("token_generation_rate", "decode rate (tok/s)"),
    ("generation-rate", "decode rate (tok/s)"),
    ("ppr", "prefill rate (tok/s)"),
    ("tgr", "decode rate (tok/s)"),
    ("init", "init (ms)"),
    ("total", "total (ms)"),
]


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node


def label_for(path):
    low = path.lower()
    for needle, name in HINTS:
        if needle in low:
            return name
    return None


def main():
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        raise SystemExit(__doc__)
    for f in files:
        print(f"\n===== {f.name} =====")
        try:
            doc = json.loads(f.read_text())
        except Exception as exc:                                  # noqa: BLE001
            print(f"  unparseable: {exc}")
            continue
        leaves = list(walk(doc))
        named = [(label_for(p), p, v) for p, v in leaves]
        hits = [(n, p, v) for n, p, v in named if n]
        if hits:
            print("  recognised:")
            seen = set()
            for n, p, v in hits:
                if n in seen:
                    print(f"    {'':22s} (also {p} = {v})")
                    continue
                seen.add(n)
                print(f"    {n:22s} {v:>14}   [{p}]")
        else:
            print("  no key matched the expected timing/rate names --")
            print("  read the raw dump below and tell us the real key names;")
            print("  do NOT assume a missing metric is zero.")
        print(f"  all numeric leaves ({len(leaves)}):")
        for p, v in leaves:
            print(f"    {p} = {v}")


if __name__ == "__main__":
    main()
