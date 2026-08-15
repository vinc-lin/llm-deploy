#!/usr/bin/env python
"""Classify a ctx-bin's decode topology, so nobody compares a blended tok/s
number against a pure one again.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
Genie picks a graph by numeric best-fit on (AR, CL). A prefill graph whose
attention_mask is [1, AR, AR] registers ctx_size == AR -- a "bertcache" variant
-- and `kvmanager.cpp:421-429` then keeps *generating through it*, one token per
step, re-processing the whole AR-wide window, until the KV cache passes AR. Only
then does the AR-1 decode graph take over.

So a bundle carrying a CL=128 bertcache prefill does not have one decode rate.
It has two, and any tok/s figure measured over a fixed token budget is a
time-weighted blend of them:

    2026-08-13, `qwen3_06b_w8a16_local`, 56-token prompt, 128 generated tokens:
        72 bertcache steps @ 40.1 ms  +  56 AR-1 steps @ 142.0 ms
        = 10,837 ms = 11.72 tok/s  <-- reported for two years as an "AR-1" rate

    the same build's true AR-1 decode, measured 2026-08-15:  6.84 tok/s

That 11.72 figure was then compared against pure-topology bundles and produced
three separate phantom findings -- a "~75% build gap", a "64 ms unexplained
term", and "our builds are +51% faster than the device team's". All three were
this artifact. See `docs/MAX_TPS_QWEN3_0.6B_V4.md` section 1.

The trap is nasty because the blend is FAST: bertcache steps attend over AR
positions instead of CL, so a blended bin flatters itself. The failure mode is a
confident wrong number, not an obviously broken one.

WHAT THIS SCRIPT ASSERTS
------------------------
  pure    -- every logit-producing graph is past-KV (CL > AR). One decode rate.
             Basic-mode tok/s is directly comparable to other pure bins.
  blended -- at least one graph has AR == CL. Basic-mode tok/s is a phase blend
             and is NOT comparable to anything. Quote it only with the prompt
             length and token budget attached.

Run:
  lint_bundle_topology.py <ctxbin.bin | *.info.json> [...]
  lint_bundle_topology.py --require-pure <bin> ...     # exit 1 if any is blended
  lint_bundle_topology.py --prompt 56 --budget 128 <bin>   # project the blend
  lint_bundle_topology.py --stamp <bin>                # emit a manifest fragment
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_info(path: Path) -> dict:
    """Accept a .bin (dump it) or an already-dumped .info.json."""
    if path.suffix == ".json":
        return json.load(open(path))
    out = path.with_suffix(".info.json")
    if not out.exists():
        r = subprocess.run(
            ["qnn-context-binary-utility", "--context_binary", str(path),
             "--json_file", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"qnn-context-binary-utility failed on {path}:\n"
                             f"{r.stderr[:2000]}")
    return json.load(open(out))


def _tensors(graph_info: dict, key: str) -> list:
    return [t.get("info", t) for t in graph_info.get(key, [])]


def graph_shape(graph_info: dict):
    """Return (AR, CL, has_logits, has_past_kv) for one graph."""
    ar = cl = None
    for t in _tensors(graph_info, "graphInputs"):
        if t.get("name") == "attention_mask":
            d = t.get("dimensions", [])
            if len(d) >= 2:
                ar, cl = int(d[-2]), int(d[-1])
    has_past = any(t.get("name", "").startswith("past_key_")
                   and t.get("name", "").endswith("_in")
                   for t in _tensors(graph_info, "graphInputs"))
    has_logits = any(t.get("name") == "logits"
                     for t in _tensors(graph_info, "graphOutputs"))
    return ar, cl, has_logits, has_past


def classify(info: dict):
    rows = []
    for g in info["info"]["graphs"]:
        gi = g["info"]
        ar, cl, has_logits, has_past = graph_shape(gi)
        kind = "?"
        if ar is None:
            kind = "no-mask"
        elif ar == cl:
            kind = "BERTCACHE"          # ctx_size == AR: generates through itself
        elif ar == 1:
            kind = "decode"
        elif has_past:
            kind = "past-KV prefill" if ar > 1 else "decode"
        rows.append(dict(name=gi["graphName"], ar=ar, cl=cl, kind=kind,
                         logits=has_logits, past_kv=has_past))
    blended = any(r["kind"] == "BERTCACHE" for r in rows)
    return rows, blended


def project(rows, prompt: int, budget: int):
    """How many generated tokens run in each phase, given a prompt and budget."""
    bert = [r for r in rows if r["kind"] == "BERTCACHE"]
    if not bert:
        return None
    ar = min(r["ar"] for r in bert)
    if prompt >= ar:
        return dict(bert_tokens=0, ar1_tokens=budget, note=(
            f"prompt ({prompt}) already fills the AR={ar} window -- generation "
            "runs entirely on the AR-1 graph, so this bin behaves as pure "
            "for THIS prompt length only"))
    n_bert = min(budget, ar - prompt)
    return dict(bert_tokens=n_bert, ar1_tokens=budget - n_bert, note=(
        f"{n_bert}/{budget} generated tokens run on the AR={ar} bertcache graph "
        f"(re-processing a {ar}-wide window each step), {budget - n_bert} on AR-1"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bins", nargs="+", type=Path)
    p.add_argument("--require-pure", action="store_true",
                   help="exit 1 if any bin is blended")
    p.add_argument("--prompt", type=int, help="prompt length, for the projection")
    p.add_argument("--budget", type=int, default=128,
                   help="generated-token budget, for the projection")
    p.add_argument("--stamp", action="store_true",
                   help="emit a JSON manifest fragment per bin")
    a = p.parse_args()

    bad = []
    for b in a.bins:
        rows, blended = classify(load_info(b))
        verdict = "BLENDED" if blended else "pure"
        print(f"\n{b.name}: {verdict}")
        for r in sorted(rows, key=lambda r: (r["ar"] or 0)):
            flag = "  <-- blends the decode rate" if r["kind"] == "BERTCACHE" else ""
            print(f"    {r['name']:<12} AR={str(r['ar']):>4} CL={str(r['cl']):>5} "
                  f"{r['kind']:<16} logits={int(r['logits'])} past_kv={int(r['past_kv'])}{flag}")

        # Genie hard rule: (AR, CL) must be unique across graphs.
        seen = {}
        for r in rows:
            k = (r["ar"], r["cl"])
            if k in seen:
                print(f"    FAIL: {r['name']} and {seen[k]} share (AR,CL)={k} "
                      "-- hard Genie load error")
                bad.append(b)
            seen[k] = r["name"]

        if blended:
            print("    NOTE: basic-mode tok/s from this bin is a phase blend. Do "
                  "not compare it against a pure bin; quote it only with the "
                  "prompt length and token budget attached.")
            print("    NOTE: this bin must never ship a lade dialog (AR==CL "
                  "inflates n_process -> heap OOB -> SIGSEGV).")
            if a.prompt is not None:
                pr = project(rows, a.prompt, a.budget)
                if pr:
                    print(f"    PROJECTION (prompt={a.prompt}, budget={a.budget}): "
                          f"{pr['note']}")
            if a.require_pure:
                bad.append(b)

        if a.stamp:
            print("    manifest: " + json.dumps(
                {"ctxbin": b.name, "topology": verdict.lower(),
                 "graphs": [{k: r[k] for k in ("name", "ar", "cl", "kind")}
                            for r in rows],
                 "decode_rate_comparable": not blended}))

    if bad:
        print(f"\nFAIL: {len(set(map(str, bad)))} bin(s) failed", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
