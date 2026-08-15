#!/usr/bin/env python
"""Diff ctx-bin variants against a control, so a knob that did nothing is
visible before it costs device time.

WHY
---
HTP backend-extension keys are silently ignored when they are wrong: no error,
no warning, exit 0, and a ctx-bin that looks completely normal but does not have
the feature you asked for. This repo has been bitten three times
(docs/NOTES-htp-config-keys.md). The `graph_names` case is caught by
ctxbin_variant.sh's own check; this catches the rest, by the only signal
available offline -- whether the built artifact actually changed.

A knob that changes nothing in the binary has not necessarily done nothing (some
are runtime hints recorded without affecting layout), but a knob that DOES change
the binary has certainly been consumed. Treat "identical" as "unproven, do not
spend a device arm on it without another reason".

Run:
  ctxbin_diff.py --control <ctrl.bin|info.json> <variant.bin|info.json> ...
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_info(path: Path) -> dict:
    if path.suffix == ".json":
        return json.load(open(path))
    out = path.with_suffix(".info.json")
    if not out.exists():
        r = subprocess.run(
            ["qnn-context-binary-utility", "--context_binary", str(path),
             "--json_file", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"qnn-context-binary-utility failed on {path}:\n"
                             f"{r.stderr[:2000]}")
    return json.load(open(out))


def bin_path(p: Path) -> Path:
    """The .bin next to an .info.json, for a file-size comparison."""
    if p.suffix == ".bin":
        return p
    cand = p.with_suffix("")           # strip .json -> ...info
    cand = cand.with_suffix(".bin")    # -> ....bin
    return cand if cand.exists() else p


def summarise(path: Path) -> dict:
    info = load_info(path)
    top = info.get("info", {})
    out = {"file": path.name}
    b = bin_path(path)
    out["file_bytes"] = b.stat().st_size if b.exists() else None
    for k in ("contextBlobSize", "contextMemorySize"):
        if k in top:
            out[k] = top[k]
    graphs = {}
    for g in top.get("graphs", []):
        gi = g["info"]
        row = {}
        for k in ("sharedWeightsSize", "constSize", "spillFillBufferSize",
                  "maxSpillFillBufferSize", "vtcmSize"):
            if k in gi:
                row[k] = gi[k]
        row["n_inputs"] = len(gi.get("graphInputs", []))
        row["n_outputs"] = len(gi.get("graphOutputs", []))
        graphs[gi["graphName"]] = row
    out["graphs"] = graphs
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("variants", nargs="+", type=Path)
    a = p.parse_args()

    ctrl = summarise(a.control)
    print(f"control: {ctrl['file']}  {ctrl.get('file_bytes'):,} B")
    for gname, row in sorted(ctrl["graphs"].items()):
        print(f"   {gname:<12} " + "  ".join(f"{k}={v:,}" if isinstance(v, int)
                                             else f"{k}={v}"
                                             for k, v in row.items()))
    print()

    hdr = f"{'variant':<26}{'file bytes':>16}{'delta':>14}  consumed?"
    print(hdr)
    print("-" * len(hdr))
    for v in a.variants:
        s = summarise(v)
        d = (s["file_bytes"] or 0) - (ctrl["file_bytes"] or 0)
        verdict = "YES - binary changed" if d else "identical -- UNPROVEN"
        tag = s["file"].replace("qwen3-0.6b-w8a16-gqafix-", "").replace(
            "-ladekv_ctx.info.json", "").replace("-ladekv_ctx.bin", "")
        print(f"{tag:<26}{s['file_bytes']:>16,}{d:>+14,}  {verdict}")
        for gname, row in sorted(s["graphs"].items()):
            cr = ctrl["graphs"].get(gname, {})
            diffs = {k: (cr.get(k), row[k]) for k in row if cr.get(k) != row[k]}
            if diffs:
                for k, (x, y) in diffs.items():
                    print(f"      {gname}.{k}: {x:,} -> {y:,}"
                          if isinstance(x, int) and isinstance(y, int)
                          else f"      {gname}.{k}: {x} -> {y}")
    print("\nNOTE: 'identical' does not prove a no-op -- some keys are runtime "
          "hints that do not alter layout. It proves only that nothing "
          "observable offline changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
