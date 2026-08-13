#!/usr/bin/env python
"""Prove a DLC's attention actually dropped the GQA KV replication.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
The `--grouped-gqa` export flag has to reach FOUR separately-exported graphs:
prefill and decode (`full_build.sh`), verify32 (`lade_build.sh`) and the past-KV
prefill (`ladekv_build.sh`). The latter two run their own `quantize_aimet.py`
calls and only receive the flag through `FUSE_FLAGS`. Forget it there and those
graphs are exported with the OLD replicating attention while adopting the
grouped build's encodings -- structurally mismatched graphs inside one ctx-bin,
and 74.7% of the decode cost still present in the graphs that kept it.

Nothing else in the pipeline notices: the build succeeds, parity passes (the two
forms are numerically equivalent), the ctx-bin loads. The only symptom is that
the device is slow, which is exactly what the session is trying to measure. So
the topology has to be asserted directly, per graph.

WHAT REPLICATION LOOKS LIKE IN THE DLC
--------------------------------------
The converter lowers ONNX `Expand` into a broadcast multiply, NOT a copy:

    Eltwise_Binary, operation: 13 (= QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
    in  [1, n_kv, 1,   D, T]  x  coef [1,1,rep,1,1] (STATIC)
    out [1, n_kv, rep, D, T]                          <-- 4.72 MB at n_kv=8, D=128, T=1152

followed by a Reshape to [1, n_heads, D, T]. Two per layer (K and V).

Grouped attention instead leaves the cache alone and batches the MatMul over the
KV heads, so the attention MatMul outputs go from [1, n_heads, AR, T] to
[1, n_kv, rep*AR, T].

Run:
  lint_gqa_ops.py <dlc> [<dlc> ...]                 # expect grouped (default)
  lint_gqa_ops.py --expect-replicating <dlc> ...    # baseline builds
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

# A row of qairt-dlc-info's op table: | id | name | type | inputs | outputs | out dims | ...
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\S.*?)\s*\|\s*(\S+)\s*\|")
OUT_DIMS = re.compile(r"\|\s*([0-9]+(?:x[0-9]+)+)\s*\|\s*[A-Z ]+\|")


def dlc_info(dlc: Path) -> str:
    r = subprocess.run(["qairt-dlc-info", "-i", str(dlc)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"qairt-dlc-info failed on {dlc}:\n{r.stderr[:2000]}")
    return r.stdout


def analyse(text: str):
    """Return (replication_ops, attn_matmul_dims) found in the op table."""
    replication = []
    matmuls = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if not m:
            continue
        _id, name, op_type = m.group(1), m.group(2), m.group(3)
        dims_m = OUT_DIMS.search(line)
        dims = dims_m.group(1) if dims_m else "?"

        # The replication op: an Eltwise_Binary whose STATIC second input is the
        # [1,1,rep,1,1] broadcast coefficient. `operation: 13` sits in the
        # parameters column, which wraps onto the following rows.
        if op_type == "Eltwise_Binary" and "Expand" in name:
            window = "\n".join(lines[i:i + 4])
            if "operation: 13" in window:
                replication.append((name, dims))

        if op_type == "MatMul" and "self_attn" in name:
            matmuls.append((name, dims))
    return replication, matmuls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dlcs", nargs="+", type=Path)
    ap.add_argument("--expect-replicating", action="store_true",
                    help="invert: assert the OLD topology (for baseline builds)")
    ap.add_argument("--n-kv", type=int, default=8)
    ap.add_argument("--layers", type=int, default=28)
    args = ap.parse_args()

    failed = 0
    for dlc in args.dlcs:
        if not dlc.is_file():
            print(f"FAIL {dlc}: not a file")
            failed += 1
            continue
        replication, matmuls = analyse(dlc_info(dlc))
        expected_repl = 2 * args.layers if args.expect_replicating else 0

        # Attention MatMuls should batch over n_kv when grouped, n_heads when not.
        lead = {d.split("x")[1] for _, d in matmuls if d != "?" and "x" in d}
        want_lead = str(args.n_kv) if not args.expect_replicating else str(args.n_kv * 2)

        ok = (len(replication) == expected_repl
              and lead == {want_lead}
              and len(matmuls) == 2 * args.layers)

        tag = "PASS" if ok else "FAIL"
        print(f"{tag} {dlc}")
        print(f"       replication ops (Eltwise_Binary MULTIPLY, /Expand): "
              f"{len(replication)}  (expected {expected_repl})")
        print(f"       attention MatMuls: {len(matmuls)} "
              f"(expected {2 * args.layers}), batch dim {sorted(lead) or '-'} "
              f"(expected {want_lead})")
        if replication[:2]:
            for n, d in replication[:2]:
                print(f"         e.g. {n} -> {d}")
        if matmuls[:2]:
            for n, d in matmuls[:2]:
                print(f"         e.g. {n} -> {d}")
        if not ok:
            failed += 1

    print(f"\n{len(args.dlcs)} DLC(s) checked, {failed} failing")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
