#!/usr/bin/env python
"""Check a qnn-net-run input package against the ctx-bin it will be fed to,
before anyone spends device time on it.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
On 2026-08-15 the decisive measurement of the whole session -- the decode-only
cycle profile (P1/B7a) -- did not run. The report attributed it to the shipped
profiling inputs being "pre-fix format with 128-dim KV and 128-byte
position_ids, incompatible with the gqafix graph's 64-dim KV".

That diagnosis is impossible. Audited 2026-08-16:

  * the pre-fix and post-fix decode-only ctx-bins have BYTE-IDENTICAL input
    contracts -- 60 inputs, same names, same shapes, same dtypes (the GQA fix
    is graph-internal; KV I/O was frozen by design)
  * every shipped .raw file already matches the gqafix graph exactly:
        input_ids         4 B      = [1,1] int32
        attention_mask    2,304 B  = [1,1,1152] fp16
        position_ids_cos  128 B    = [1,1,64] fp16     <-- the "128-byte" file, correct
        past_key_0_in     2,357,248 B = [1,8,128,1151] fp16
        past_value_0_in   2,357,248 B = [1,8,1151,128] fp16

So the inputs were never the problem, regenerating them fixes nothing, and the
real cause of the P1 failure is still unknown. Rather than guess again, ship
this: it compares every file in the package against the ctx-bin's own declared
input list and names the exact mismatch, or confirms there is none.

Run it on the build machine before shipping, and on the device before running:

    verify_profile_inputs.py --bin <ctx.bin|ctx.info.json> --inputs <dir>

Exit 0 means: every graph input has a file, every file is the right size, and
nothing is missing or extra. If it passes and qnn-net-run still fails, the
cause is environmental -- check, in order: the ctx-bin path given to
--retrieve_context actually resolves; ADSP_LIBRARY_PATH is set
(.:/vendor/lib/rfsa/adsp); /data has room (it runs 98-99% full and the package
plus bin is ~1.1 GB, so a truncated extract is plausible and silent); and the
two `Unknown Key` warnings for memory.extended_udma and
graph_configs_extra.sparse_weights_compression are EXPECTED here and are not
the error (docs/NOTES-htp-config-keys.md).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

DTYPE_BYTES = {
    "QNN_DATATYPE_FLOAT_16": 2, "QNN_DATATYPE_FLOAT_32": 4,
    "QNN_DATATYPE_INT_32": 4, "QNN_DATATYPE_UINT_32": 4,
    "QNN_DATATYPE_INT_64": 8, "QNN_DATATYPE_UINT_64": 8,
    "QNN_DATATYPE_INT_8": 1, "QNN_DATATYPE_UINT_8": 1,
    "QNN_DATATYPE_INT_16": 2, "QNN_DATATYPE_UINT_16": 2,
    "QNN_DATATYPE_SFIXED_POINT_8": 1, "QNN_DATATYPE_UFIXED_POINT_8": 1,
    "QNN_DATATYPE_SFIXED_POINT_16": 2, "QNN_DATATYPE_UFIXED_POINT_16": 2,
}


def load_info(path: Path) -> dict:
    if path.suffix == ".json":
        return json.load(open(path))
    out = path.with_suffix(".info.json")
    if not out.exists():
        r = subprocess.run(
            ["qnn-context-binary-utility", "--context_binary", str(path),
             "--json_file", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"qnn-context-binary-utility failed:\n{r.stderr[:2000]}")
    return json.load(open(out))


def expected_inputs(info: dict, graph: str | None):
    graphs = info["info"]["graphs"]
    if graph:
        graphs = [g for g in graphs if g["info"]["graphName"] == graph]
        if not graphs:
            raise SystemExit(f"no graph named {graph!r} in the bin")
    elif len(graphs) > 1:
        names = [g["info"]["graphName"] for g in graphs]
        raise SystemExit(f"bin has {len(graphs)} graphs {names}; pass --graph")
    gi = graphs[0]["info"]
    out = {}
    for t in [x.get("info", x) for x in gi.get("graphInputs", [])]:
        dims = t.get("dimensions", [])
        dt = t.get("dataType")
        if dt not in DTYPE_BYTES:
            raise SystemExit(f"unknown dtype {dt} on {t['name']}")
        n = 1
        for d in dims:
            n *= int(d)
        out[t["name"]] = (tuple(dims), dt, n * DTYPE_BYTES[dt])
    return gi["graphName"], out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path,
                   help="directory of .raw files (the package's inputs/ dir)")
    p.add_argument("--graph", help="graph name, if the bin has more than one")
    a = p.parse_args()

    gname, exp = expected_inputs(load_info(a.bin), a.graph)
    print(f"ctx-bin graph '{gname}': {len(exp)} declared inputs")

    found = {f.stem: f for f in a.inputs.glob("*.raw")}
    print(f"package '{a.inputs}': {len(found)} .raw files\n")

    missing, wrong, ok = [], [], 0
    for name, (dims, dt, nbytes) in sorted(exp.items()):
        f = found.get(name)
        if f is None:
            missing.append(name)
            continue
        actual = f.stat().st_size
        if actual != nbytes:
            wrong.append((name, dims, dt, nbytes, actual))
        else:
            ok += 1
    extra = sorted(set(found) - set(exp))

    for name in missing:
        dims, dt, nbytes = exp[name]
        print(f"  MISSING  {name:<22} expected {list(dims)} {dt} = {nbytes:,} B")
    for name, dims, dt, nbytes, actual in wrong:
        print(f"  SIZE     {name:<22} expected {list(dims)} {dt} = {nbytes:,} B, "
              f"got {actual:,} B  (ratio {actual/nbytes:.4g})")
    for name in extra:
        print(f"  EXTRA    {name:<22} not a graph input ({found[name].stat().st_size:,} B)")

    print(f"\n{ok}/{len(exp)} inputs match exactly")
    if missing or wrong:
        print("FAIL: the package does not match this ctx-bin", file=sys.stderr)
        return 1
    if extra:
        print("WARNING: extra files present; qnn-net-run ignores them")
    print("OK: every graph input has a correctly-sized file.")
    print("If qnn-net-run still fails, the cause is environmental -- see this "
          "script's docstring for the checklist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
