#!/usr/bin/env python
"""Strip residual encodings for embed/norm/lm_head from an AIMET encodings file
(summary §2.2 step 3). Writes <input>_filtered.encodings.

--keep-head-weight preserves the `lm_head.weight` PARAM encoding while still
stripping every lm_head ACTIVATION encoding. Pass it whenever
`quantize_aimet.py` ran with --quant-head: that flag deliberately keeps
lm_head's 8-bit per-channel weight quantizer (its FP16 weight is ~311 MB of the
~960 MB/token decode stream), and without this flag the filter silently deletes
the very encoding --quant-head produced, so the converter emits an FP16 lm_head
and the build is indistinguishable from a non-quant-head one."""
import argparse
import json
from pathlib import Path

STRIP_SUBSTRINGS = ("embed_tokens", "lm_head")
STRIP_EXACT_PREFIXES = ("norm.",)  # final norm only — NOT layernorms inside layers
HEAD_WEIGHT_NAMES = ("lm_head.weight",)


def keep(name: str, section: str = "", keep_head_weight: bool = False) -> bool:
    if keep_head_weight and section == "param_encodings" and name in HEAD_WEIGHT_NAMES:
        return True
    if any(s in name for s in STRIP_SUBSTRINGS):
        return False
    if any(name.startswith(p) or name == p.rstrip(".") for p in STRIP_EXACT_PREFIXES):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encodings")
    ap.add_argument("--keep-head-weight", action="store_true",
                    help="retain the lm_head.weight param encoding "
                         "(required when quantize_aimet.py ran with --quant-head)")
    args = ap.parse_args()
    p = Path(args.encodings)
    data = json.loads(p.read_text())

    report = {}
    kept_head = False
    for section in ("activation_encodings", "param_encodings"):
        if section not in data:
            continue
        before = len(data[section])
        if isinstance(data[section], dict):  # AIMET 0.x dict schema
            data[section] = {k: v for k, v in data[section].items()
                             if keep(k, section, args.keep_head_weight)}
            names = data[section].keys()
        else:  # AIMET 2.x "1.0.0" list schema: entries carry a "name" field
            data[section] = [e for e in data[section]
                             if keep(e.get("name", ""), section, args.keep_head_weight)]
            names = [e.get("name", "") for e in data[section]]
        if section == "param_encodings":
            kept_head = any(n in HEAD_WEIGHT_NAMES for n in names)
        report[section] = (before, len(data[section]))

    if args.keep_head_weight and not kept_head:
        raise SystemExit(
            "--keep-head-weight was requested but no lm_head.weight param encoding "
            "is present in the input. Did quantize_aimet.py run with --quant-head?")

    out = p.with_name(p.stem + "_filtered" + p.suffix)
    out.write_text(json.dumps(data, indent=2))
    for sec, (b, a) in report.items():
        print(f"{sec}: {b} -> {a} (stripped {b - a})")
    if args.keep_head_weight:
        print("retained lm_head.weight param encoding (--keep-head-weight)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
