#!/usr/bin/env python
"""Strip residual encodings for embed/norm/lm_head from an AIMET encodings file
(summary §2.2 step 3). Writes <input>_filtered.encodings."""
import argparse
import json
from pathlib import Path

STRIP_SUBSTRINGS = ("embed_tokens", "lm_head")
STRIP_EXACT_PREFIXES = ("norm.",)  # final norm only — NOT layernorms inside layers


def keep(name: str) -> bool:
    if any(s in name for s in STRIP_SUBSTRINGS):
        return False
    if any(name.startswith(p) or name == p.rstrip(".") for p in STRIP_EXACT_PREFIXES):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encodings")
    args = ap.parse_args()
    p = Path(args.encodings)
    data = json.loads(p.read_text())

    report = {}
    for section in ("activation_encodings", "param_encodings"):
        if section not in data:
            continue
        before = len(data[section])
        if isinstance(data[section], dict):  # AIMET 0.x dict schema
            data[section] = {k: v for k, v in data[section].items() if keep(k)}
        else:  # AIMET 2.x "1.0.0" list schema: entries carry a "name" field
            data[section] = [e for e in data[section] if keep(e.get("name", ""))]
        report[section] = (before, len(data[section]))

    out = p.with_name(p.stem + "_filtered" + p.suffix)
    out.write_text(json.dumps(data, indent=2))
    for sec, (b, a) in report.items():
        print(f"{sec}: {b} -> {a} (stripped {b - a})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
