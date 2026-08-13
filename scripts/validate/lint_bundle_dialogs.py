#!/usr/bin/env python
"""Refuse Genie dialog JSONs that crash or mis-load on device.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
`"type": "lade"` together with `"max-num-tokens"` SIGSEGVs on device -- the
process exits 139 on the first speculation step, with no diagnostic (device
measurement report 2026-08-13 §6.1).  That combination shipped inside
`genie_dialog_demo.json` in THREE bundles (fuseqkvgu, socmodel72, hvx8), so
every demo run of every recent bundle died.

It shipped because the demo/basic dialogs are not written by `bundle.sh` -- it
installs exactly one dialog as `genie_dialog.json`.  The extra dialogs are
added afterwards by hand (BUILD_GUIDE.md §"extra dialogs"), i.e. by the one
path in the pipeline with no gate on it.  This linter is that gate, and it runs
over an assembled bundle DIRECTORY so it sees the hand-added files too, not
just the one `bundle.sh` knows about.

Bounded generation for a lade dialog comes from `context.size` plus EOS, not
from `max-num-tokens`.

WHAT THIS DOES NOT CHECK
------------------------
Graph-level contracts -- (AR, CL) uniqueness, no AR==CL graph in a lade bundle,
graph names matching `htp_backend_ext_config.json` -- are properties of the
ctx-bin, not the dialog, and are gated in `ladekv_build.sh` via
`qnn-context-binary-utility`.  This linter only reads JSON.

Run:
  lint_bundle_dialogs.py <bundle_dir> [<bundle_dir> ...]
  lint_bundle_dialogs.py <dialog.json> [<dialog.json> ...]
  lint_bundle_dialogs.py --configs        # lint the repo's configs/ directory
"""
import argparse
import json
import sys
from pathlib import Path

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def is_assembled_bundle(d: Path) -> bool:
    """An assembled bundle has the runner and a ctx-bin next to the dialogs.

    configs/ is a TEMPLATE directory: its dialogs name ctx-bins that only exist
    after bundle.sh rewrites them at assembly time, so the ctx-bin existence
    check must not run there.
    """
    return (d / "genie-t2t-run").is_file() or any(d.glob("*.bin"))


def find_dialogs(target: Path):
    """A dialog is any *.json whose top level has a 'dialog' key."""
    paths = sorted(target.glob("*.json")) if target.is_dir() else [target]
    out = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            out.append((p, None, f"invalid JSON: {e}"))
            continue
        if isinstance(d, dict) and "dialog" in d:
            out.append((p, d["dialog"], None))
    return out


def check_dialog(path: Path, dialog: dict, bundle_dir: Path | None):
    """Return a list of failure strings (empty == pass)."""
    fails = []
    dtype = dialog.get("type")

    # 1. The SIGSEGV pair.
    if dtype == "lade" and "max-num-tokens" in dialog:
        fails.append(
            f'"type": "lade" + "max-num-tokens": {dialog["max-num-tokens"]!r} '
            "SIGSEGVs on device (exit 139, report §6.1). Remove max-num-tokens "
            "(context.size + EOS bound generation) or set type to \"basic\".")

    # 2. Referenced ctx-bins must actually be in the bundle.
    if bundle_dir is not None:
        binary = (dialog.get("engine", {}).get("model", {}).get("binary", {}))
        for name in binary.get("ctx-bins", []):
            if not (bundle_dir / name).is_file():
                fails.append(
                    f"ctx-bins references {name!r}, which is not in the bundle "
                    f"directory {bundle_dir.name}/ -- Genie fails to load.")

    # 3. A lade dialog must carry lade parameters, and vice versa.
    if dtype == "lade" and "lade" not in dialog:
        fails.append('"type": "lade" but no "lade" parameter block.')
    if dtype != "lade" and "lade" in dialog:
        fails.append(f'"lade" parameter block present but type is {dtype!r} '
                     "-- the block is silently ignored; likely a copy/paste error.")

    # 4. lade guardrail: (ngram-1) * (window + gcap) <= 32 (verify graph AR).
    if dtype == "lade" and isinstance(dialog.get("lade"), dict):
        lade = dialog["lade"]
        try:
            budget = (int(lade["ngram"]) - 1) * (int(lade["window"]) + int(lade["gcap"]))
        except (KeyError, TypeError, ValueError):
            fails.append(f'"lade" block has missing/malformed ngram/window/gcap: {lade!r}')
        else:
            if budget > 32:
                fails.append(
                    f"lade guardrail: (ngram-1)*(window+gcap) = {budget} > 32, "
                    "which exceeds the verify graph's AR.")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", type=Path,
                    help="bundle directories and/or dialog JSON files")
    ap.add_argument("--configs", action="store_true",
                    help=f"also lint {REPO_CONFIGS}")
    args = ap.parse_args()

    targets = list(args.targets)
    if args.configs:
        targets.append(REPO_CONFIGS)
    if not targets:
        ap.error("give at least one bundle dir / dialog JSON, or --configs")

    total_fail = 0
    total_checked = 0
    for target in targets:
        if not target.exists():
            print(f"FAIL {target}: does not exist")
            total_fail += 1
            continue
        bundle_dir = (target if target.is_dir() and is_assembled_bundle(target)
                      else None)
        dialogs = find_dialogs(target)
        if not dialogs:
            print(f"     {target}: no dialog JSONs found")
            continue
        for path, dialog, parse_err in dialogs:
            total_checked += 1
            if parse_err:
                print(f"FAIL {path}: {parse_err}")
                total_fail += 1
                continue
            fails = check_dialog(path, dialog, bundle_dir)
            tag = f'type={dialog.get("type")!r}'
            if fails:
                total_fail += 1
                print(f"FAIL {path}  ({tag})")
                for f in fails:
                    print(f"       {f}")
            else:
                print(f"PASS {path}  ({tag})")

    print(f"\n{total_checked} dialog(s) checked, {total_fail} failing")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
