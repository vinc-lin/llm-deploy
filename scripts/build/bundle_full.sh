#!/usr/bin/env bash
# Assemble a bundle WITH its full dialog set, gate it, and tar it.
#
# Usage: bundle_full.sh <bundle_name> <ctxbin_path> <mode>
#   mode = basic   -> genie_dialog.json (basic) only
#   mode = lade    -> genie_dialog.json (lade) + genie_dialog_basic.json
#                     + genie_dialog_demo.json
#
# WHY THIS EXISTS
# ---------------
# bundle.sh installs exactly ONE dialog. The extra dialogs for a lade bundle
# were previously added by hand, following a snippet in BUILD_GUIDE.md -- the
# one step in the pipeline with no gate on it. That is precisely how a demo
# dialog carrying `max-num-tokens` (which SIGSEGVs with type "lade", exit 139)
# reached three shipped bundles. This script does the hand-add and then runs
# the linter over the finished directory, so the gate cannot be skipped.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?bundle name}
CTXBIN=${2:?ctx-bin path}
MODE=${3:?mode: basic|lade}

CFG=$LLMDEPLOY_ROOT/configs
OUT=$LLMDEPLOY_DATA/bundles/$NAME
BIN_NAME=$(basename "$CTXBIN")

case "$MODE" in
    basic) PRIMARY=$CFG/genie_dialog_qwen3_0.6b.json ;;
    lade)  PRIMARY=$CFG/genie_dialog_qwen3_0.6b_lade.json ;;
    *)     echo "ABORT: mode must be basic|lade" >&2; exit 1 ;;
esac

disk_guard 6
"$LLMDEPLOY_ROOT/scripts/build/bundle.sh" "$NAME" "$CTXBIN" "$PRIMARY"

if [ "$MODE" = "lade" ]; then
    python3 - "$OUT" "$BIN_NAME" "$CFG" <<'PYEOF'
import json, sys
from pathlib import Path
out, bin_name, cfg = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for src, dst in ((cfg / "genie_dialog_qwen3_0.6b.json", "genie_dialog_basic.json"),
                 (cfg / "genie_dialog_qwen3_0.6b_lade_demo.json", "genie_dialog_demo.json")):
    d = json.loads(src.read_text())
    d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = [bin_name]
    (out / dst).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print(f"  added {dst}")
PYEOF
fi

# The gate that bundle.sh could not apply, because these files did not exist yet.
python3 "$LLMDEPLOY_ROOT/scripts/validate/lint_bundle_dialogs.py" "$OUT"

disk_guard 4
tar -C "$LLMDEPLOY_DATA/bundles" -czf "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz" "$NAME"
ls -lh "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz"
echo "BUNDLE READY: $NAME"
