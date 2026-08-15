#!/usr/bin/env bash
# Generate a ctx-bin from an explicit DLC list with an explicit graph-config,
# without touching the shared configs/ directory.
#
# Usage:
#   ctxbin_variant.sh <out_dir> <bin_name_no_suffix> <dlc_csv> <graph_names_csv> [json_overrides]
#
# <json_overrides> is a JSON object merged into the "graphs" entry (e.g.
# '{"dlbc":1}'); a "__context" key is merged into the "context" section instead.
#
# THE TRAP THIS AVOIDS
# --------------------
# A graph's name is baked in at conversion time from the --output_path basename,
# and `graph_names` in the backend config must match the names *inside* the
# ctx-bin exactly. A mismatch does not error: that graph silently reverts to
# backend defaults (4 MB VTCM, 24 MB spill) -- or, under lade, null-pointer
# SIGSEGVs on the first speculation step. This script therefore builds the
# config from the graph names you pass, and then VERIFIES the built bin's actual
# graph list against them, plus the (AR, CL) uniqueness rule Genie dispatches on.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

OUT_DIR=$(realpath -m "${1:?out dir}")
BIN_NAME=${2:?binary name (no .bin)}
DLC_CSV=${3:?comma-separated dlc paths}
GRAPHS_CSV=${4:?comma-separated graph names}
OVERRIDES=${5:-'{}'}

mkdir -p "$OUT_DIR"
CFGDIR=$(mktemp -d)
trap 'rm -rf "$CFGDIR"' EXIT

python3 - "$CFGDIR" "$GRAPHS_CSV" "$OVERRIDES" <<'PYEOF'
import json, sys
cfgdir, graphs_csv, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
ctx_over = overrides.pop("__context", {})
dev_over = overrides.pop("__devices", {})
graph = {"graph_names": graphs_csv.split(","), "O": 3, "vtcm_mb": 16, "hvx_threads": 4}
graph.update(overrides)
# soc_model defaults to 0 (generic). REFERENCE.md 8.4: the SDK maps SA8797 to
# soc_id 72 and Qualcomm document extra O=3 algorithms behind naming it, but it
# has never been A/B'd -- override with '{"__devices": {"soc_model": 72,
# "soc_id": 72}}' to build that arm.
device = {"dsp_arch": "v81", "soc_model": 0, "pd_session": "unsigned",
          "cores": [{"core_id": 0, "perf_profile": "burst",
                     "rpc_control_latency": 100, "rpc_polling_time": 9999}]}
device.update(dev_over)
backend = {
    "graphs": [graph],
    "devices": [device],
    # extended_udma lives in "context", NOT "memory" -- docs/NOTES-htp-config-keys.md
    "context": dict({"weight_sharing_enabled": True}, **ctx_over),
}
json.dump(backend, open(f"{cfgdir}/htp_backend_config.json", "w"), indent=2)
json.dump({"backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_config.json"}},
    open(f"{cfgdir}/htp_config.json", "w"), indent=2)
print("graph_names:", graph["graph_names"])
print("graph opts :", {k: v for k, v in graph.items() if k != "graph_names"})
print("context    :", backend["context"])
print("devices    :", {k: v for k, v in device.items() if k != "cores"})
PYEOF

disk_guard 6
cd "$CFGDIR"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC_CSV" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT_DIR" --binary_file "$BIN_NAME" \
    --config_file htp_config.json

qnn-context-binary-utility --context_binary "$OUT_DIR/$BIN_NAME.bin" \
    --json_file "$OUT_DIR/$BIN_NAME.info.json"

python3 - "$OUT_DIR/$BIN_NAME.info.json" "$GRAPHS_CSV" <<'PYEOF'
import json, sys
info, want = json.load(open(sys.argv[1])), sys.argv[2].split(",")
got = [g["info"]["graphName"] for g in info["info"]["graphs"]]
print("graphs in bin:", got)
if sorted(got) != sorted(want):
    sys.exit(f"FAIL: graph names in bin {got} != configured {want} -- that graph "
             "would silently take backend defaults")
ar_cl = []
for g in info["info"]["graphs"]:
    gi = g["info"]
    m = [t for t in gi.get("graphInputs", [])
         if (t.get("info", t)).get("name") == "attention_mask"]
    if m:
        d = (m[0].get("info", m[0])).get("dimensions", [])
        ar_cl.append((gi["graphName"], tuple(d[-2:])))
print("(AR, CL) per graph:", ar_cl)
seen = {}
for name, k in ar_cl:
    if k in seen:
        sys.exit(f"FAIL: {name} and {seen[k]} share (AR,CL)={k}; Genie picks by "
                 "numeric best-fit and cannot distinguish them")
    seen[k] = name
print("OK: graph names match and every (AR, CL) is unique")
PYEOF
echo "CTXBIN VARIANT READY: $OUT_DIR/$BIN_NAME.bin"
