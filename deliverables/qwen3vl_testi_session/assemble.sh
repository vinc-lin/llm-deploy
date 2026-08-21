#!/usr/bin/env bash
# Materialise the Test I device package.
#
# Test I's core is one minute of device time and three text files: ask the 4B the
# question in the CHAT-TEMPLATED form it was calibrated for, instead of the raw
# form every previous test used. No rebuild, no new ctx-bins.
#
# It also carries a qnn-net-run kit with the same two prompts as probe cases, so
# the device can reproduce the host measurement (raw clips 1.64x and lands a
# 1.35x boundary gain; templated stays in range at 1.0001) rather than trusting
# it. That kit needs no KV cache -- both cases are single-chunk with an empty
# one -- so nothing here is large and it uploads in one commit.
#
#   qwen3vl_testi_session/
#     TEST_I_templated_prompt.md            <- they read this
#     README.md                             <- the 60-second version
#     ROOTCAUSE_qwen3vl_4b_boundary_gain.md <- why the last four tests misread
#     testi/
#       prompt_2plus2_templated.txt         <- push to /data/local/tmp/testi
#       prompt_weather_templated.txt
#       prompt_2plus2_raw.txt               <- the control, for reference
#       kit/                                <- merge into the v5 folder (optional §5)
#     host_refs/                            <- host side; NOT pushed
#     analyze_realistic_probe.py
#     MANIFEST.md
#
# Usage: assemble.sh [outdir]
set -euo pipefail
source "$(dirname "$0")/../../scripts/env.sh"

OUT=${1:-$LLMDEPLOY_DATA/bundles/qwen3vl_testi_session}
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=${KIT:-$LLMDEPLOY_DATA/work/text_probe_i}
VL_BINS=${VL_BINS:-$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv-u16in}
CASES="i0_raw i1_templated"

disk_guard 2
[ -d "$KIT" ] || { echo "FATAL: no kit at $KIT -- build it on tank first:"; \
    echo "  build_text_probe_kit.py --suite i ..."; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/testi/kit"

echo "== docs =="
cp "$LLMDEPLOY_ROOT/docs/TEST_I_templated_prompt.md" "$OUT/"
cp "$LLMDEPLOY_ROOT/docs/ROOTCAUSE_qwen3vl_4b_boundary_gain.md" "$OUT/"
cp "$LLMDEPLOY_ROOT/scripts/validate/analyze_realistic_probe.py" "$OUT/"

echo "== prompts =="
# Copied byte-for-byte. The trailing newline after <|im_start|>assistant is
# LOAD-BEARING: it is part of what apply_chat_template emits, and $(cat file)
# strips it, which is why the guide uses the `printf x` trick.
cp "$HERE/prompts/"*.txt "$OUT/testi/"
for f in "$OUT/testi/"*.txt; do
    printf '   %-34s %s bytes\n' "$(basename "$f")" "$(stat -c%s "$f")"
done

echo "== qnn-net-run kit =="
for c in $CASES; do
    [ -d "$KIT/$c" ] || { echo "FATAL: kit is missing case $c"; exit 1; }
    cp -r "$KIT/$c" "$OUT/testi/kit/"
done
cp "$KIT/cases.json" "$OUT/testi/kit/"
printf '%s\n' $CASES > "$OUT/testi/kit/probe_cases.txt"
cp "$LLMDEPLOY_ROOT/configs/run_text_probe.sh" "$OUT/testi/kit/"

echo "== host-side references (not pushed to the device) =="
mkdir -p "$OUT/host_refs"
for c in $CASES; do
    mkdir -p "$OUT/host_refs/$c"
    cp -r "$KIT/$c/ref" "$OUT/host_refs/$c/"
done
cp "$KIT/cases.json" "$OUT/host_refs/"
find "$OUT/testi/kit" -name 'ref' -type d -prune -exec rm -rf {} +

echo "== MANIFEST.md =="
{
    echo "# Test I — manifest"
    echo
    echo "## ctx-bins — NOT shipped; verify these are what you already have"
    echo
    echo '| file | bytes | md5 |'
    echo '|---|---:|---|'
    for n in 1 2; do
        f=$VL_BINS/${n}_of_2/qwen3vl-4b-w8a16_${n}_of_2.bin
        [ -f "$f" ] || continue
        printf '| `%s` | %s | `%s` |\n' "qwen3vl-4b-w8a16_${n}_of_2.bin" \
            "$(stat -c%s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
    done
    echo
    echo "## prompts"
    echo
    echo "The templated files end with a newline after \`<|im_start|>assistant\`."
    echo "That newline is part of the template and \`\$(cat file)\` strips it —"
    echo "use the \`printf x\` form in the guide, not a bare command substitution."
    echo "\`prompt_2plus2_templated.txt\` must tokenize to **20** ids beginning"
    echo "**151644**; verified against the checkpoint tokenizer 2026-08-21."
    echo
    echo '| file | bytes | md5 |'
    echo '|---|---:|---|'
    (cd "$OUT/testi" && for f in *.txt; do
        printf '| `%s` | %s | `%s` |\n' "$f" "$(stat -c%s "$f")" \
            "$(md5sum "$f" | cut -d' ' -f1)"
    done)
    echo
    echo "## kit cases — same question, template the only variable"
    echo
    echo '| case | tokens | sink row | host row RMS |'
    echo '|---|---:|---|---|'
    "$PY_DEPLOY" - "$KIT/cases.json" <<'PY'
import json, sys
for m in json.load(open(sys.argv[1])):
    rms = " / ".join(f"{v:.3f}" for v in m["ref_row_rms"])
    sink = max(range(len(m["ref_row_rms"])), key=lambda i: m["ref_row_rms"][i])
    print(f'| `{m["case"]}` | {m["n_token_rows"]} | row {m["real_rows"][sink]} | {rms} |')
PY
    echo
    echo "Raw puts the sink at row 0 (RMS 107.2) where calibration never saw one;"
    echo "templated puts it at row 1 (RMS 220.3), which calibration covers."
    echo "Measured boundary gain under a full clamp: raw **1.3505**, templated"
    echo "**1.0001**."
} > "$OUT/MANIFEST.md"

cat > "$OUT/README.md" <<'EOF'
# Test I — 60-second version

**No rebuild. No new ctx-bins.** The core check is three text files and one
minute on device.

## The point

Every 4B text test we have ever run fed a **raw** prompt:

```sh
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "What is 2+2? Answer with one number."
```

But the 4B was **calibrated only on chat-templated prompts**
(`apply_chat_template`, every window). Raw input puts the attention sink at row
0, where calibration never saw one; it clips, and position 0's KV is corrupted
in both shards, so everything that attends to it degenerates.

**The 4B has never been asked a question in the form it was built for.** The
0.6B is fine on raw prompts because *its* calibration used raw prompts.

Measured on the host, same question, template the only variable:

| | worst overshoot vs calibrated range | boundary row-0 gain |
|---|---|---:|
| raw | `layers.0/mlp/down_proj` **1.64×** | **1.3505** |
| templated | 1.01× | **1.0001** |

## Run it

```sh
adb push testi /data/local/tmp/testi
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x genie-t2t-run

# A - templated (THE test).  The trailing newline matters, hence `printf x`.
P=$(cat /data/local/tmp/testi/prompt_2plus2_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile testi_templated.json

# B - the same question raw (the old command), as a control
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile testi_raw.json
```

**Check the profile's token count first:** the templated prompt must be **20**
tokens. Much larger means Genie's tokenizer split `<|im_start|>` instead of
matching the added token — itself the finding.

**Expected: A coherent (`4`), B garbage.** That confirms the defect is an input
contract violation and the fix needs no rebuild.

Send back the generated text for both, verbatim, and the profiles.

Full detail: `TEST_I_templated_prompt.md`. Why the last four tests misread this:
`ROOTCAUSE_qwen3vl_4b_boundary_gain.md`.
EOF

echo
du -sh "$OUT"/* 2>/dev/null
echo
echo "Test I package complete: $OUT"
