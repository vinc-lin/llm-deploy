#!/usr/bin/env bash
# Segmented, resumable download for the QAIRT SDK zip (server supports ranges).
# Each segment retries independently; alternates direct/proxy on failures.
set -u
URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.48.40.260702/v2.48.40.260702.zip"
TOTAL=2387723706
DIR=/mnt/x/code/llm-deploy/downloads/segments
FINAL=/mnt/x/code/llm-deploy/downloads/qairt-2.48.40.260702.zip
PROXY=http://127.0.0.1:17890
NSEG=12
mkdir -p "$DIR"

# Seed segment 0 from any previously downloaded prefix
if [ -f "$FINAL" ] && [ ! -f "$DIR/seeded" ]; then
    mv "$FINAL" "$DIR/part_seed"
    touch "$DIR/seeded"
fi
SEED=$( [ -f "$DIR/part_seed" ] && stat -c%s "$DIR/part_seed" || echo 0 )

REST=$((TOTAL - SEED))
SEGLEN=$(( (REST + NSEG - 1) / NSEG ))

fetch_seg() {
    local idx=$1 start=$2 end=$3 f="$DIR/part_$1"
    local len=$((end - start + 1))
    for attempt in $(seq 1 60); do
        local have=0
        [ -f "$f" ] && have=$(stat -c%s "$f")
        [ "$have" -ge "$len" ] && return 0
        local from=$((start + have))
        local px=()
        [ $((attempt % 2)) -eq 0 ] && px=(-x "$PROXY")
        curl -sL --max-time 600 --speed-limit 10240 --speed-time 30 \
             "${px[@]}" -r "${from}-${end}" "$URL" >> "$f" 2>/dev/null
        sleep 2
    done
    return 1
}

pids=()
for i in $(seq 0 $((NSEG - 1))); do
    s=$((SEED + i * SEGLEN))
    e=$((s + SEGLEN - 1))
    [ "$e" -ge "$TOTAL" ] && e=$((TOTAL - 1))
    [ "$s" -gt "$e" ] && continue
    fetch_seg "$i" "$s" "$e" &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -ne 0 ] && { echo "SEGMENT FAILURE"; exit 1; }

# Assemble
cat "$DIR/part_seed" > "$FINAL" 2>/dev/null || : > "$FINAL"
for i in $(seq 0 $((NSEG - 1))); do
    [ -f "$DIR/part_$i" ] && cat "$DIR/part_$i" >> "$FINAL"
done
SZ=$(stat -c%s "$FINAL")
if [ "$SZ" -eq "$TOTAL" ]; then
    echo "ASSEMBLED OK: $SZ bytes"
    rm -rf "$DIR"
else
    echo "SIZE MISMATCH: $SZ != $TOTAL"
    exit 1
fi
