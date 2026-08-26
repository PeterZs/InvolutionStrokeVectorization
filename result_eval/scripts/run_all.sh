#!/usr/bin/env bash
set -euo pipefail


OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"

methods=(
    "Ours"
    "Mo2021"
    "DeepSketch2024"
    "Bessmeltsev2019"
    "Puhachov2021"
)

pids=()
for method in "${methods[@]}"; do
    log="$OUTPUT_DIR/log_${method// /_}.txt"
    echo "Launching: $method -> $log"
    uv run main.py datapaths.json "$method" --output "$OUTPUT_DIR" >"$log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done

if (( fail )); then
    echo "One or more runs failed; check logs in $OUTPUT_DIR/" >&2
    exit 1
fi
