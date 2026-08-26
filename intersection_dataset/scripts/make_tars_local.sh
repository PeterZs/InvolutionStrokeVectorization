#!/bin/bash

# Define directories
BASE="debug_files/train/Dataset2"
OUT="debug_files/permanent/train/Dataset2"

# Get unique subfolder names (e.g. a, b, c) from the first shard as a sample
# Or define them manually if known: NAMES="a b c d"
NAMES=$(ls "$BASE/shard_0") 
echo "$NAMES" | xargs -P 4 -I {} uv run scripts/make_tars.py --base_dir "$BASE" --tar_dir "$OUT" --name {}