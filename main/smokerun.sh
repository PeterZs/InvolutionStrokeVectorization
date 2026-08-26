source config.env
trap 'exit' INT
set -euo pipefail


name="v110_smooth"
mkdir "$TARGET_DIR/$name"
uv run main.py --timeit  --smooth --deg3 --no-deg4 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
uv run main.py --timeit  --smooth --deg3 --no-deg4 --sharp "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
uv run main.py --timeit  --smooth --deg3 --no-deg4 --sharp "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
name="v111_smooth"
mkdir "$TARGET_DIR/$name"
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
