source config.env
trap 'exit' INT
# name="smoothnotsharpnotdeg3"
# mkdir $TARGET_DIR/$name
# uv run main.py --timeit --smooth "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
# uv run main.py --timeit --smooth "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
# uv run main.py --timeit --smooth "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
# name="notsmoothnotsharpnotdeg3"
# mkdir $TARGET_DIR/$name
# uv run main.py --timeit "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
# uv run main.py --timeit "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
# uv run main.py --timeit "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
# name="v4"
# mkdir $TARGET_DIR/$name
# uv run main.py --timeit --deg3 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
# uv run main.py --timeit --deg3 --sharp "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
# uv run main.py --timeit --deg3 --sharp "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
name="cmo2"
mkdir $TARGET_DIR/$name
uv run main.py --timeit --deg3 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
uv run main.py --timeit --deg3 --sharp "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
uv run main.py --timeit --deg3 --sharp "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
name="cmo2_smooth"
mkdir $TARGET_DIR/$name
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
uv run main.py --timeit  --smooth --deg3 --sharp "$DATASET/512x512" -o "$TARGET_DIR/$name/512"
# name="ge2_2_smooth"
# mkdir $TARGET_DIR/$name
# uv run main.py --timeit  --smooth --ge2 "$DATASET/1024x1024" -o "$TARGET_DIR/$name/1024"
# uv run main.py --timeit  --smooth --ge2 "$DATASET/768x768" -o "$TARGET_DIR/$name/768"
# uv run main.py --timeit  --smooth --ge2 "$DATASET/512x512" -o "$TARGET_DIR/$name/512"

