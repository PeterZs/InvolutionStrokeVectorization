# Stroke Vectorization via Involution Prediction

This repository contains the source code for our VMV 2026 paper
[Stroke Vectorization via Involution Prediction](https://igl.ethz.ch/projects/involution-stroke-vectorization/).
It covers running the whole vectorization pipeline, model training and dataset generation.

## Repository layout

| Folder                  | What it does                                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `main/`                 | The end-to-end inference pipeline (documented below).                                                                             |
| `centerline_model/`     | Trains the stage 1 centerline extraction model. See [centerline_model/README.md](centerline_model/README.md).                     |
| `intersection_model/`   | Trains the stage 3 intersection resolution model. See [intersection_model/README.md](intersection_model/README.md).               |
| `dataset_gen/`          | Generates the training image pairs for the centerline model. See [dataset_gen/README.md](dataset_gen/README.md).                  |
| `intersection_dataset/` | Generates the synthetic dataset for the intersection model. See [intersection_dataset/README.md](intersection_dataset/README.md). |
| `result_eval/`          | Computes benchmark metrics and produces the comparison plots. See [result_eval/README.md](result_eval/README.md).                 |
| `svgutils/`             | Shared SVG helpers used by `dataset_gen/` and `result_eval/`. See [svgutils/README.md](svgutils/README.md).                       |

Each subproject is a self-contained [uv](https://docs.astral.sh/uv/) project with its
own `pyproject.toml`, lockfile and virtual environment. Run commands from inside the
folder you are working in.

## Running the vectorization pipeline

The pipeline lives in `main/`. Run everything from that directory.

### Setup

```bash
cd main
uv sync
touch config.env
```

The pipeline loads two TorchScript models at runtime, whose locations are read from the
`config.env` file:

```env
CENTERLINE_MODEL=<path/to/centerline_traced.pt>
INTERSECTION_MODEL=<path/to/intersection_traced.pt>
```

### Pre-trained models

Download the pre-trained checkpoints:

```bash
curl -LO https://igl.ethz.ch/projects/involution-stroke-vectorization/centerline_checkpoint.pt
curl -LO https://igl.ethz.ch/projects/involution-stroke-vectorization/intersection_checkpoint.pt
```

Then export them to TorchScript:

```bash
uv sync --project ../centerline_model
uv run --project ../centerline_model ../centerline_model/trace_export.py ResNextUNetLarge centerline_checkpoint.pt centerline_traced.pt
```

```bash
uv sync --project ../intersection_model
uv run --project ../intersection_model ../intersection_model/trace_export.py --model SceneGraphVectorModel --save_to intersection_traced.pt --trace_device cpu intersection_checkpoint.pt
```

Finally, register the exported models in `config.env`:

```bash
echo "CENTERLINE_MODEL='centerline_traced.pt'" >> config.env
echo "INTERSECTION_MODEL='intersection_traced.pt'" >> config.env
```

### Single image

```bash
uv run main.py someimage.png --show-more -o results/image
```

The input can be a single image or a directory. In the directory case, every image is
processed, and results are written next to the input unless an output directory is given with `-o`. If an
image fails, its traceback is written to a `bad/` folder and the run continues.

### Command line flags

| Flag                 | Effect                                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `-o`, `--output-dir` | Output directory. Defaults to the input directory.                                                       |
| `--show-more`        | Also save intermediate results (preprocessed image, centerline, polyline SVG, half edge visualizations). |
| `--overlay`          | Draw the result on top of the original input image.                                                      |
| `--skip-preprocess`  | Skip the normalization step (use for inputs that are already clean line art).                            |
| `--noinvert`         | Treat the input as white on black instead of black on white.                                             |
| `--save_lines`       | Stop after polyline extraction and dump the lines as JSON, skipping intersection resolution.             |
| `--save_matrix`      | Save the predicted intersection relation matrix as CSV.                                                  |
| `--smooth`           | Smooth the extracted polylines.                                                                          |
| `--timeit`           | Append the per-image runtime to `runtime.csv`.                                                           |
| `--sharp`            | Graph extraction refinement: replace short trailing branches with sharp V-shaped tips.                   |
| `--deg3`             | Graph extraction refinement: special handling of degree-3 junction clusters.                             |
| `--no-deg4`          | Disable the special handling of degree-4 junction clusters (enabled by default).                         |
| `--ge2`              | Use the alternative `graph_extraction2` pipeline. This option is currently disabled in `main.py`.        |

### Examples

```bash
# overlay the result on the input
uv run main.py drawings/bag.png --show-more --overlay -o results/bag

# already clean, white-on-black line art
uv run main.py drawings/hat.png --skip-preprocess --noinvert --save_lines

# extraction refinements plus smoothing
uv run main.py drawings/shell.png --smooth --deg3 --sharp --show-more
```

### Batch evaluation

`run_all.sh` is a convenience script that runs the pipeline over a benchmark dataset at several resolutions and
writes the timings to CSV. It reads `DATASET` and `TARGET_DIR` from `config.env`:

```env
TARGET_DIR=<path/to/output/root>
DATASET=<path/to/benchmark/dataset>
```

A typical invocation inside that script looks like:

```bash
uv run main.py --timeit --deg3 --sharp "$DATASET/1024x1024" -o "$TARGET_DIR/run_name/1024"
```

The results can then be evaluated with [`result_eval`](result_eval/README.md).

## Citation

```
@inproceedings{Gerstner:InvolutionStrokeVectorization:2026,
  title = {Stroke Vectorization via Involution Prediction},
  author = {Gerstner, Philipp and Magne, Tanguy and Sorkine-Hornung, Olga},
  booktitle = {Proceedings of the Symposium on Vision, Modeling and Visualization (VMV)},
  publisher = {Eurographics Association},
  year = {2026},
}
```
