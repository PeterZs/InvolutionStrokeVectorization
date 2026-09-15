# Result evaluation

Computes the benchmark metrics and comparison plots that quantify how our pipeline performs
against other vectorization methods. This is split into two steps:

1. `main.py` takes the SVG output of each method, aligns it with the ground truth drawings,
   and saves the per-sample and aggregated metrics to CSV files.
2. Separate scripts produce the plots and tables from these results.

The code works for any benchmark dataset, as long as it follows the format described below.

## Configuration

Benchmark-specific paths (ground truth, results) are listed in a JSON file (`datapaths.json`, not tracked in git) that must look as follows:

```json
{
  "input_names": "<path/to/ground_truth_svg>",
  "intersections": "annotation_points",
  "results": {
    "Ours": {
      "1024": "<path/to/results/ours/1024>",
      "768":  "<path/to/results/ours/768>",
      "512":  "<path/to/results/ours/512>"
    },
    "Mo2021": { "1024": "<path/to/results/mo2021/1024>" }
  }
}
```

- `input_names` is the directory of ground truth SVGs, which also defines the set of sample names.
- `intersections` points to the annotated intersection points used by the Correct Intersection Rate metric.
- `results` maps each method to its result directories, per resolution. When looking up a method's output for a sample, several filename
  suffixes are tried (`""`, `_final`, `_final_no_smoothing`, `_result_linenoise0`).

## Setup

```bash
uv sync
```

## Running

All commands below are run from the `result_eval/` folder.

Evaluate one method, writing one CSV per metric into the output directory:

```bash
uv run main.py datapaths.json Ours --output result_data
```

`scripts/run_all.sh` evaluates all methods listed in the script in parallel:

```bash
bash scripts/run_all.sh result_data
```

This produces per-metric CSVs (for example `Chamfer Distance_Ours.csv`) and a
`success_rate_<method>.csv` file recording how many samples each method handled.

## Plots and tables

`plot.py` renders box or IQR plots from the CSVs. `scripts/plot_all.sh` is the wrapper that
produces the figures used in the paper (set `DIR` and `methods` at the top of the script):

```bash
bash scripts/plot_all.sh
```

For example:

```bash
uv run plot.py "Chamfer Distance" result_data --reorder Ours DeepSketch2024 Mo2021 \
  --ymax 9 --iqr --output chamfer_iqr.pdf
```

For tables:

- `make_metrics_table.py` produces a table for one metric (at a given resolution) instead of a plot.
- `make_intersections_table.py` produces the summary table of the Correct Intersection Rate across methods.
- `runtime.py` summarizes the runtime statistics of each method, based on the
  `runtime.csv` files written by the pipeline's `--timeit` flag.

`scripts/all_tables.sh` is the wrapper that produces all tables at once (only at the largest resolution).

## Helper scripts

The following scripts live in `scripts/`:

- `parse_annotations.py`: converts raw intersection annotations into the
  `annotation_points/` format.
- `debug_intersection.py`: visualizes detected versus annotated intersections.
- `top_diff.py`, `top_samples.py`: find the samples where methods differ the most.
- `total_length.py`: total stroke length per sample.
- `recolor.py`: recolors result SVGs for figures.

## Metrics

Per-sample geometric metrics (`metrics.py`):

- **Length diff**: difference in total stroke length between the prediction and the ground truth.
- **Chamfer Distance**: symmetric point-to-point distance between the two polyline sets.
- **Stroke Density ratio**: ratio of stroke densities (number of strokes divided by total arc length), which captures over- or under-segmentation.

Two stateful metrics aggregate results across a method:

- **Correct Intersection Rate**: how often predicted intersections match hand-annotated
  ground truth intersections (from `annotation_points/`).
- **Turning Angle Histogram Distance**: distance between the distributions of turning angles.
