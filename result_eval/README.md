# Result evaluation

Computes the benchmark metrics and comparison plots that quantify how our pipeline does
against other vectorization methods. This is separated into 2 steps:
- A script takes the SVG output of each method, lines it up
against the ground truth drawings, and saves the per sample and aggregate metrics to a csv file.
- Separate scripts make the plots and tables from these results.

The code works for any benchmark dataset as long as the format is standardised. 

## Configuration

Paths specific to a benchmark (groun truth, results) are listed in a datapaths JSON file (`datapaths.json`) that must look as follows:

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

`input_names` is the directory of ground truth SVGs, which also defines the set of
sample names. `intersections` points at the annotated intersection points used by the
Correct Intersection Rate metric. `results` maps each method to its result directories
per resolution. When looking up a method's output for a sample, several filename
suffixes are tried (`""`, `_final`, `_final_no_smoothing`, `_result_linenoise0`).

## Setup

```bash
uv sync
```

## Running

Evaluate one method, writing one CSV per metric into the output directory:

```bash
uv run main.py datapaths.json Ours --output result_data
```

`run_all.sh` evaluates every method in parallel:

```bash
bash run_all.sh result_data
```

This produces per metric CSVs (for example `Chamfer Distance_Ours.csv`) and a
`success_rate_<method>.csv` recording how many samples each method handled.

## Plots and tables

`plot.py` renders box or IQR plots from the CSVs, and `plot_all.sh` is the wrapper that
produces the figures used in the writeup:

```bash
bash plot_all.sh
```

For example:

```bash
uv run plot.py "Chamfer Distance" result_data --reorder Ours DeepSketch2024 Mo2021 \
  --ymax 9 --iqr --output chamfer_iqr.pdf
```


Regarding tables:
- `make_table.py` produces a table per metric (at a given resolution) instead of a plot.
- `make_intersection_table.py` produces the summary table of the correct intersection rate across methods. 
- `runtime.py` summarizes the runtime statistics of each method, by looking at the 
`runtime.csv` files emitted by the pipeline's `--timeit` flag.

`all_tables.sh` is the wrapper that produces all tables combined (only at the largest resolution).


## Helper scripts

- `parse_annotations.py`: turns raw intersection annotations into the
  `annotation_points/` format.
- `debug_intersection.py`: visualizes detected versus annotated intersections.
- `top_diff.py`, `top_samples.py`: find the samples where methods differ most.
- `total_length.py`: total stroke length per sample.
- `recolor.py`: recolor result SVGs for figures.

## Metrics

Per sample geometric metrics (`metrics.py`):

- Length diff: difference in total stroke length between prediction and ground truth.
- Chamfer Distance: symmetric point to point distance between the two polyline sets.
- Stroke Density ratio: ratio of stroke density, which catches over or under drawing.

Two stateful metrics aggregate across a method:

- Correct Intersection Rate: how often predicted intersections match hand annotated
  ground truth intersections (from `annotation_points/`).
- Turning Angle Histogram Distance: distance between the distributions of turning angles.