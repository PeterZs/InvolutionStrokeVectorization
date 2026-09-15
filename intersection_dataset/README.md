# Intersection dataset

Generates the synthetic training data for the [intersection model](../intersection_model/README.md). Each sample is a
set of random polylines that cross each other, together with the ground truth indicating
how the strokes pass through every junction.

For each sample, the generator produces:

- an antialiased raster drawing of the strokes,
- a segmentation image that labels each half edge with its own color,
- the relation matrix that connects half edges belonging to the same stroke,
- the half edge geometry as JSON, plus vector and debug visualizations.

## How a sample is built

1. Sample a set of polylines with `sampler.line_sampler.sample_tri_method`. Candidate
   points are placed using a triangulation, and paths are traced through them.
2. Compute the intersection points and split each polyline at its crossings into half
   edges (`rasterizer`).
3. Color the half edges into a segmentation image and build the adjacency matrix that
   records which half edges continue into which.
4. Validate the sample. Invalid samples are rejected and generation is retried with the same seed.

A polyline configuration is considered valid when:

- every self-loop has a maximum inscribed circle of radius at least `d/2`,
- all intersection points are at least a distance `d` apart,
- it satisfies the red point and black point criterion.

The diagonal entries of the relation matrix are set to 1, because a stroke may end at any
intersection. Half edges are also stored as pairs, so that the original paths can be
reconstructed.

## Setup

```bash
uv sync
```

## Generating samples

The entry point is `main.py`. The sampling parameters are ranges from which values are drawn for each
seed:

```bash
uv run main.py --out_dir samples --from_seed 1 --to_seed 5 \
  --closeness 3 10 --npoints 10 80 --n_add 0 4
```

| Flag                       | Meaning                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `--out_dir`                | Base output directory.                                        |
| `--from_seed`, `--to_seed` | Inclusive range of seeds; one sample per seed.                |
| `--npoints MIN MAX`        | Range for the number of candidate points (3 to 100).          |
| `--closeness MIN MAX`      | Range for how tightly points are packed (at least 3).         |
| `--n_add MIN MAX`          | Range for the number of extra random lines added on top.      |
| `--target_dim`             | Canvas size in pixels (default 500).                          |
| `--debug`                  | Also write split line and path reconstruction visualizations. |

Output is written into one subfolder per output type in `--out_dir`: `vector/`, `img/`,
`segmentation/`, `mat/`, `half_edges_vis/`, `half_edges_pairs/`, `lines_json/`, plus a
`stats.csv` file with per-sample statistics.

`main_method2.py` is an alternative generation method, kept for comparison.

## Generating on a cluster

Large runs are submitted with SLURM and packaged into tar archives for training.
The scripts read `OUT_DIR` (the output location) and `SLURM_ACCOUNT` from a `config.env` file.

```bash
# generate
bash scripts/submit.sh

# inspect a generated dataset
uv run scripts/analyze_dataset.py <path/to/dataset>

# pack into tar archives consumed by intersection_model (paths relative to OUT_DIR)
bash scripts/submit_make_tars.sh <DatasetName> train/<DatasetArchiveName>
```

`analyze_dataset.py` reports column sums over the samples (the number of junctions of each
degree, how many are terminal, how many have all tangents resolved, and so on), which
is useful for checking that the generated distribution looks reasonable.

## Layout

| Path                        | Content                                                                                                                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sampler/`                  | Polyline sampling. `line_sampler.py` is the driver; `triangulations.py`, `hobbycurve.py`, `randompath.py`, `stars.py`, `intersections.py` and `transform.py` provide the building blocks. |
| `rasterizer.py`             | Intersection splitting, half edge handling, segmentation coloring, the adjacency matrix, validity checks and reconstruction.                                                              |
| `curve_intersections/`      | Bézier curve intersection code (see its [README](curve_intersections/README.md)).                                                                                                         |
| `plotutils.py`, `plot_*.py` | Visualization helpers.                                                                                                                                                                    |
| `scripts/`                  | Cluster generation and packaging (see above).                                                                                                                                             |
