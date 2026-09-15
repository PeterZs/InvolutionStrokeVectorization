# Vectorization pipeline

This folder contains the end-to-end inference pipeline. See the [root README](../README.md#running-the-vectorization-pipeline) for setup and usage.

## How it fits together

| File                   | Role                                                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`              | Drives the whole pipeline and handles I/O.                                                                                                               |
| `imgpreprocess.py`     | Grayscale conversion, optional inversion and normalization.                                                                                              |
| `centerline.py`        | Loads `CENTERLINE_MODEL`, pads the input to a power of two, runs inference (on MPS when available) and returns the centerline image.                     |
| `graph_extraction/`    | Builds the planar line graph and extracts the polylines and their connectivity.                                                                          |
| `intersections.py`     | Builds the per-junction segmentation and relation matrix, loads `INTERSECTION_MODEL`, and predicts how half edges connect (via maximum-weight matching). |
| `plotutils/`, `utils/` | Shared rendering and helper code, including writing the final SVG.                                                                                       |

The `scripts_experiments/` folder (which also contains the alternative `graph_extraction2.py`, currently disabled in `main.py`)
is scratch space for development and is not part of the pipeline.
