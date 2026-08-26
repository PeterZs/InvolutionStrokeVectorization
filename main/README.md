This folder contains the end to end inference pipeline. Please refer to the parent folder on usage.

## How it fits together

- `main.py` drives the whole pipeline and handles I/O.
- `imgpreprocess.py` does grayscale conversion, optional inversion and normalization.
- `centerline.py` loads `CENTERLINE_MODEL`, pads the input to a power of two, runs
  inference (on MPS when available) and returns the centerline image.
- `graph_extraction/` (and the alternative `graph_extraction2.py`) build the planar
  line graph and extract polylines plus their connectivity.
- `intersections.py` builds the per junction segmentation and relation matrix, loads
  `INTERSECTION_MODEL`, predicts how half edges connect, and writes the final SVG.
- `plotutils/` and `utils/` hold shared rendering and helper code.

The standalone `*test.py` files and the `debug_files/` directory are scratch space for
development and are not part of the pipeline.
