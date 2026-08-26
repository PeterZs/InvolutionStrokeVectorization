# svgutils

Shared SVG helpers used by more than one subproject. It covers SVG parsing and
normalization, affine transforms, sampling points off paths, and rendering polylines back
to SVG or WebP. It used to be copied (and slowly diverging) inside `dataset_gen/` and
`result_eval/`; this is the single source of truth.

The module is imported as `svgutils` from the consuming subprojects, for example:

```python
import svgutils
from svgutils import getpoints, normalize_svg_topath_rect
```

## How it is consumed

This is a normal, pip-installable package. Each subproject that needs it lists `svgutils`
as a dependency and installs it editable from this folder, so edits here are picked up
without reinstalling.

With uv (the subprojects already declare a path source for it):

```bash
cd ../dataset_gen   # or ../result_eval
uv sync
```

With plain pip:

```bash
pip install -e ../svgutils
```

## Notable functions

- `normalize_svg`, `normalize_svg_topath`, `normalize_svg_topath_rect`: parse an SVG,
  flatten transforms, and rescale paths to fit a square or rectangle.
- `get_paths_with_transform`, `normalize_path_strings`, `normalize_path`: lower level path
  extraction and transform flattening (handles `<path>` and `<polyline>`).
- `getpoints`, `getpoints_maxdist`: sample point arrays off an svgpathtools `Path`.
- `points_to_path`, `get_svg_string`, `get_svg_string_paths`: go back from points/paths to
  an SVG string.
- `save_webp_from_svg`, `modify_thickness`, `remove_clip_paths`, `remove_duplicated`:
  rasterization and SVG cleanup helpers.
- `AffineTransform`, `compute_matrix_for_rescaling`, `compute_matrix_for_rescaling_rect`,
  `rescale_points`: the affine math underneath.
