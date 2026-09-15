# svgutils

Shared SVG helpers used by more than one subproject. The package covers SVG parsing and
normalization, affine transforms, sampling points along paths, and rendering polylines back
to SVG or WebP.

The module is imported as `svgutils` from the consuming subprojects, for example:

```python
import svgutils
from svgutils import getpoints, normalize_svg_topath_rect
```

## Installation

This is a regular pip-installable package. Each subproject that needs it lists `svgutils`
as a dependency and installs it in editable mode from this folder, so changes made here are
picked up without reinstalling.

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

| Functions                                                                                                | Purpose                                                                                   |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `normalize_svg`, `normalize_svg_topath`, `normalize_svg_topath_rect`                                     | Parse an SVG, flatten transforms, and rescale paths to fit a square or a rectangle.       |
| `get_paths_with_transform`, `normalize_path_strings`, `normalize_path`                                   | Lower-level path extraction and transform flattening (handles `<path>` and `<polyline>`). |
| `getpoints`, `getpoints_maxdist`                                                                         | Sample point arrays along an svgpathtools `Path`.                                         |
| `points_to_path`, `get_svg_string`, `get_svg_string_paths`                                               | Convert points or paths back to an SVG string.                                            |
| `save_webp_from_svg`, `modify_thickness`, `remove_clip_paths`, `remove_duplicated`                       | Rasterization and SVG cleanup helpers.                                                    |
| `AffineTransform`, `compute_matrix_for_rescaling`, `compute_matrix_for_rescaling_rect`, `rescale_points` | The underlying affine math.                                                               |
