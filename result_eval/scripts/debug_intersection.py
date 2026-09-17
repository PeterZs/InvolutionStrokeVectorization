"""Debug helper for the correct-intersection metric.

Given an SVG file, compute GT intersections (from the per-sample CSV in the
annotated_intersections folder), prediction intersections, and the matched subset,
then write three SVGs with the drawing in the background so the three stages
can be eyeballed side-by-side.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import svgutils
from main import load_svg_to_str, load_polylines
from metrics import _load_gt, _compute_pred_points, _match_gt_indices


RESULT_SUFFIXES_TO_TRY = ["_final", "_final_no_smoothing", "_result_linenoise0"]


def bbox_str(pts: np.ndarray) -> str:
    return (f"x=[{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}] "
            f"y=[{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")



#uv run debug_intersection.py "your_sample.svg"
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", help="SVG file to debug")
    ap.add_argument("--intersections-dir", default="annotated_intersections",
                    help="Directory containing per-sample GT CSVs")
    ap.add_argument("--n", type=int, default=80, help="Samples per path segment")
    ap.add_argument("--d-thresh", type=float, default=1.5, help="Match distance threshold")
    ap.add_argument("--prefix", default="debug", help="Output filename prefix")
    ap.add_argument("--point-radius", type=float, default=0.5)
    args = ap.parse_args()

    stem: str = os.path.splitext(os.path.basename(args.svg))[0]
    for x in RESULT_SUFFIXES_TO_TRY:
        if stem.endswith(x):
            stem = stem.partition(x)[0]
            break
    

    svg_str = load_svg_to_str(args.svg)
    w, h = svgutils.get_svg_dimensions(svg_str)
    mul = 512.0/max(w, h)
    target_w, target_h = mul*w, mul*h
    lines = load_polylines(svg_str, target_w, target_h, args.n)
    pred_pts = _compute_pred_points(lines)
    gt_pts= _load_gt(args.intersections_dir, stem, target_w, target_h)
    if gt_pts is None:
        raise RuntimeError(
            f"No GT CSV for '{stem}' in {args.intersections_dir}"
        )
    
    
    matched_idx = _match_gt_indices(pred_pts, gt_pts, args.d_thresh)
    matched_pts = gt_pts[matched_idx] if len(matched_idx) else np.zeros((0, 2))
    unmatched_mask = np.ones(len(gt_pts), dtype=bool)
    unmatched_mask[matched_idx] = False
    unmatched_pts = gt_pts[unmatched_mask] if len(gt_pts) else np.zeros((0, 2))

    total_pts = sum(len(p) for p in lines)
    print(f"SVG:                {args.svg}")
    print(f"GT CSV:             {os.path.join(args.intersections_dir, stem + '.csv')}")
    print(f"SVG dimensions:     {w} x {h}")
    print(f"Polylines:          {len(lines)} polylines, {total_pts} sampled points")
    print(f"GT intersections:   {len(gt_pts)}")
    denom = max(1, len(gt_pts))
    print(f"Matched:            {len(matched_pts)} / {len(gt_pts)} "
          f"({len(matched_pts) / denom * 100:.2f}%) @ d_thresh={args.d_thresh}")
    print(f"Unmatched:          {len(unmatched_pts)} / {len(gt_pts)}")

    paths = [svgutils.points_to_path(p) for p in lines if len(p) >= 2]

    def save(points: np.ndarray, color: str, suffix: str) -> None:
        out = f"{args.prefix}_{suffix}.svg"
        with open(out, "w", encoding="utf-8") as f:
            f.write(svgutils.get_svg_string_paths(
                paths, w, h,
                points=points,
                point_radius=args.point_radius,
                pointcolor=color,
            ))
        print(f"  wrote {out} ({len(points)} points)")

    save(gt_pts, "red", "gt")
    save(pred_pts, "blue", "pred")
    save(matched_pts, "green", "matched")
    save(unmatched_pts, "orange", "unmatched")


if __name__ == "__main__":
    main()
