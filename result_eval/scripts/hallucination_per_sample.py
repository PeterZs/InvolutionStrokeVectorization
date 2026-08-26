"""Per-sample hallucination stats for the correct-intersection metric.

Recomputes the predicted/GT intersection matching per sample (same pipeline
as metrics.CorrectIntersectionRate) for one method at one resolution, then
ranks samples by hallucination rate ((pred - matched) / pred). Useful for
finding small samples (few intersections) that are easy to debug with
scripts/debug_intersection.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

import svgutils
from main import (
    RESULT_SUFFIXES_TO_TRY, load_polylines, load_sample_names, load_svg_to_str,
)
from metrics import _compute_pred_points, _load_gt, _match_gt_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datapaths", help="Path to the datapaths JSON file")
    ap.add_argument("method", help="Method name (key in config['results'])")
    ap.add_argument("--resolution", default="1024")
    ap.add_argument("--n", type=int, default=80, help="Samples per path segment")
    ap.add_argument("--d-thresh", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N samples")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-pred", type=int, default=None,
                    help="Only rank samples with at most this many predicted intersections")
    ap.add_argument("--out", default=None,
                    help="Write the full per-sample table to this CSV")
    args = ap.parse_args()

    with open(args.datapaths) as f:
        config = json.load(f)
    result_dir = config["results"][args.method][args.resolution]
    intersections_dir = config["intersections"]

    sample_map = dict(sorted(load_sample_names(config["input_names"]).items())[:args.limit])

    rows = []
    for i, (name, gt_path) in enumerate(sample_map.items()):
        svg_path = next(
            (os.path.join(result_dir, name + suffix + ".svg")
             for suffix in RESULT_SUFFIXES_TO_TRY
             if os.path.exists(os.path.join(result_dir, name + suffix + ".svg"))),
            None,
        )
        if svg_path is None:
            print(f"{i}/{len(sample_map)} MISSING: {name}", file=sys.stderr)
            continue
        w, h = svgutils.get_svg_dimensions(load_svg_to_str(gt_path))
        gt_pts = _load_gt(intersections_dir, name, w, h)
        if gt_pts is None:
            print(f"{i}/{len(sample_map)} NO ANNOTATIONS: {name}", file=sys.stderr)
            continue
        pred_lines = load_polylines(load_svg_to_str(svg_path), w, h, args.n)
        pred_pts = _compute_pred_points(pred_lines)
        matched = len(_match_gt_indices(pred_pts, gt_pts, args.d_thresh))
        n_pred = len(pred_pts)
        halluc = (n_pred - matched) / n_pred if n_pred > 0 else None
        rows.append({"sample": name, "n_pred": n_pred, "n_gt": len(gt_pts),
                     "matched": matched, "halluc_rate": halluc})
        print(f"{i}/{len(sample_map)} {name}: pred={n_pred} gt={len(gt_pts)} "
              f"matched={matched} halluc={halluc if halluc is None else round(halluc, 3)}",
              file=sys.stderr)

    df = pl.DataFrame(rows)
    if args.out:
        df.write_csv(args.out)
        print(f"Wrote {args.out}", file=sys.stderr)

    ranked = df
    if args.max_pred is not None:
        ranked = ranked.filter(pl.col("n_pred") <= args.max_pred)
    ranked = ranked.sort(["halluc_rate", "n_pred"],
                         descending=[True, False], nulls_last=True)

    suffix = f" (n_pred <= {args.max_pred})" if args.max_pred is not None else ""
    print(f"\nTop {args.topk} by hallucination rate{suffix} "
          f"— {args.method} @ {args.resolution}:")
    with pl.Config(tbl_rows=args.topk, tbl_hide_dataframe_shape=True,
                   float_precision=3):
        print(ranked.head(args.topk))


if __name__ == "__main__":
    main()
