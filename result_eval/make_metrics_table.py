"""Print a simple per-method summary table (mean, median) for one metric at
one resolution, from the per-method `{metric}_*.csv` files."""
import argparse

import polars as pl

from plot import find_and_load_metric_csvs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-method mean/median table for a metric at a given resolution."
    )
    ap.add_argument("metric", help="Metric name prefix (e.g. 'Chamfer Distance'); "
                                   "all matching {metric}_*.csv files in csv_dir are joined")
    ap.add_argument("csv_dir", help="Directory to search for CSV files")
    ap.add_argument("resolution", type=int, help="Resolution to report (e.g. 1024)")
    ap.add_argument("--reorder", nargs="+", default=None,
                    help="Keep only these methods, in this order.")
    args = ap.parse_args()

    methods, resolutions, df = find_and_load_metric_csvs(args.csv_dir, args.metric)
    if args.resolution not in resolutions:
        raise SystemExit(
            f"No data for resolution {args.resolution}. Available: {resolutions}"
        )
    if args.reorder is not None:
        reorder = [m.replace('*', ' ') for m in args.reorder]
        unknown = [m for m in reorder if m not in methods]
        if unknown:
            raise SystemExit(f"Methods not found in data: {unknown}. "
                             f"Available: {methods}")
        methods = reorder

    res_df = df.filter(pl.col("resolution") == args.resolution)
    table = pl.DataFrame({
        "method": methods,
        "mean":   [res_df[m].drop_nulls().mean() for m in methods],
        "median": [res_df[m].drop_nulls().median() for m in methods],
        "min": [res_df[m].drop_nulls().min() for m in methods],
    })

    print(f"{args.metric} @ {args.resolution}:")
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, float_precision=5):
        print(table)


if __name__ == "__main__":
    main()
