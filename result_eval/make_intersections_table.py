"""Collect the per-method `Correct Intersections` results and print a combined
table (plain + LaTeX booktabs, in percent).

Totals are aggregated from the per-sample CSVs (`Correct Intersections_*.csv`)
when those carry the count columns, so `--skip-blank` can drop un-annotated
samples (gt == 0) and recompute. Older result dirs without per-sample counts
fall back to the precomputed `Correct Intersections_totals_*.csv` files."""
import argparse
import glob
import os

import polars as pl


METRIC = "Correct Intersections"
TOTALS_PREFIX = f"{METRIC}_totals_"
COUNT_COLS = ["matched", "gt", "pred", "unmatched"]


def list_methods(directory: str) -> list[str]:
    """Available method names, taken from the totals CSV filenames."""
    paths = sorted(glob.glob(os.path.join(directory, f"{TOTALS_PREFIX}*.csv")))
    return [os.path.splitext(os.path.basename(p))[0][len(TOTALS_PREFIX):] for p in paths]


def totals_from_samples(sdf: pl.DataFrame, method: str, skip_blank: bool) -> pl.DataFrame:
    """Aggregate a per-sample counts frame to one totals row per resolution.

    With `skip_blank`, samples without GT annotations (gt == 0) are dropped
    before summing. Null counts (samples whose result SVG was missing) are
    ignored by the sums and by the gt > 0 filter.
    """
    if skip_blank:
        sdf = sdf.filter(pl.col("gt") > 0)
    agg = sdf.group_by("resolution").agg(
        pl.col("matched").sum().alias("total_matched"),
        pl.col("gt").sum().alias("total_gt"),
        pl.col("pred").sum().alias("total_pred"),
        pl.col("unmatched").sum().alias("total_unmatched"),
    )
    return agg.with_columns(
        pl.lit(method).alias("method"),
        pl.when(pl.col("total_gt") > 0)
          .then(pl.col("total_matched") / pl.col("total_gt"))
          .otherwise(None).alias("total_rate"),
        pl.when(pl.col("total_pred") > 0)
          .then(pl.col("total_unmatched") / pl.col("total_pred"))
          .otherwise(None).alias("hallucination_rate"),
    )


def load_totals(directory: str, methods: list[str] | None, skip_blank: bool) -> pl.DataFrame:
    """Return one totals row per (method, resolution), in `methods` order.

    Aggregates from per-sample CSVs when they have count columns; otherwise
    falls back to the precomputed totals CSV (which cannot honour --skip-blank).
    """
    available = list_methods(directory)
    methods = methods if methods is not None else available

    frames = []
    for m in methods:
        sample_path = os.path.join(directory, f"{METRIC}_{m}.csv")
        sdf = pl.read_csv(sample_path) if os.path.exists(sample_path) else None
        if sdf is not None and all(c in sdf.columns for c in COUNT_COLS):
            frames.append(totals_from_samples(sdf, m, skip_blank))
            continue
        if skip_blank:
            raise SystemExit(
                f"--skip-blank needs per-sample counts ({', '.join(COUNT_COLS)}) "
                f"for method {m!r} in {directory}, but they are missing. "
                f"Re-run main.py to regenerate the '{METRIC}_*.csv' files."
            )
        totals_path = os.path.join(directory, f"{TOTALS_PREFIX}{m}.csv")
        if not os.path.exists(totals_path):
            raise SystemExit(
                f"No data for method {m!r} in {directory}. Available: {available}"
            )
        frames.append(pl.read_csv(totals_path).with_columns(pl.lit(m).alias("method")))

    # "diagonal" tolerates columns missing from older totals CSVs (total_pred,
    # hallucination_rate) by filling them with nulls instead of failing.
    return pl.concat(frames, how="diagonal")


def pivot_rates(df: pl.DataFrame, values: str = "total_rate") -> pl.DataFrame:
    """Pivot to rows=method, columns=resolution, values=`values`.

    Preserves the method row order of the input (pivot's `maintain_order`).
    """
    wide = df.pivot(
        values=values, index="method", on="resolution",
        maintain_order=True,
    )
    # Sort resolution columns
    res_cols = [c for c in wide.columns if c != "method"]
    res_cols = sorted(res_cols, key=lambda x: int(x))
    return wide.select(["method", *res_cols])


def to_latex(wide: pl.DataFrame) -> str:
    """Render the table as a booktabs LaTeX tabular (siunitx S columns).

    Uses siunitx S columns for decimal-point alignment. Header titles are
    centred via \\multicolumn{1}{c}{...}, which also suppresses the
    per-column @{\\%} glue that would otherwise append a percent sign to
    the header. Best value per column is bold (percent sign stays roman).
    """
    res_cols = [c for c in wide.columns if c != "method"]
    best = {c: wide[c].max() for c in res_cols}

    col_spec = "l" + (" c<{\\%}" * len(res_cols))
    header_cells = ["Method"] + [f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in res_cols]
    header = " & ".join(header_cells) + r" \\"

    body_lines = []
    for row in wide.iter_rows(named=True):
        cells = [row["method"]]
        for c in res_cols:
            v = row[c]
            if v is None:
                cells.append("--")
                continue
            pct = v * 100
            num = f"{pct:.1f}"
            if pct < 10:
                num = f"\\phantom{{0}}{num}"
            cells.append(f"{{\\bfseries {num}}}" if v == best[c] else num)
        body_lines.append(" & ".join(cells) + r" \\")

    return (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        + header + "\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Folder containing the result CSV files")
    ap.add_argument("--reorder", nargs="+", default=None,
                    help="Keep only these methods, in this order.")
    ap.add_argument("--skip-blank", action="store_true", default=False,
                    help="Exclude samples without GT annotations (gt == 0) from "
                         "the totals. Requires per-sample count columns; re-run "
                         "main.py to regenerate them if missing.")
    ap.add_argument("--tex", action="store_true", default=False)
    args = ap.parse_args()
    reorder = [m.replace('*', ' ') for m in args.reorder] if args.reorder else None
    df = load_totals(args.directory, reorder, args.skip_blank)
    wide = pivot_rates(df)

    def print_table(t: pl.DataFrame) -> None:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True,
                       float_precision=4):
            print(t)

    note = " (blank samples excluded)" if args.skip_blank else ""
    print(f"Correct intersection rate{note} (per method, per resolution):")
    print_table(wide)

    if "total_pred" in df.columns and df["total_pred"].null_count() < df.height:
        # print()
        # print("Predicted intersections (per method, per resolution):")
        # print_table(pivot_rates(df, "total_pred"))

        print()
        print(f"Hallucination rate{note} (non-matched / predicted):")
        print_table(pivot_rates(df, "hallucination_rate"))
    else:
        print()
        print("(No total_pred column in the totals CSVs — re-run main.py to "
              "get predicted-intersection counts and hallucination rates.)")

    if args.tex:
        print()
        print(to_latex(wide))


if __name__ == "__main__":
    main()
