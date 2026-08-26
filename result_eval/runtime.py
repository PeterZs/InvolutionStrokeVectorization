"""Compute per-resolution mean and stddev of runtime for a given method."""

import argparse
import json
import os
import sys

import polars as pl



def _table_format(values: list[float], decimals: int = 2) -> str:
    """Compute siunitx table-format string, e.g. '2.2' for values up to 99.xx."""
    int_digits = max(len(str(int(abs(v)))) for v in values)
    return f"{int_digits}.{decimals}"


def print_tex_table(df: pl.DataFrame, decimals: int = 2) -> None:
    """Print the runtime stats DataFrame as a LaTeX booktabs table (siunitx S columns)."""
    rows = list(df.iter_rows(named=True))
    fmt = f".{decimals}f"

    # Compute table-format for each S column from data.
    mean_fmt = _table_format([r["Mean"] for r in rows], decimals)
    std_fmt = _table_format([r["stddev"] for r in rows], decimals)
    col_keys = ["median", "p95", "p90", "min", "max"]
    col_fmts = {k: _table_format([r[k] for r in rows], decimals) for k in col_keys}

    # Build tabular spec.
    s_cols = "\n".join(
        f"  S[table-format={col_fmts[k]}]" for k in col_keys
    )
    print(f"\n\\begin{{tabular}}{{")
    print(f"  c")
    print(f"  S[table-format={mean_fmt}]@{{\\, {{\\footnotesize$\\pm$}}\\,}}")
    print(f"  r")
    print(s_cols)
    print(f"}}")
    print(r"\toprule")
    print(r"Resolution & \multicolumn{2}{c}{Mean[s]} & {Median[s]} & {p95[s]} & {p90[s]} & {Min[s]} & {Max[s]} \\")
    print(r"\midrule")

    for row in rows:
        cells = [
            str(row["Input Resolution"]),
            f"{row['Mean']:{fmt}}",
            f"{{\\footnotesize {row['stddev']:{fmt}}}}",
            f"{row['median']:{fmt}}",
            f"{row['p95']:{fmt}}",
            f"{row['p90']:{fmt}}",
            f"{row['min']:{fmt}}",
            f"{row['max']:{fmt}}",
        ]
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate runtime statistics per resolution.")
    parser.add_argument("datapaths", help="Path to datapaths.json")
    parser.add_argument("method", help="Method name (must exist in datapaths.json results)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: result_data/runtime_{method}.csv)")
    parser.add_argument("--top-k", type=int, default=0, help="Print the top K slowest samples per resolution")
    parser.add_argument("--tex", action="store_true", help="Print the table as LaTeX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.datapaths, "r", encoding="utf-8") as f:
        config = json.load(f)

    results = config["results"]
    if args.method not in results:
        print(f"Error: method '{args.method}' not found in datapaths. Available: {list(results.keys())}")
        sys.exit(1)

    resolutions = results[args.method]
    rows: list[dict[str, object]] = []
    per_res_dfs: list[tuple[str, pl.DataFrame]] = []

    for resolution, dir_path in sorted(resolutions.items(), key=lambda x: int(x[0])):
        csv_path = os.path.join(dir_path, "runtime.csv")
        if not os.path.exists(csv_path):
            print(f"Error: runtime.csv not found at {csv_path}")
            sys.exit(1)

        df = pl.read_csv(csv_path, has_header=False, new_columns=["filename", "time"])
        per_res_dfs.append((resolution, df))
        mean = df["time"].mean()
        std = df["time"].std()
        mn = df["time"].min()
        mx = df["time"].max()
        med = df["time"].median()
        p95 = df["time"].quantile(0.95)
        p90 = df["time"].quantile(0.90)
        rows.append({
            "Input Resolution": int(resolution),
            "Mean": round(mean, 2),
            "stddev": round(std, 2),
            "median": round(med, 2),
            "p95": round(p95, 2),
            "p90": round(p90, 2),
            "min": round(mn, 2),
            "max": round(mx, 2)})

    out = pl.DataFrame(rows)
    print(out)

    if args.tex:
        print_tex_table(out, decimals=1)

    if args.top_k > 0:
        for resolution, df in per_res_dfs:
            top = df.sort("time", descending=True).head(args.top_k)
            print(f"\nTop {args.top_k} slowest at {resolution}:")
            for row in top.iter_rows(named=True):
                print(f"  {row['time']:8.2f}s  {row['filename']}")

    output_path = args.output or os.path.join("result_data", f"runtime_{args.method}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.write_csv(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
