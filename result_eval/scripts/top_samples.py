import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description="Print top-k samples by metric score.")
    parser.add_argument("folder", help="Folder containing result CSVs")
    parser.add_argument("metric", help="Metric name")
    parser.add_argument("method", help="Method name")
    parser.add_argument("--highest", action="store_true", help="Show highest scores (default: lowest)")
    parser.add_argument("--topk", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--res", type=int, help="Filter to a specific resolution")
    args = parser.parse_args()

    filename = f"{args.metric}_{args.method}.csv"
    filepath = os.path.join(args.folder, filename)

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    by_res: dict[str, list[tuple[str, float]]] = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                res = row["resolution"]
                score = float(row[args.method])
                by_res.setdefault(res, []).append((row["sample"], score))
            except (KeyError, ValueError):
                continue

    resolutions = sorted(by_res) if args.res is None else [str(args.res)]
    label = "highest" if args.highest else "lowest"

    for res in resolutions:
        if res not in by_res:
            print(f"Resolution {res} not found.")
            continue
        rows = sorted(by_res[res], key=lambda x: x[1], reverse=args.highest)
        print(f"Top {args.topk} {label} '{args.metric}' for '{args.method}' @ {res}px:\n")
        for i, (sample, score) in enumerate(rows[: args.topk], 1):
            print(f"  {i:>3}. {score:.6f}  {sample}")
        print()


if __name__ == "__main__":
    main()
