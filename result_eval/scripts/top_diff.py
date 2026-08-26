import argparse
import csv
import os


def load_scores(folder: str, metric: str, method: str, res: str) -> dict[str, float]:
    filepath = os.path.join(folder, f"{metric}_{method}.csv")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    scores: dict[str, float] = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row["resolution"] != res:
                    continue
                scores[row["sample"]] = float(row[method])
            except (KeyError, ValueError):
                continue
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Print top-k samples where a metric differs most between two methods."
    )
    parser.add_argument("folder", help="Folder containing result CSVs")
    parser.add_argument("metric", help="Metric name")
    parser.add_argument("res", type=int, help="Resolution to filter to")
    parser.add_argument("method_a", help="First method name")
    parser.add_argument("method_b", help="Second method name")
    parser.add_argument("--topk", type=int, default=10, help="Number of pairs to show")
    parser.add_argument(
        "--maxp",
        type=float,
        default=None,
        help="Only consider samples whose first-method score is below this percentile (0-100)",
    )
    parser.add_argument(
        "--minp",
        type=float,
        default=None,
        help="Only consider samples whose first-method score is at or above this percentile (0-100)",
    )
    args = parser.parse_args()

    res = str(args.res)
    try:
        scores_a = load_scores(args.folder, args.metric, args.method_a, res)
        scores_b = load_scores(args.folder, args.metric, args.method_b, res)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    common = sorted(set(scores_a) & set(scores_b))
    if not common:
        print(f"No common samples at {res}px for both methods.")
        return

    def percentile(values: list[float], p: float) -> float:
        # Linear-interpolation percentile (numpy-compatible)
        rank = (p / 100) * (len(values) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (rank - lo) * (values[hi] - values[lo])

    for name, val in (("--maxp", args.maxp), ("--minp", args.minp)):
        if val is not None and not 0 <= val <= 100:
            print(f"{name} must be between 0 and 100.")
            return
    if args.minp is not None and args.maxp is not None and args.minp >= args.maxp:
        print("--minp must be less than --maxp.")
        return

    max_threshold: float | None = None
    min_threshold: float | None = None
    if args.maxp is not None or args.minp is not None:
        values = sorted(scores_a[s] for s in common)
        if args.maxp is not None:
            max_threshold = percentile(values, args.maxp)
            common = [s for s in common if scores_a[s] < max_threshold]
        if args.minp is not None:
            min_threshold = percentile(values, args.minp)
            common = [s for s in common if scores_a[s] >= min_threshold]
        if not common:
            print(f"No samples in the requested percentile range of '{args.method_a}'.")
            return

    diffs = [
        (sample, scores_a[sample], scores_b[sample], abs(scores_a[sample] - scores_b[sample]))
        for sample in common
    ]
    diffs.sort(key=lambda x: x[3], reverse=True)

    header = (
        f"Top {args.topk} largest '{args.metric}' differences "
        f"({args.method_a} vs {args.method_b}) @ {res}px"
    )
    filter_parts: list[str] = []
    if min_threshold is not None:
        filter_parts.append(f"p{args.minp:g} = {min_threshold:.6f} <= {args.method_a}")
    if max_threshold is not None:
        filter_parts.append(f"{args.method_a} < p{args.maxp:g} = {max_threshold:.6f}")
    if filter_parts:
        header += f"\n(filtered to {', '.join(filter_parts)}, {len(common)} samples)"
    print(header + "\n")
    print(f"  {'#':>3}  {'|diff|':>12}  {args.method_a:>14}  {args.method_b:>14}  sample")
    for i, (sample, a, b, d) in enumerate(diffs[: args.topk], 1):
        print(f"  {i:>3}  {d:>12.6f}  {a:>14.6f}  {b:>14.6f}  {sample}")
    print()


if __name__ == "__main__":
    main()
