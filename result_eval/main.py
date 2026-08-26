import argparse
import json
import os

import numpy as np
import polars as pl

from svgutils import normalize_svg_topath_rect, getpoints, getpoints_maxdist, get_svg_dimensions, points_to_path, get_svg_string_paths
from metrics import (
    compute_metrics_dict, METRICS,
    MetricState, CorrectIntersectionRate, TurningAngleHistogramDistance,
)

RESULT_SUFFIXES_TO_TRY = ["", "_final", "_final_no_smoothing", "_result_linenoise0"]


def load_sample_names(input_dir: str) -> dict[str, str]:
    """Returns {sample_name: svg_path} for all SVGs in input_dir."""
    names = {}
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(".svg"):
            name = os.path.splitext(fname)[0]
            names[name] = os.path.join(input_dir, fname)
    return names


def load_svg_to_str(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_polylines(svg_str: str, target_w: float, target_h: float, n: int) -> list[np.ndarray]:
    paths = normalize_svg_topath_rect(svg_str, target_w, target_h)
    return [getpoints_maxdist(p, n) for p in paths]


# Returns (metric_values_per_sample, missing_count)
# metric_values_per_sample: {sample_name: {metric_name: value}}
# metric_states have add_sample called for each present sample; their own
# outputs are saved separately via each class's save_csvs method.
def get_metrics_on_method_results(
    results_input_dir: str,
    sample_map: dict[str, str],
    n: int,
    metric_states: list[MetricState],
) -> tuple[dict[str, dict[str, float]], int]:
    results: dict[str, dict[str, float]] = {}
    missing = 0

    for idx, name in enumerate(sample_map):
        print(f"{idx}/{len(sample_map)}", end=" ")
        print(f"    {name}")
        svg_path = next(
            (os.path.join(results_input_dir, name + suffix + ".svg")
             for suffix in RESULT_SUFFIXES_TO_TRY
             if os.path.exists(os.path.join(results_input_dir, name + suffix + ".svg"))),
            None,
        )

        if svg_path is None:
            print(f"    MISSING: {name}")
            missing += 1
            continue

        gt_svg = load_svg_to_str(sample_map[name])
        pred_svg = load_svg_to_str(svg_path)
        w, h = get_svg_dimensions(gt_svg)

        gt = load_polylines(gt_svg, w, h, n)
        pred = load_polylines(pred_svg, w, h, n)
        print("lines loaded")

        results[name] = compute_metrics_dict(pred, gt)

        for state in metric_states:
            state.add_sample(name, pred, gt, w, h)

    return results, missing


def metric_dataframe(
    sample_map: dict[str, str],
    # {resolution: {method_name: {sample_name: value}}}
    all_results: dict[str, dict[str, dict[str, dict[str, float]]]],
    metric_name: str,
) -> pl.DataFrame:
    rows = []
    for resolution, method_samples in all_results.items():
        methods = list(method_samples.keys())
        for sample_name in sample_map:
            row: dict = {"sample": sample_name, "resolution": resolution}
            for method in methods:
                row[method] = method_samples[method].get(sample_name, {}).get(metric_name)
            rows.append(row)
    return pl.DataFrame(rows)


def success_rate_dataframe(
    missing_data_counts: dict[str, dict[str, int]],
    results_config: dict[str, dict[str, str]],
    total: int,
) -> pl.DataFrame:
    rows: dict[str, list] = {"method": [], "resolution": [], "success_rate": []}
    for method, resolutions in results_config.items():
        for resolution in resolutions:
            missing = missing_data_counts.get(resolution, {}).get(method, 0)
            success = total - missing
            rows["method"].append(method)
            rows["resolution"].append(resolution)
            rows["success_rate"].append(f"{success}/{total}")
    return pl.DataFrame(rows)




def run_method(
    method: str,
    resolutions: dict[str, str],
    sample_map: dict[str, str],
    n: int,
    state_factories: list,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int], list[dict[str, MetricState]]]:
    """
    Evaluate one method across all its resolutions.

    Args:
        state_factories: zero-arg callables that each construct a fresh MetricState.

    Returns:
        results      – {resolution: {sample: {metric: value}}}
        missing_data – {resolution: missing_count}
        per_state    – list (parallel to state_factories) of {resolution: MetricState}
    """
    results:      dict[str, dict[str, dict[str, float]]] = {}
    missing_data: dict[str, int]                         = {}
    per_state:    list[dict[str, MetricState]]           = [{} for _ in state_factories]

    for resolution, result_dir in resolutions.items():
        print(f"  Resolution: {resolution}")
        assert os.path.exists(result_dir), result_dir
        states = [make() for make in state_factories]
        samples, missing = get_metrics_on_method_results(result_dir, sample_map, n, states)
        results[resolution]      = samples
        missing_data[resolution] = missing
        for i, s in enumerate(states):
            per_state[i][resolution] = s
        total = len(sample_map)
        print(f"  Success: {total - missing}/{total}")

    return results, missing_data, per_state


def save_metric_csvs(
    method: str,
    sample_map: dict[str, str],
    method_results: dict[str, dict[str, dict[str, float]]],
    output_dir: str,
) -> None:
    """Save one CSV per metric, named {metric}_{method}.csv."""
    # Repack into the shape metric_dataframe expects: {resolution: {method: {sample: {metric: value}}}}
    all_results = {res: {method: samples} for res, samples in method_results.items()}
    for metric_name in METRICS:
        df   = metric_dataframe(sample_map, all_results, metric_name)
        path = os.path.join(output_dir, f"{metric_name}_{method}.csv")
        df.write_csv(path)
        print(f"Saved {path}")


def save_success_rate_csv(
    method: str,
    resolutions: dict[str, str],
    missing_data: dict[str, int],
    total: int,
    output_dir: str,
) -> None:
    """Save success_rate_{method}.csv."""
    # success_rate_dataframe expects {method: {resolution: missing_count}}
    missing_nested = {method: missing_data}
    results_config = {method: resolutions}
    df   = success_rate_dataframe(missing_nested, results_config, total)
    path = os.path.join(output_dir, f"success_rate_{method}.csv")
    df.write_csv(path)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vectorization results.")
    parser.add_argument("datapaths", help="Path to the datapaths JSON file")
    parser.add_argument("method", help="Name of the method to evaluate (must be a key in config['results'])")
    parser.add_argument("--n", type=int, default=80,
                        help="Number of sample points per path segment for getpoints()")
    parser.add_argument("--output", default=".", help="Directory to save CSV files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N ground truth samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.datapaths, "r") as f:
        config = json.load(f)

    if args.method not in config["results"]:
        available = list(config["results"].keys())
        raise SystemExit(f"Unknown method '{args.method}'. Available: {available}")

    all_samples = load_sample_names(config["input_names"])
    sample_map  = dict(sorted(all_samples.items())[:args.limit])
    print(f"Loaded {len(sample_map)} sample names.")

    os.makedirs(args.output, exist_ok=True)

    print(f"\n=== {args.method} ===")
    resolutions       = config["results"][args.method]
    intersections_dir = config.get("intersections")

    state_factories: list = []
    if intersections_dir:
        state_factories.append(lambda: CorrectIntersectionRate(intersections_dir))
    state_factories.append(lambda: TurningAngleHistogramDistance())

    results, missing_data, per_state = run_method(
        args.method, resolutions, sample_map, args.n, state_factories,
    )

    save_metric_csvs(args.method, sample_map, results, args.output)
    save_success_rate_csv(args.method, resolutions, missing_data, len(sample_map), args.output)

    for states in per_state:
        if not states:
            continue
        state_cls = type(next(iter(states.values())))
        state_cls.save_csvs(args.method, sample_map, states, args.output)


if __name__ == "__main__":
    main()
