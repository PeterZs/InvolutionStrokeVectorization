import argparse
import functools
import glob
import os
from typing import Any, NamedTuple, Optional

import numpy as np
import polars as pl
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

WHISKER_ALPHA = 1

def load_metric_csv(path: str) -> tuple[list[str], list[str], pl.DataFrame]:
    """Return (methods, resolutions, df) from a single metric CSV."""
    df = pl.read_csv(path)
    methods     = [c for c in df.columns if c not in ("sample", "resolution")]
    resolutions = sorted(df["resolution"].unique().to_list())
    return methods, resolutions, df


def find_and_load_metric_csvs(
    csv_dir: str,
    metric: str,
) -> tuple[list[str], list[str], pl.DataFrame]:
    """
    Discover all files matching ``{metric}_*.csv`` in *csv_dir*, load each one,
    and join them on (sample, resolution) into a single wide DataFrame.

    Each per-method CSV has columns: sample, resolution, <method>.
    The result has columns: sample, resolution, <method1>, <method2>, …

    Returns (methods, resolutions, df).
    """
    pattern = os.path.join(csv_dir, f"{metric}_*.csv")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found matching '{pattern}'")

    frames = [pl.read_csv(p) for p in paths]

    # Join all frames on the key columns; "full" outer-join preserves every
    # (sample, resolution) pair even if a method has partial coverage.
    df = functools.reduce(
        lambda a, b: a.join(b, on=["sample", "resolution"], how="full", coalesce=True),
        frames,
    )

    methods     = [c for c in df.columns if c not in ("sample", "resolution")]
    resolutions = sorted(df["resolution"].unique().to_list())
    return methods, resolutions, df


def box_stats(values: np.ndarray, iqr: bool = False) -> dict[str, Any]:
    """Box plot stats.

    By default, whiskers span the data extrema (min/max). If `iqr` is True,
    whiskers follow the Tukey 1.5×IQR fence convention, clamped to the
    observed data range.
    """
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    lo = float(np.min(values))
    hi = float(np.max(values))
    if iqr:
        fence = 1.5 * (q3 - q1)
        lo_candidates = values[values >= q1 - fence]
        hi_candidates = values[values <= q3 + fence]
        whislo = float(lo_candidates.min()) if lo_candidates.size else q1
        whishi = float(hi_candidates.max()) if hi_candidates.size else q3
    else:
        whislo, whishi = lo, hi
    return {
        "med":    float(np.median(values)),
        "q1":     q1,
        "q3":     q3,
        "whislo": whislo,
        "whishi": whishi,
        "fliers": [],
    }


def build_plot_data(
    df: pl.DataFrame,
    methods: list[str],
    resolutions: list[str],
    box_gap: float,
    resolution_gap: float,
    iqr: bool = False,
) -> tuple[list[float], list[dict[str, Any]], list[Any], list[str], list[tuple[float, str]], list[tuple[int, int]]]:
    """
    Compute per-box positions, stats, colors, tick labels, and group labels.

    Boxes within a group are spaced `box_gap` apart (centre to centre).
    An extra `resolution_gap` is inserted between resolution groups.

    Returns:
        positions       – x position for each box
        stats_list      – box_stats dict for each box
        box_colors      – colour for each box
        tick_labels     – x-tick label for each box (method name)
        group_labels    – list of (center_x, resolution_name) for group annotations
        group_slices    – list of (start_idx, end_idx) index ranges into positions per group
    """
    cmap = plt.get_cmap("tab10")
    method_color: dict[str, Any] = {m: cmap(i % 10) for i, m in enumerate(methods)}

    positions:    list[float]              = []
    stats_list:   list[dict[str, Any]]     = []
    box_colors:   list[Any]                = []
    tick_labels:  list[str]                = []
    group_labels: list[tuple[float, str]]  = []
    group_slices: list[tuple[int, int]]    = []

    x = 0.0
    for res in resolutions:
        res_df = df.filter(pl.col("resolution") == res)
        group_start = x
        start_idx = len(positions)
        for method in methods:
            vals = res_df[method].drop_nulls().to_numpy()
            if vals.size == 0:
                continue
            positions.append(x)
            stats_list.append(box_stats(vals, iqr=iqr))
            box_colors.append(method_color[method])
            tick_labels.append(method)
            x += box_gap
        end_idx = len(positions)
        group_labels.append(((group_start + x - box_gap) / 2.0, str(res)))
        group_slices.append((start_idx, end_idx))
        x += resolution_gap

    return positions, stats_list, box_colors, tick_labels, group_labels, group_slices






def draw_boxes(
    ax: matplotlib.axes.Axes,
    stats_list: list[dict[str, Any]],
    positions: list[float],
    box_width: float,
    box_colors: list[Any],
) -> None:
    bp = ax.bxp(
        stats_list,
        positions=positions,
        widths=box_width,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(edgecolor='none'),
    )
    # brokenaxes' __getattr__ forwards bxp to every sub-axes and returns a list
    # of dicts; a plain Axes returns a single dict.
    bps = bp if isinstance(bp, list) else [bp]
    for bp in bps:
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
        # Whiskers and caps share the box colour at reduced opacity.
        # bxp produces 2 whiskers and 2 caps per box, interleaved in order.
        for i, color in enumerate(box_colors):
            for line in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1],
                         bp["caps"][2 * i],     bp["caps"][2 * i + 1]):
                line.set_color(color)
                line.set_alpha(WHISKER_ALPHA)


def draw_resolution_annotations(
    ax: matplotlib.axes.Axes,
    group_labels: list[tuple[float, str]],
    positions: list[float],
    group_slices: list[tuple[int, int]],
    label_y: float = -1.0,
) -> None:
    """
    Draw resolution group labels inside the plot, just below a horizontal baseline at y=0.

    Layout in data coordinates:
        boxes (y > 0)  →  baseline at y=0  →  resolution labels at y=label_y

    The y-axis is extended below 0 to fit the labels but yticks are kept at >= 0.

    Args:
        group_slices: (start_idx, end_idx) index ranges into *positions* per group.
        label_y:  Y position (data units) of the resolution labels.  Should be
                  negative and sized relative to the data range (default: -2.0).
    """
    # Extend ylim downward to make room for labels below the baseline.
    _, top = ax.get_ylim()
    bottom = label_y
    ax.set_ylim(bottom=bottom)

    # Baseline at y=0
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    for i, (cx, res_name) in enumerate(group_labels):
        # Solid vertical divider running from the bottom up to y=0 only
        if i > 0:
            prev_end = group_slices[i - 1][1]  # end index of previous group
            cur_start = group_slices[i][0]      # start index of current group
            if prev_end > 0 and cur_start < len(positions):
                sep_x = (positions[prev_end - 1] + positions[cur_start]) / 2.0
                ax.vlines(sep_x, ymin=bottom, ymax=0,
                          colors="black", linewidth=0.8, alpha=0.5)

        # Resolution label in data coordinates, centred below the baseline
        ax.text(
            cx, label_y*0.5,
            str(res_name),
            ha="center", va="center",
            fontsize=11, color="black",
        )


def draw_clipped_whisker_labels(
    ax: matplotlib.axes.Axes,
    stats_list: list[dict[str, Any]],
    positions: list[float],
    box_colors: list[Any],
    y_max: float,
) -> None:
    """Annotate boxes whose upper whisker exceeds *y_max* with the actual value."""
    for stats, pos, color in zip(stats_list, positions, box_colors):
        whishi = stats["whishi"]
        if whishi > y_max:
            ax.text(
                pos, y_max, f" {whishi:.1f}",
                ha="center", va="top",
                fontsize=8, color=color, fontweight="bold",
                rotation=30,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )


def draw_legend(
    ax: matplotlib.axes.Axes,
    methods: list[str],
    method_color: dict[str, Any],
) -> None:
    handles = [
        mpatches.Patch(facecolor=method_color[m], alpha=0.75, label=m)
        for m in methods
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Box plot (min/max whiskers) for a metric, grouped by method × resolution."
    )
    parser.add_argument("metric", help="Metric name prefix (e.g. 'chamfer_distance'); "
                                       "all matching {metric}_*.csv files in --csv-dir are joined")
    parser.add_argument("csv_dir", help="Directory to search for CSV files (default: .)")
    parser.add_argument("--output", default=None, help="Save plot to this path instead of showing it")
    parser.add_argument("--resolution-gap", type=float, default=0.8,
                        help="Extra horizontal gap between resolution groups (default: 0.8)")
    parser.add_argument("--box-width", type=float, default=0.3,
                        help="Width of each box (default: 0.3)")
    parser.add_argument("--box-gap", type=float, default=0.45,
                        help="Centre-to-centre spacing between boxes within a group (default: 0.4)")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Cap the y-axis at this value (default: no cap)")
    parser.add_argument("--reorder", nargs="*", default=None,
                        help="Method names in desired display order; only these are plotted. "
                             "If omitted, all methods are shown in discovery order.")
    parser.add_argument("--rename", nargs=2, action="append", default=None,
                        metavar=("FROM", "TO"),
                        help="Relabel method FROM as TO on the plot (display only). "
                             "Repeatable. Applied after --reorder, so FROM is the "
                             "original method name. '*' stands for a space, as in --reorder.")
    parser.add_argument("--hline", type=float, default=None,
                        help="add a grey horizontal line")
    parser.add_argument("--iqr", action="store_true",
                        help="Use Tukey 1.5×IQR fences for whiskers "
                             "(default: min/max extrema)")
    parser.add_argument("--rescale_y", type=float, default=1.0,
                        help="Vertically scale the axes box. 1.0 = unchanged, "
                             "<1 shortens the plot area (y-axis crammed down), "
                             ">1 stretches it. Fonts/margins untouched.")
    parser.add_argument("--break-y", nargs=4, type=float, default=None,
                        metavar=("BOT_LO", "BOT_HI", "TOP_LO", "TOP_HI"),
                        help="Break the y-axis into two ranges (bottom + top). "
                             "e.g. --break-y 0 3 10 13. Incompatible with --ymax.")
    parser.add_argument("--break-ratio", type=float, default=3.0,
                        help="When --break-y is set, height ratio bottom:top. "
                             "e.g. 3.0 (default) = bottom is 3× taller than top, "
                             "compressing the upper half.")
    return parser.parse_args()


class PlotData(NamedTuple):
    """The per-box layout computed by build_plot_data (see its docstring)."""
    positions:    list[float]
    stats_list:   list[dict[str, Any]]
    box_colors:   list[Any]
    tick_labels:  list[str]
    group_labels: list[tuple[float, str]]
    group_slices: list[tuple[int, int]]


def apply_reorder(methods: list[str], reorder: Optional[list[str]]) -> list[str]:
    """Restrict *methods* to the names in *reorder*, in that order.

    '*' in a reorder entry stands for a space (lets method names with spaces
    pass through the shell unquoted).
    """
    if not reorder:
        return methods
    reorder = [m.replace('*', ' ') for m in reorder]
    unknown = [m for m in reorder if m not in methods]
    if unknown:
        print(f"Warning: methods not found in data: {unknown}")
    return [m for m in reorder if m in methods]


def apply_rename(labels: list[str], renames: Optional[list[list[str]]]) -> list[str]:
    """Relabel methods for display only; *labels* are the per-box tick labels.

    Each (FROM, TO) pair renames method FROM to TO. Runs after --reorder, so
    FROM is the original method name. '*' stands for a space, as in --reorder.
    """
    if not renames:
        return labels
    mapping = {a.replace('*', ' '): b.replace('*', ' ') for a, b in renames}
    unknown = [a for a in mapping if a not in labels]
    if unknown:
        print(f"Warning: methods to rename not found in data: {unknown}")
    return [mapping.get(label, label) for label in labels]


def compute_major_yticks(y_top: float) -> list[float]:
    """Nicely rounded major y-tick positions covering [0, y_top]."""
    locator = ticker.MaxNLocator(nbins=6, min_n_ticks=3)
    return [t for t in locator.tick_values(0, y_top) if t >= 0]


def annotate_resolutions(
    ax: matplotlib.axes.Axes,
    data: PlotData,
    major_ticks: list[float],
) -> None:
    """Add minor y-ticks and the resolution group labels below the baseline.

    The labels sit 1.5× the minor tick step below y=0. The minor tick
    positions are read before draw_resolution_annotations extends ylim
    downward (which would shift them), then pinned with a FixedLocator.
    """
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    minor_locs = ax.yaxis.get_minor_locator()()
    if len(minor_locs) >= 2:
        minor_step = minor_locs[1] - minor_locs[0]
    elif len(major_ticks) >= 2:
        minor_step = major_ticks[1] - major_ticks[0]
    else:
        minor_step = 1.0
    draw_resolution_annotations(
        ax, data.group_labels, data.positions, data.group_slices,
        label_y=-1.5 * minor_step,
    )
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_locs))


def plot_broken_y(fig: plt.Figure, args: argparse.Namespace, data: PlotData) -> Any:
    """Draw onto a two-range broken y-axis; returns the brokenaxes object."""
    from brokenaxes import brokenaxes
    bot_lo, bot_hi, top_lo, top_hi = args.break_y
    # brokenaxes takes ranges bottom-first in the tuple; the higher range
    # is stacked on top of the figure. `despine=True` (default) is kept
    # so the outer spines don't get duplicate diagonal break markers.
    # height_ratios in brokenaxes follow gridspec row order (top-first),
    # i.e. reversed from ylims (which is bottom-first). So passing
    # [1, break_ratio] makes the bottom row break_ratio× taller than top.
    ax = brokenaxes(
        ylims=((bot_lo, bot_hi), (top_lo, top_hi)),
        hspace=0.075, fig=fig,
        height_ratios=[1, args.break_ratio],
    )
    bottom_ax, top_ax = ax.axs[1], ax.axs[0]

    # --- Forwarded drawing first. brokenaxes routes these through
    # subax_call, which calls standardize_ticks() at the end, clobbering
    # any manual tick setup. So any manual tick work must come AFTER.
    draw_boxes(ax, data.stats_list, data.positions, args.box_width, data.box_colors)

    # --- Direct sub-axes work: does NOT go through subax_call, so ticks
    # set here are safe.
    ax.set_ylabel(args.metric, fontsize=12, labelpad=30)
    for a in ax.axs:
        a.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        a.tick_params(axis="y", labelsize=11)

    if args.hline is not None:
        # Direct per-sub-axes call bypasses subax_call.
        for a in ax.axs:
            a.axhline(args.hline, color="grey", linewidth=0.8, zorder=2)

    bottom_ax.set_xticks(data.positions)
    bottom_ax.set_xticklabels(data.tick_labels, rotation=30, ha="right", fontsize=11)

    # Force `top_lo` to be an explicit tick on the top sub-axes. brokenaxes'
    # MultipleLocator often picks a round number strictly greater than
    # `top_lo` (e.g. 20 when top_lo=11), leaving the bottom of the top
    # range unlabeled.
    top_ticks = sorted(
        {top_lo, *(t for t in top_ax.get_yticks() if top_lo < t <= top_hi)}
    )
    top_ax.set_yticks(top_ticks)

    # y-ticks + resolution annotations on the bottom sub-axes, same logic
    # as the single-axis path (using bot_hi as the top of the lower range).
    ticks = compute_major_yticks(bot_hi)
    bottom_ax.set_yticks(ticks)
    annotate_resolutions(bottom_ax, data, ticks)
    return ax


def plot_single_y(fig: plt.Figure, args: argparse.Namespace, data: PlotData) -> matplotlib.axes.Axes:
    """Draw onto a single y-axis, optionally capped at --ymax."""
    ax = fig.add_subplot(1, 1, 1)

    draw_boxes(ax, data.stats_list, data.positions, args.box_width, data.box_colors)

    ax.set_xticks(data.positions)
    ax.set_xticklabels(data.tick_labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(args.metric, fontsize=12)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)
        draw_clipped_whisker_labels(ax, data.stats_list, data.positions, data.box_colors, args.ymax)

    _, top = ax.get_ylim()
    ticks = compute_major_yticks(top)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=11)
    ax.set_ylim(0, top)
    annotate_resolutions(ax, data, ticks)

    if args.hline is not None:
        ax.axhline(args.hline, color="grey", linewidth=0.8, zorder=2)
    return ax


def finalize_broken_axes(ax: Any, rescale_y: float) -> None:
    """Post-tight_layout fixups for the broken-axes figure."""
    # Shrink the whole broken-axes block vertically (both sub-axes + their
    # hspace) by scaling each sub-axes' position towards the baseline of
    # the bottom sub-axes, so x-tick labels and resolution annotations stay
    # put. Scaling a single sub-axes would open a gap between them.
    if rescale_y != 1.0:
        anchor = ax.axs[1].get_position().y0
        # Include big_ax (the invisible wrapper that hosts the y-label) so
        # that bbox_inches="tight" crops to the new height instead of the
        # original one.
        for a in [*ax.axs, ax.big_ax]:
            p = a.get_position()
            a.set_position([p.x0, anchor + rescale_y * (p.y0 - anchor),
                            p.width, rescale_y * p.height])

    # brokenaxes draws diagonal break markers in its __init__ using axes
    # positions from BEFORE tight_layout (and before our rescale).
    # Remove the stale handles and redraw them at the final positions.
    for handle in ax.diag_handles:
        handle.remove()
    ax.draw_diags()


def pin_box_aspect(fig: plt.Figure, ax: matplotlib.axes.Axes, rescale_y: float) -> None:
    """Non-break path: pin the axes box aspect ratio. Measured after
    tight_layout so rescale_y=1.0 is a no-op (box_aspect = current h/w).
    bbox_inches="tight" then crops the whitespace that opens above/below.
    """
    pos = ax.get_position()
    w_in = pos.width * fig.get_figwidth()
    h_in = pos.height * fig.get_figheight()
    ax.set_box_aspect(rescale_y * h_in / w_in)


def main() -> None:
    args = parse_args()
    
    if args.break_y is not None and args.ymax is not None:
        raise SystemExit("--break-y and --ymax are mutually exclusive")

    methods, resolutions, df = find_and_load_metric_csvs(args.csv_dir, args.metric)
    methods = apply_reorder(methods, args.reorder)

    data = PlotData(*build_plot_data(
        df, methods, resolutions, args.box_gap, args.resolution_gap, iqr=args.iqr
    ))
    if not data.positions:
        print("No data to plot.")
        return

    # Display-only relabeling of the x-tick (method) labels, after reorder.
    data = data._replace(tick_labels=apply_rename(data.tick_labels, args.rename))


    fig_w = max(6.0, len(data.positions) * args.box_gap + len(resolutions) * args.resolution_gap)
    fig = plt.figure(figsize=(fig_w, 6))

    if args.break_y is not None:
        ax = plot_broken_y(fig, args, data)
    else:
        ax = plot_single_y(fig, args, data)

    plt.tight_layout()

    # Layout fixups that must run after tight_layout has settled positions.
    if args.break_y is not None:
        finalize_broken_axes(ax, args.rescale_y)
    elif args.rescale_y != 1.0:
        pin_box_aspect(fig, ax, args.rescale_y)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
