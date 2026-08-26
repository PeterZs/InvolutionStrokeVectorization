import abc
import os

import numpy as np
import polars as pl
from scipy.spatial import KDTree
import intersections
import svgutils

def total_length(polylines: list[np.ndarray]) -> float:
    total = 0.0
    for pts in polylines:
        diffs = np.diff(pts, axis=0)
        total += float(np.sum(np.linalg.norm(diffs, axis=1)))
    return total


def length_diff(lines_res: list[np.ndarray], lines_gt: list[np.ndarray]) -> float:
    """Total path length of prediction minus total path length of ground truth."""
    a = total_length(lines_res)
    b = total_length(lines_gt)
    #print("a, b", a, "  ", b)
    return abs(a - b)

def chamfer_distance_vec(pts_res: np.ndarray, pts_gt: np.ndarray):

    KDtree_for_res = KDTree(pts_gt)
    KDtree_for_gt = KDTree(pts_res)
    
    # compute chamfer distance from results to gt
    dd_res, _ = KDtree_for_res.query(pts_res, k=1, workers = -1)
    dist_res_to_gt = dd_res.sum()/len(pts_res)

    # compute chamfer distance from gt to results
    dd_res, _ = KDtree_for_gt.query(pts_gt, k=1, workers = -1)
    dist_gt_to_res = dd_res.sum()/len(pts_gt)
    
    return dist_gt_to_res + dist_res_to_gt

def chamfer_distance_polylines(pts_res: list[np.ndarray], pts_gt: list[np.ndarray]) ->float:
    assert len(pts_res)
    assert len(pts_gt)
    return chamfer_distance_vec(np.concat(pts_res), np.concat(pts_gt))


def stroke_number(pts_res: list[np.ndarray], pts_gt: list[np.ndarray]) ->float:
    return len(pts_res)/len(pts_gt)

def stroke_density_ratio(pts_res, pts_gt):

    density_res = len(pts_res) / total_length(pts_res)
    density_gt = len(pts_gt) / total_length(pts_gt)
    
    return density_res / density_gt

def _load_gt(intersections_dir: str, sample_name: str, target_w, target_h) -> np.ndarray | None:
    """Load GT intersection points for `sample_name` from `{dir}/{sample_name}.csv`.

    Returns None if the file is missing. Missing sample warnings are the caller's job.
    """
    path = os.path.join(intersections_dir, sample_name + ".csv")
    if not os.path.exists(path):
        return None
    pts: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split(",")
            pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2))
    #the longest edge in the source is 1024px
    mul = 1024.0/max(target_w, target_h)
    source_w, source_h = target_w*mul, target_h*mul
    
    return svgutils.rescale_points(np.array(pts), source_w, source_h, target_w, target_h)


def _compute_pred_points(lines: list[np.ndarray]) -> np.ndarray:
    """Run the intersection pipeline on polylines, returning unique intersection points."""
    ptsres = intersections.compute_intersections_deduped_fast(lines)
    ptsres = intersections.filter_endpoint_connections(ptsres, lines)
    return intersections.all_ipoints(ptsres)


def _match_gt_indices(
    allpts_res: np.ndarray,
    allpts_gt: np.ndarray,
    d_thresh: float,
) -> np.ndarray:
    """Greedy nearest-neighbour matching of predicted points to GT points.

    Returns indices into `allpts_gt` of distinct points matched within `d_thresh`.
    """
    if len(allpts_gt) == 0 or len(allpts_res) == 0:
        return np.array([], dtype=int)
    tree = KDTree(allpts_gt)
    dd, ii = tree.query(allpts_res, k=1, workers=-1)
    order = np.argsort(dd)

    matched_gt: list[int] = []
    taken: set[int] = set()
    for idx in order:
        if dd[idx] < d_thresh and ii[idx] not in taken:
            taken.add(int(ii[idx]))
            matched_gt.append(int(ii[idx]))
    return np.array(matched_gt, dtype=int)


class MetricState(abc.ABC):
    """Accumulates per-sample state for a metric and knows how to save its
    own CSV outputs once all samples across all resolutions are processed.
    """

    @abc.abstractmethod
    def add_sample(
        self,
        sample_name: str,
        pred_lines: list[np.ndarray],
        gt_lines: list[np.ndarray],
        target_w: float,
        target_h: float,
    ) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def save_csvs(
        cls,
        method: str,
        sample_map: dict[str, str],
        per_resolution: dict[str, "MetricState"],
        output_dir: str,
    ) -> None:
        ...


class CorrectIntersectionRate(MetricState):
    """Tracks correct-intersection-rate stats across samples.

    For each sample the GT intersection points are loaded from a CSV in
    `intersections_dir` named `{sample_name}.csv`. Predicted points are matched
    to GT points by the greedy KDTree assignment in `_match_gt_indices`.

    Per sample, stores (matched, gt_count, pred_count). Since the matching is
    one-to-one, `pred_count - matched` is the number of hallucinated
    (non-matched) predicted intersections.
    """

    METRIC_NAME = "Correct Intersections"

    def __init__(self, intersections_dir: str, d_thresh: float = 1.5):
        self.intersections_dir = intersections_dir
        self.d_thresh = d_thresh
        self._per_sample: dict[str, tuple[int, int, int]] = {}

    def add_sample(self, sample_name, pred_lines, gt_lines, target_w, target_h) -> None:
        gt = _load_gt(self.intersections_dir, sample_name, target_w, target_h)
        if gt is None:
            print(f"    WARNING: missing intersection annotations for '{sample_name}'")
            return
        pred_pts = _compute_pred_points(pred_lines)
        matched = len(_match_gt_indices(pred_pts, gt, self.d_thresh))
        self._per_sample[sample_name] = (matched, len(gt), len(pred_pts))

    def matched(self, sample_name: str) -> int | None:
        entry = self._per_sample.get(sample_name)
        return entry[0] if entry is not None else None

    def gt_count(self, sample_name: str) -> int | None:
        entry = self._per_sample.get(sample_name)
        return entry[1] if entry is not None else None

    def pred_count(self, sample_name: str) -> int | None:
        entry = self._per_sample.get(sample_name)
        return entry[2] if entry is not None else None

    def unmatched(self, sample_name: str) -> int | None:
        """Predicted intersections on this sample not matched to any GT point."""
        entry = self._per_sample.get(sample_name)
        return entry[2] - entry[0] if entry is not None else None

    def rate(self, sample_name: str) -> float | None:
        entry = self._per_sample.get(sample_name)
        if entry is None:
            return float("nan")
        m, g, _ = entry
        if g == 0:
            return 1.0
        return m / g

    def total_matched(self) -> int:
        return sum(m for m, _, _ in self._per_sample.values())

    def total_gt(self) -> int:
        return sum(g for _, g, _ in self._per_sample.values())

    def total_pred(self) -> int:
        return sum(p for _, _, p in self._per_sample.values())

    def total_unmatched(self) -> int:
        return self.total_pred() - self.total_matched()

    def total_rate(self) -> float:
        tg = self.total_gt()
        return self.total_matched() / tg if tg > 0 else float("nan")

    def hallucination_rate(self) -> float:
        """Fraction of predicted intersections not matched to any GT point."""
        tp = self.total_pred()
        return (tp - self.total_matched()) / tp if tp > 0 else float("nan")

    @classmethod
    def save_csvs(cls, method, sample_map, per_resolution, output_dir) -> None:
        # Per sample we record the rate (column named after the method, kept for
        # backward compatibility) plus the raw counts. The counts let downstream
        # tooling (make_intersections_table.py --skip-blank) re-aggregate while
        # excluding un-annotated samples (gt == 0). Counts are null for samples
        # whose result SVG was missing.
        rows = []
        for resolution, metric in per_resolution.items():
            for sample_name in sample_map:
                rows.append({
                    "sample": sample_name,
                    "resolution": resolution,
                    method: metric.rate(sample_name),
                    "matched": metric.matched(sample_name),
                    "gt": metric.gt_count(sample_name),
                    "pred": metric.pred_count(sample_name),
                    "unmatched": metric.unmatched(sample_name),
                })
        path = os.path.join(output_dir, f"{cls.METRIC_NAME}_{method}.csv")
        pl.DataFrame(rows).write_csv(path)
        print(f"Saved {path}")

        total_rows = []
        for resolution, metric in per_resolution.items():
            total_rows.append({
                "method": method,
                "resolution": resolution,
                "total_matched": metric.total_matched(),
                "total_gt": metric.total_gt(),
                "total_rate": metric.total_rate(),
                "total_pred": metric.total_pred(),
                "total_unmatched": metric.total_unmatched(),
                "hallucination_rate": metric.hallucination_rate(),
            })
        path = os.path.join(output_dir, f"{cls.METRIC_NAME}_totals_{method}.csv")
        pl.DataFrame(total_rows).write_csv(path)
        print(f"Saved {path}")


def total_turning(polyline: np.ndarray) -> float:
    """Total signed turning distance of a 2D polyline (Nx2 array)."""
    d = np.diff(polyline, axis=0)          # segment vectors, shape (N-1, 2)
    # cross and dot between consecutive segments
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot   = d[:-1, 0] * d[1:, 0] + d[:-1, 1] * d[1:, 1]
    angles = np.arctan2(cross, dot)        # signed turning angles
    return float(np.sum(angles))


class TurningAngleHistogramDistance(MetricState):
    """Tracks per-sample histograms of per-polyline total turning angles and
    computes a TVD-style distance between prediction and ground-truth histograms.

    Binning: regular bins of width pi/4 covering [-4*pi, 4*pi] (32 bins), plus
    two overflow bins for values < -4*pi and > 4*pi, giving 34 bins total.
    Bin 0 is the <-4*pi overflow, bin 33 is the >4*pi overflow.

    Distance (unnormalised TVD):
        d(P, Q) = sum_i |Q_i - P_i| / (sum(Q) + sum(P))
    where P is the GT histogram and Q is the prediction histogram.
    """

    METRIC_NAME = "Turning Angle Histogram Distance"

    BIN_STEP = np.pi / 4
    BIN_LO = -4 * np.pi
    BIN_HI = 4 * np.pi
    N_REGULAR_BINS = 32  # (BIN_HI - BIN_LO) / BIN_STEP
    N_BINS = N_REGULAR_BINS + 2  # + two overflow bins

    def __init__(self):
        # Per-sample histograms: {sample_name: (pred_hist, gt_hist)}
        self._per_sample: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def histogram(cls, polylines: list[np.ndarray]) -> np.ndarray:
        """Compute the 34-bin histogram of total turning angles for `polylines`."""
        hist = np.zeros(cls.N_BINS, dtype=np.int64)
        for pl in polylines:
            if len(pl) < 3:
                # Need at least 3 points to have any turning angle.
                continue
            v = total_turning(pl)
            if v < cls.BIN_LO:
                hist[0] += 1
            elif v >= cls.BIN_HI:
                hist[-1] += 1
            else:
                # Regular bins occupy indices 1 .. N_REGULAR_BINS.
                idx = int((v - cls.BIN_LO) / cls.BIN_STEP) + 1
                hist[idx] += 1
        return hist

    def add_sample(self, sample_name, pred_lines, gt_lines, target_w, target_h) -> None:
        """Compute and store (prediction, ground-truth) histograms for a sample."""
        self._per_sample[sample_name] = (
            self.histogram(pred_lines),
            self.histogram(gt_lines),
        )

    def pred_hist(self, sample_name: str) -> np.ndarray | None:
        entry = self._per_sample.get(sample_name)
        return entry[0] if entry is not None else None

    def gt_hist(self, sample_name: str) -> np.ndarray | None:
        entry = self._per_sample.get(sample_name)
        return entry[1] if entry is not None else None

    def distance(self, sample_name: str) -> float:
        """TVD-style distance between prediction and GT histograms for a sample."""
        entry = self._per_sample.get(sample_name)
        if entry is None:
            return float("nan")
        q, p = entry
        denom = q.sum() + p.sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(q - p).sum() / denom)

    def total_pred_hist(self) -> np.ndarray:
        if not self._per_sample:
            return np.zeros(self.N_BINS, dtype=np.int64)
        return np.sum([q for q, _ in self._per_sample.values()], axis=0)

    def total_gt_hist(self) -> np.ndarray:
        if not self._per_sample:
            return np.zeros(self.N_BINS, dtype=np.int64)
        return np.sum([p for _, p in self._per_sample.values()], axis=0)

    def total_distance(self) -> float:
        """Distance computed from histograms aggregated across all samples."""
        q = self.total_pred_hist()
        p = self.total_gt_hist()
        denom = q.sum() + p.sum()
        if denom == 0:
            return float("nan")
        return float(np.abs(q - p).sum() / denom)

    @classmethod
    def _bin_edges(cls) -> list[tuple[float, float]]:
        """Half-open ranges [lo, hi) for each bin. Overflow bins use +/- inf."""
        edges = [(-np.inf, cls.BIN_LO)]
        for i in range(cls.N_REGULAR_BINS):
            edges.append((cls.BIN_LO + i * cls.BIN_STEP,
                          cls.BIN_LO + (i + 1) * cls.BIN_STEP))
        edges.append((cls.BIN_HI, np.inf))
        return edges

    @classmethod
    def save_csvs(cls, method, sample_map, per_resolution, output_dir) -> None:
        # Per-sample distance CSV.
        rows = []
        for resolution, metric in per_resolution.items():
            for sample_name in sample_map:
                rows.append({
                    "sample": sample_name,
                    "resolution": resolution,
                    method: metric.distance(sample_name),
                })
        path = os.path.join(output_dir, f"{cls.METRIC_NAME}_{method}.csv")
        pl.DataFrame(rows).write_csv(path)
        print(f"Saved {path}")

        # Total distribution CSV. GT counts are method-agnostic; the file gets
        # overwritten by later runs for other methods, which is fine.
        edges = cls._bin_edges()
        total_rows = []
        for resolution, metric in per_resolution.items():
            pred = metric.total_pred_hist()
            gt = metric.total_gt_hist()
            for i, (lo, hi) in enumerate(edges):
                total_rows.append({
                    "resolution": resolution,
                    "bin_index": i,
                    "bin_low": lo,
                    "bin_high": hi,
                    "pred_count": int(pred[i]),
                    "gt_count": int(gt[i]),
                })
        path = os.path.join(output_dir, f"{cls.METRIC_NAME}_totals_{method}.csv")
        pl.DataFrame(total_rows).write_csv(path)
        print(f"Saved {path}")


def compute_metrics_dict(pred: list[np.ndarray], gt: list[np.ndarray]) -> dict[str, float]:
    return {name: fn(pred, gt) for name, fn in METRICS.items()}



METRICS = {
    "Length diff": length_diff,
    "Chamfer Distance": chamfer_distance_polylines,
    "Stroke Density ratio": stroke_density_ratio,
}

