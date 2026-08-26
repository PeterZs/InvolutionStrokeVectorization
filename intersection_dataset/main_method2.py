import matplotlib.pyplot as plt
import matplotlib
import sampler.line_sampler as line_sampler
import plotutils
import numpy as np
import cv2
import sampler.intersections as intersections
import rasterizer
import random
import argparse
import os
import sys
from PIL import Image
from pprint import pprint
import sampler.stars as stars
from pathlib import Path
import utils

matplotlib.rcParams['path.simplify'] = False
TOOL_PATH = "../main"



def clip_polyline(line: np.ndarray, topleft: tuple, bottomright: tuple) -> np.ndarray:
    """
    Clip a polyline to a rectangle, returning the longest visible segment.

    Args:
        line: Array of shape (N, 2) with (x, y) coordinates.
        topleft: (x_min, y_min) of the clipping rectangle.
        bottomright: (x_max, y_max) of the clipping rectangle.

    Returns:
        Array of shape (M, 2) with the longest clipped segment, or empty (0,2) array.
    """
    if len(line) < 2:
        return line.copy() if len(line) == 1 and _inside(line[0], topleft, bottomright) else np.empty((0, 2))

    xmin, ymin = topleft
    xmax, ymax = bottomright

    # Build all clipped segments by walking each edge and clipping with Cohen-Sutherland
    segments = []  # list of list of points
    current_seg = []

    for i in range(len(line) - 1):
        p0 = line[i].copy()
        p1 = line[i + 1].copy()
        clipped = _cohen_sutherland_clip(p0, p1, xmin, ymin, xmax, ymax)

        if clipped is not None:
            cp0, cp1 = clipped
            if current_seg:
                # Check continuity: if the start of this clipped edge matches the end of current segment
                if np.allclose(current_seg[-1], cp0):
                    current_seg.append(cp1)
                else:
                    segments.append(current_seg)
                    current_seg = [cp0, cp1]
            else:
                current_seg = [cp0, cp1]
        else:
            # Edge fully outside — break the current segment
            if current_seg:
                segments.append(current_seg)
                current_seg = []

    if current_seg:
        segments.append(current_seg)

    if not segments:
        return np.empty((0, 2))

    # Pick the longest segment by polyline length
    best = max(segments, key=_polyline_length)
    return np.array(best)


def _polyline_length(pts):
    total = 0.0
    for i in range(len(pts) - 1):
        total += np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i]))
    return total


def _inside(pt, topleft, bottomright):
    return topleft[0] <= pt[0] <= bottomright[0] and topleft[1] <= pt[1] <= bottomright[1]


def _outcode(x, y, xmin, ymin, xmax, ymax):
    code = 0
    if x < xmin:   code |= 1  # LEFT
    elif x > xmax: code |= 2  # RIGHT
    if y < ymin:   code |= 4  # BOTTOM
    elif y > ymax: code |= 8  # TOP
    return code


def _cohen_sutherland_clip(p0, p1, xmin, ymin, xmax, ymax):
    """Clip a single line segment. Returns (clipped_p0, clipped_p1) or None."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    code0 = _outcode(x0, y0, xmin, ymin, xmax, ymax)
    code1 = _outcode(x1, y1, xmin, ymin, xmax, ymax)

    while True:
        if not (code0 | code1):
            return (np.array([x0, y0]), np.array([x1, y1]))
        if code0 & code1:
            return None

        code_out = code0 if code0 else code1
        dx = x1 - x0
        dy = y1 - y0

        if code_out & 8:    # TOP
            x = x0 + dx * (ymax - y0) / dy if dy else x0
            y = ymax
        elif code_out & 4:  # BOTTOM
            x = x0 + dx * (ymin - y0) / dy if dy else x0
            y = ymin
        elif code_out & 2:  # RIGHT
            y = y0 + dy * (xmax - x0) / dx if dx else y0
            x = xmax
        else:               # LEFT
            y = y0 + dy * (xmin - x0) / dx if dx else y0
            x = xmin

        if code_out == code0:
            x0, y0 = x, y
            code0 = _outcode(x0, y0, xmin, ymin, xmax, ymax)
        else:
            x1, y1 = x, y
            code1 = _outcode(x1, y1, xmin, ymin, xmax, ymax)



def geodesic_midpoint(polyline: np.ndarray) -> np.ndarray:
    """Find the point at half the total arc length along the polyline."""
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    half = cum_lengths[-1] / 2.0

    idx = np.searchsorted(cum_lengths, half) - 1
    idx = np.clip(idx, 0, len(seg_lengths) - 1)

    remainder = half - cum_lengths[idx]
    t = remainder / seg_lengths[idx] if seg_lengths[idx] > 0 else 0.0
    return polyline[idx] + t * diffs[idx]

def match_polylines(source: dict[int, np.ndarray], targets: list[np.ndarray]) -> dict[int, np.ndarray]:
    """Match target polylines to source polylines by closest geodesic midpoints (greedy).

    Args:
        source: dict mapping id -> polyline (N x 2/3 array of points)
        targets: list of polylines (each M x 2/3 array of points)

    Returns:
        dict mapping each source id -> matched target polyline
    """
    assert len(source) == len(targets)

    # Precompute midpoints
    source_ids = list(source.keys())
    source_mids = {sid: geodesic_midpoint(source[sid]) for sid in source_ids}
    target_mids = [geodesic_midpoint(t) for t in targets]

    # Greedy matching: pick closest pair, remove both, repeat
    used_targets: set[int] = set()
    result: dict[int, np.ndarray] = {}

    # Build all pairwise distances
    pairs = []
    for sid in source_ids:
        for ti, tmid in enumerate(target_mids):
            dist = np.linalg.norm(source_mids[sid] - tmid)
            pairs.append((dist, sid, ti))
    pairs.sort()

    matched_sources: set[int] = set()
    for dist, sid, ti in pairs:
        if sid in matched_sources or ti in used_targets:
            continue
        result[sid] = targets[ti]
        matched_sources.add(sid)
        used_targets.add(ti)
        if len(result) == len(source):
            break

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Generate geometric samples.")
    
    parser.add_argument('--out_dir', type=str, default="samples",
                        help='Base output directory.')
    
    parser.add_argument('--target_dim', type=int, default=512,
                        help='Target dimension for the image/canvas.')
    
    parser.add_argument('--from_seed', dest='start_seed', type=int, required=True,
                        help='Starting seed (inclusive).')
    
    parser.add_argument('--to_seed', dest='end_seed', type=int, required=True,
                        help='Ending seed (inclusive).')
    
    parser.add_argument('--n_add', type=int, nargs=2, default= [0, 0],
                        help='Min and Max (inclusive) for number of extra random lines to add')
    
    parser.add_argument('--center_perturb', type=int, default=25,
                        help='Min and Max (inclusive) for number of extra random lines to add')
    
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

        
    if args.start_seed > args.end_seed:
        print(f"Error: from_seed ({args.start_seed}) cannot be greater than to_inclusive ({args.end_seed})")
        sys.exit(1)

    return args

def ensure_directories(base_dir):
    subdirs = ["vector", "img", "segmentation", "mat", "half_edges_vis", "half_edges_pairs", "lines_json"]
    for sd in subdirs:
        os.makedirs(os.path.join(base_dir, sd), exist_ok=True)



def main():
    args = parse_args()
    
    ensure_directories(args.out_dir)
    
    DIM = args.target_dim
    MIN_DIST = 5
    
    stats_dir = os.path.join(args.out_dir, "stats.csv")
    if os.path.exists(stats_dir): os.unlink(stats_dir)
    

    for seed_idx in range(args.start_seed, args.end_seed + 1):
        attempt = 0
        valid_sample_found = False
        random.seed(seed_idx)
        np.random.seed(seed_idx)
        name = f"{seed_idx}"
        print(f"Processing Seed ID: {seed_idx}...")
        
        n_add = random.randint(args.n_add[0], args.n_add[1])
        while True: 
            polylines = line_sampler.sample_star_method(DIM, args.center_perturb, MIN_DIST, n_add)
            polylines = [clip_polyline(pl, (0, 0), (DIM, DIM)) for pl in polylines]
            print("num polylines", len(polylines))
            path = os.path.join(args.out_dir, "vector", f"{name}.svg")
            if args.debug:
                with plotutils.ImageOverlay(path, dims=(DIM, DIM)) as ax:
                    plotutils.plot_lines_and_points(ax, lines=polylines,
                                                    color=plotutils.COLORS, lw=0.5, 
                                    linealpha=1, pointcolor="grey")  
                #plotutils.plot_lines_and_points(ax, points =all_ipoints, pointcolor="black", psize=1)
            

            res = intersections.compute_intersections_deduped(polylines)
            all_ipoints = intersections.all_ipoints(res)
            #assert len(all_ipoints) == 1
            half_edges, pairs, nxt = rasterizer.polyline_half_edges_full(polylines, res)
            print("got num", len(half_edges))
            M = rasterizer.get_matrix_M(half_edges, nxt)
            #print(nxt)
            #print(M)
            path = os.path.join(args.out_dir, "vector", f"{name}_vis.svg")
            if args.debug:
                plotutils.plot_half_edges_lines(DIM, M, half_edges, all_ipoints, path=path, annotate=True)
            
            
            img_dr = np.zeros((DIM, DIM, 3), dtype=np.uint8)
            for pl in polylines:
                rasterizer.draw_polyline_cv2(img_dr, pl, (255, 255, 255), aa=True)
            def command(inp: str):
                return ["uv", "run", "--directory", TOOL_PATH,  "main.py", str(inp), "--skip-preprocess", "--noinvert", "--save_lines"]
            def imagesave(saveto: str):
                cv2.imwrite(saveto, img_dr)
            print("running cli..")
            try:
                data = utils.run_cli_tool(imagesave, command, file_ext=".png")
            except:
                continue
            lines_perturbed = utils.load_lines(data)
            path = os.path.join(args.out_dir, "vector", f"{name}_perturbed.svg")
            if args.debug:
                with plotutils.ImageOverlay(path, dims=(DIM, DIM)) as ax:
                    plotutils.plot_lines_and_points(ax, lines=lines_perturbed,
                                                    color=plotutils.COLORS, lw=0.5, 
                                    linealpha=1, pointcolor="grey")  
                
            print("number of lines obtained", len(lines_perturbed))
            half_edges_perturbed = rasterizer.half_edges_from_sublines(lines_perturbed)
            if len(half_edges_perturbed) == len(half_edges):
                break
        
        
        half_edges_matched = match_polylines(half_edges, half_edges_perturbed)
        
        path = os.path.join(args.out_dir, "vector", f"{name}_matched_vis.svg")
        if args.debug:
            plotutils.plot_half_edges_lines(DIM, M, half_edges_matched, all_ipoints, path=path, annotate=True)
        
        #save pair info
        pair_path = os.path.join(args.out_dir, "half_edges_pairs", f"{name}.csv")
        utils.dict_to_kv_csv(pairs, pair_path)
            
        # save half-edge polyline data
        linepath= os.path.join(args.out_dir, "lines_json", f"{name}.json")
        utils.save_polylines_to_json(half_edges_matched, linepath)
        
        #save adjacency matrix
        mat_path = os.path.join(args.out_dir, "mat", f"{name}.csv")
        np.savetxt(mat_path, M, delimiter=",", fmt="%d")

if __name__ == "__main__":
    main()