import cv2
import numpy as np
import networkx as nx
from pathlib import Path
from networkx.utils import UnionFind
from scipy.ndimage import minimum_filter, maximum_filter, binary_dilation
from scipy.spatial import ConvexHull, QhullError
from matplotlib.collections import LineCollection
import plotutils.plotutils as plu
import graph_extraction.image as gimg
import graph_extraction.planargraph_core as planargraph_core
import graph_extraction.shapeutils as shapeutils


OFFSETS_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def _canon(a, b):
    return (a, b) if a < b else (b, a)

def remove_short_branches(G, k):
    """Remove branches of length <= k. A branch is the path from a degree-1
    leaf to the nearest degree>=3 junction, walking along degree-2 internal
    nodes. The leaf and the (at most k-1) internal nodes are removed; the
    junction is kept. Standalone paths/cycles (no junction) are untouched.
    """
    to_remove = set()
    leaves = [n for n, d in G.degree() if d == 1]
    for leaf in leaves:
        walked = [leaf]
        prev = None
        cur = leaf
        for _ in range(k):
            neighbors = [n for n in G.neighbors(cur) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            d_nxt = G.degree(nxt)
            if d_nxt >= 3:
                to_remove.update(walked)
                break
            if d_nxt == 1:
                # standalone path, not a branch
                break
            walked.append(nxt)
            prev, cur = cur, nxt
    G.remove_nodes_from(to_remove)
    return G

def close_short_loops(G, img, disallowed_nodes, disallowed_edges):
    """For every degree-1 endpoint, search for a 3-edge (= 2 intermediate
    pixels) pixel path through eligible 8-neighbor pixels that reaches
    another degree-1 endpoint. If found, add the 2 intermediate pixels as
    new nodes and wire the 3 edges into G to close the gap.

    An eligible intermediate pixel is nonzero in `img`, not in
    `disallowed_nodes`, and not already a graph node. An eligible edge has
    its canonical pixel pair not in `disallowed_edges`. Returns the number
    of endpoint pairs closed.
    """
    H, W = img.shape

    existing_pixels = set()
    for n in G.nodes:
        ppos = G.nodes[n]["pos"]
        existing_pixels.add((int(np.floor(ppos[1])), int(np.floor(ppos[0]))))

    def is_eligible_pixel(p):
        i, j = p
        if not (0 <= i < H and 0 <= j < W):
            return False
        if img[i, j] == 0:
            return False
        if p in disallowed_nodes:
            return False
        if p in existing_pixels:
            return False
        return True

    def is_edge_allowed(p1, p2):
        return _canon(p1, p2) not in disallowed_edges

    closed = 0
    for start_node in list(G.nodes):
        if start_node not in G or G.degree(start_node) != 1:
            continue
        start_pos = G.nodes[start_node]["pos"]
        start_pixel = (
            int(np.floor(start_pos[1])),
            int(np.floor(start_pos[0])),
        )

        # Map degree-1 endpoint pixels (excluding `start_node`) -> node id.
        endpoint_pixels = {}
        for n, d in G.degree():
            if d != 1 or n == start_node:
                continue
            p = G.nodes[n]["pos"]
            endpoint_pixels[
                (int(np.floor(p[1])), int(np.floor(p[0])))
            ] = n

        found = [None]

        def dfs(path):
            if found[0] is not None:
                return
            if len(path) == 4:
                end_pixel = path[-1]
                if end_pixel in endpoint_pixels:
                    found[0] = (endpoint_pixels[end_pixel], list(path))
                return
            cur = path[-1]
            for di, dj in OFFSETS_8:
                if found[0] is not None:
                    return
                nxt = (cur[0] + di, cur[1] + dj)
                if nxt in path:
                    continue
                if not is_edge_allowed(cur, nxt):
                    continue
                if len(path) < 3:
                    if not is_eligible_pixel(nxt):
                        continue
                else:
                    if nxt not in endpoint_pixels:
                        continue
                dfs(path + [nxt])

        dfs([start_pixel])
        if found[0] is None:
            continue

        end_node, path = found[0]
        next_id = (max(G.nodes) + 1) if G.nodes else 0
        new_ids = []
        for p in path[1:-1]:
            G.add_node(
                next_id,
                pos=(p[1] + 0.5, p[0] + 0.5),
                pixel=p,
                value=int(img[p[0], p[1]]),
            )
            new_ids.append(next_id)
            existing_pixels.add(p)
            next_id += 1
        node_path = [start_node] + new_ids + [end_node]
        for a, b in zip(node_path[:-1], node_path[1:]):
            G.add_edge(a, b)
        closed += 1
    return closed


def contract_close_junction_pairs(G, max_len=1.25):
    """Single-pass contraction of clean adjacent-junction pairs whose edge
    length is at most `max_len`. A 'clean pair' is an edge (u, v) with both
    endpoints degree>=3 and short enough, AND where u and v each have
    exactly one such candidate edge incident — so chains of overlapping
    junctions are left alone rather than greedily merged. Each surviving
    pair is contracted to a single node at the midpoint of u and v's
    positions. Designed to also pick up zero-length 'overlapping' junctions
    produced by the degree-3 collapse pass.
    """
    candidates = []
    counts = {}
    for u, v in G.edges:
        if G.degree(u) < 3 or G.degree(v) < 3:
            continue
        pu = np.array(G.nodes[u]["pos"], dtype=float)
        pv = np.array(G.nodes[v]["pos"], dtype=float)
        if float(np.linalg.norm(pu - pv)) > max_len:
            continue
        candidates.append((u, v))
        counts[u] = counts.get(u, 0) + 1
        counts[v] = counts.get(v, 0) + 1

    contracted = 0
    for u, v in candidates:
        if u not in G or v not in G:
            continue
        if counts.get(u, 0) != 1 or counts.get(v, 0) != 1:
            continue
        pu = np.array(G.nodes[u]["pos"], dtype=float)
        pv = np.array(G.nodes[v]["pos"], dtype=float)
        mid = (pu + pv) / 2.0
        nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
        G.nodes[u]["pos"] = (float(mid[0]), float(mid[1]))
        G.nodes[u].pop("contraction", None)
        contracted += 1
    return contracted


def contract_adjacent_junctions(G):
    """Contract every edge whose both endpoints have degree >= 3 into a single
    node at the midpoint of their positions. Repeats until no such edge
    remains (since merging two junctions can produce new junction-junction
    edges)."""
    while True:
        target = None
        for u, v in G.edges:
            if G.degree(u) >= 3 and G.degree(v) >= 3:
                target = (u, v)
                break
        if target is None:
            break
        u, v = target
        pu = np.array(G.nodes[u]["pos"], dtype=float)
        pv = np.array(G.nodes[v]["pos"], dtype=float)
        mid = ((pu + pv) / 2.0)
        nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
        G.nodes[u]["pos"] = (float(mid[0]), float(mid[1]))
        # nx.contracted_nodes attaches a 'contraction' attr to u; drop it to
        # keep attribute bookkeeping clean.
        G.nodes[u].pop("contraction", None)
    return G


def color_junction_clusters(G, k=5):
    """Find pairs of degree>=3 junctions whose graph distance is <= k. Union
    them via UnionFind so connected pairs form clusters. Each cluster includes
    its junction nodes plus every internal node on the connecting short paths.
    Each cluster is then colored with one entry from plu.COLORS_ON_BLACK
    (cycling), set as G.nodes[u]["color"] so plot_graph picks it up.
    Returns the list of clusters (each as a set of node ids).
    """
    junctions = [n for n, d in G.degree() if d >= 3]
    if not junctions:
        return []
    junction_set = set(junctions)

    uf = UnionFind(junctions)
    short_paths = []
    seen_pair = set()
    for u in junctions:
        sp = nx.single_source_shortest_path(G, u, cutoff=k)
        for v, path in sp.items():
            if v == u or v not in junction_set:
                continue
            pair = (u, v) if u < v else (v, u)
            if pair in seen_pair:
                continue
            seen_pair.add(pair)
            uf.union(u, v)
            short_paths.append(path)

    cluster_nodes = {}
    for path in short_paths:
        root = uf[path[0]]
        cluster_nodes.setdefault(root, set()).update(path)

    palette = plu.COLORS_ON_BLACK
    clusters = list(cluster_nodes.values())
    for i, nodes in enumerate(clusters):
        c = palette[i % len(palette)]
        for n in nodes:
            G.nodes[n]["color"] = c
    return clusters


def smooth_paths(G, d=1.0, alpha=0.5, iterations=20):
    """Laplacian-smooth every maximal degree-2 chain in G under a per-node
    displacement constraint d, updating G.nodes[u]['pos']. Endpoints of the
    chain (junctions / leaves) are not moved."""
    pos_attr = nx.get_node_attributes(G, "pos")
    visited = set()
    for start in list(G.nodes):
        if G.degree(start) == 2:
            continue
        for nbr in G.neighbors(start):
            if G.degree(nbr) != 2 or nbr in visited:
                continue
            path = [start, nbr]
            visited.add(nbr)
            prev, cur = start, nbr
            while G.degree(cur) == 2:
                nxts = [n for n in G.neighbors(cur) if n != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
                if G.degree(cur) == 2:
                    visited.add(cur)
            if len(path) < 3:
                continue
            pts = np.array([pos_attr[n] for n in path], dtype=float)
            smoothed = shapeutils.smooth_polyline_constrained(
                pts, d, alpha=alpha, iterations=iterations
            )
            for n, p in zip(path[1:-1], smoothed[1:-1]):
                G.nodes[n]["pos"] = (float(p[0]), float(p[1]))
                pos_attr[n] = (float(p[0]), float(p[1]))


def get_outbound_info(G, u, v):
    """Walk u -> v(=v1) -> v2 -> v3 -> v4 along the outgoing degree-2 chain
    leaving node u through neighbor v. Always returns
    (v1_node, v2_node, v2_pos, vec) for every actual neighbor — never None
    for a degree-related reason — so that the count matches the cluster
    boundary "degree".

    `vec` is the normalized outgoing direction:
      * v3 exists: v3 - v2, averaged with v3 - v4 when v3 has a single
        continuation v4 of degree <= 2.
      * walk of length 3 (no v3): average of v1 - u and v2 - v1.
      * walk of length 2 (no v2): v1 - u; v2_node and v2_pos collapse to v1
        so that the caller can still wire an edge to that immediate neighbor.
    Only returns None when the resulting direction is the zero vector.
    """
    walk = [u, v]
    prev, cur = u, v
    while len(walk) < 4:
        nxts = [n for n in G.neighbors(cur) if n != prev]
        if len(nxts) != 1:
            break
        prev, cur = cur, nxts[0]
        walk.append(cur)

    u_pos = np.array(G.nodes[walk[0]]["pos"], dtype=float)
    v1_node = walk[1]
    v1 = np.array(G.nodes[walk[1]]["pos"], dtype=float)

    if len(walk) >= 3:
        v2_node = walk[2]
        v2 = np.array(G.nodes[walk[2]]["pos"], dtype=float)
        if len(walk) >= 4:
            v3 = np.array(G.nodes[walk[3]]["pos"], dtype=float)
            d = v3 - v2
            # Smoothing: if v3 has a single continuing neighbor v4 (v3 degree
            # 2) and that v4 itself has degree <= 2, average v2->v3 / v3->v4.
            nxts4 = [n for n in G.neighbors(walk[3]) if n != walk[2]]
            if len(nxts4) == 1 and G.degree(nxts4[0]) <= 2:
                v4 = np.array(G.nodes[nxts4[0]]["pos"], dtype=float)
                d = ((v3 - v2) + (v4 - v3)) / 2.0
        else:
            # no v3: average of u->v1 and v1->v2
            d = ((v1 - u_pos) + (v2 - v1)) / 2.0
    else:
        # length-2 walk: there is no v2 deeper than v1. Treat v1 itself as v2
        # so the caller still has an endpoint to wire an edge to (the v1
        # node will be preserved by collapse_cluster_to_point in this case).
        v2_node = v1_node
        v2 = v1
        d = v1 - u_pos

    n = np.linalg.norm(d)
    if n == 0:
        return None
    return v1_node, v2_node, v2, d / n


def cluster_outbound_vectors(G, cluster):
    """For every cluster boundary node u with an edge leaving the cluster,
    use get_outbound_info to walk the outgoing chain. Returns
    (outer_points, v2_points, vecs):
      * outer_points -- v0 positions, used for angle scoring
      * v2_points    -- v2 positions, used for visibility (these are the
        actual graph endpoints after a collapse-and-replace)
      * vecs         -- normalized outgoing directions
    """
    cluster_set = set(cluster)
    outer_points = []
    v2_points = []
    vecs = []
    for u in cluster:
        for v in G.neighbors(u):
            if v in cluster_set:
                continue
            info = get_outbound_info(G, u, v)
            if info is None:
                continue
            _, _, v2, vec = info
            outer_points.append(np.array(G.nodes[u]["pos"], dtype=float))
            v2_points.append(v2)
            vecs.append(vec)
    if not outer_points:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(outer_points), np.array(v2_points), np.array(vecs)


def cluster_best_collapse_point(G, cluster, img, background_pixels, grid_step=0.5):
    """Reproduce process_general's construction for a cluster:

    * Sample a grid over the AABB of the cluster node positions.
    * Filter to candidates lying on a nonzero pixel (img > 0) AND visible from
      every boundary outer point (no zero pixel on the connecting segment).
    * Pick the candidate with the highest angle-alignment score against the
      cluster's outgoing vectors.

    Returns the chosen (x, y) point or None.
    """
    outer_points, v2_points, vecs = cluster_outbound_vectors(G, cluster)
    if len(outer_points) == 0:
        return None

    pos = nx.get_node_attributes(G, "pos")
    cluster_positions = np.array([pos[u] for u in cluster], dtype=float)
    cluster_positions = np.concat([cluster_positions, v2_points])
    aabb = shapeutils.compute_aabb(cluster_positions)
    candidates = shapeutils.generate_grid_in_aabb(aabb, grid_step)
    if len(candidates) == 0:
        return None

    H, W = img.shape

    def on_nonzero(p):
        ix, iy = int(np.floor(p[0])), int(np.floor(p[1]))
        return 0 <= iy < H and 0 <= ix < W and img[iy, ix] > 0

    def visible_from_all(p):
        # check visibility to v2 points -- those are the actual edge endpoints
        # produced by collapse_cluster_to_point.
        return not any(
            shapeutils.segment_blocked(p, q, background_pixels) for q in v2_points
        )

    filtered = np.array(
        [p for p in candidates if on_nonzero(p) and visible_from_all(p)]
    )
    if len(filtered) == 0:
        return None

    best_point, _ = shapeutils.best_point_angle_optimized(filtered, v2_points, vecs)

    # Reject if the worst (largest) dot product between (best_point - v2)/|...|
    # and the outgoing vector (v3 - v2)/|...| is greater than -0.85: that
    # would mean at least one branch is poorly anti-aligned with the path.
    bp = np.array(best_point, dtype=float)
    worst_dot = -np.inf
    for v2_p, vec in zip(v2_points, vecs):
        d = bp - v2_p
        dn = np.linalg.norm(d)
        if dn == 0:
            continue
        dot = float(np.dot(d / dn, vec))
        if dot > worst_dot:
            worst_dot = dot
    # if worst_dot > -0.85:
    #     return None
    return best_point


def annotate_cluster_v2_dots(G, cluster, point):
    """For every outgoing branch of `cluster`, walk to v2/v3 and attach
    G.nodes[v2]["info"] with the dot product between (point - v2)/|...| and
    (v3 - v2)/|...|. Values close to +1 mean the connection points along the
    outgoing direction (good alignment with the path); close to -1 means it
    points the opposite way."""
    pos = nx.get_node_attributes(G, "pos")
    cluster_set = set(cluster)
    p = np.array(point, dtype=float)
    for u in cluster:
        for v in G.neighbors(u):
            if v in cluster_set:
                continue
            walk = [u, v]
            prev, cur = u, v
            while len(walk) < 4:
                nxts = [n for n in G.neighbors(cur) if n != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                walk.append(cur)
            if len(walk) < 4:
                continue
            v2_id = walk[2]
            v2_p = np.array(pos[v2_id], dtype=float)
            v3_p = np.array(pos[walk[3]], dtype=float)
            outvec = v3_p - v2_p
            on = np.linalg.norm(outvec)
            dirvec = p - v2_p
            dn = np.linalg.norm(dirvec)
            if on == 0 or dn == 0:
                continue
            dot = float(np.dot(dirvec / dn, outvec / on))
            G.nodes[v2_id]["info"] = f"{dot:.2f}"


def add_band_debug(G, cluster, length=5.0):
    """Visualize each cluster outgoing branch as a polyline through v2 along
    the smoothed (v2->v3 averaged with v3->v4) direction. The two endpoints
    are v2 +/- length * vec; the line is added to G via
    planargraph_core.add_path_subgraph so it gets picked up by plot_graph."""
    _, v2_points, vecs = cluster_outbound_vectors(G, cluster)
    for v2, vec in zip(v2_points, vecs):
        p1 = v2 - vec * length
        p2 = v2 + vec * length
        planargraph_core.add_path_subgraph(G, np.array([p1, p2]))


def cluster_best_collapse_point_band(G, cluster, img, background_pixels, d=0.75, grid_step=0.125):
    """Variant of cluster_best_collapse_point: instead of maximizing the
    angle-alignment score, require the candidate's perpendicular distance to
    every (v2, vec)-line to be < d (a band of width 2d around each line).
    Returns the first candidate satisfying that constraint plus the usual
    on-nonzero-pixel and v2-visibility filters, or None if no point qualifies.
    """
    outer_points, v2_points, vecs = cluster_outbound_vectors(G, cluster)
    if len(outer_points) == 0:
        return None

    pos = nx.get_node_attributes(G, "pos")
    cluster_positions = np.array([pos[u] for u in cluster], dtype=float)
    cluster_positions = np.concat([cluster_positions, v2_points])

    aabb = shapeutils.compute_aabb(cluster_positions)
    candidates = shapeutils.generate_grid_in_aabb(aabb, grid_step)
    if len(candidates) == 0:
        return None

    H, W = img.shape

    def on_nonzero(p):
        ix, iy = int(np.floor(p[0])), int(np.floor(p[1]))
        return 0 <= iy < H and 0 <= ix < W and img[iy, ix] > 0

    def visible_from_all(p):
        return not any(
            shapeutils.segment_blocked(p, q, background_pixels) for q in v2_points
        )

    for p in candidates:
        if not on_nonzero(p):
            continue
        if not visible_from_all(p):
            continue
        # planargraph_core.add_path_subgraph(G, np.array([p]))
        diffs = p - v2_points
        # 2D perp distance = |cross(diff, unit_vec)| since vec is normalized.
        cross = diffs[:, 0] * vecs[:, 1] - diffs[:, 1] * vecs[:, 0]
        if np.all(np.abs(cross) < d):
            return p
    return None


def collapse_cluster_to_point(G, cluster, point):
    """Replace `cluster` (plus the immediate v1 stub on every outgoing branch)
    with a single new node at `point`, and connect that node to each v2.

    For every cluster boundary node v0 with an outgoing edge v0->v1 along a
    degree-2 chain, walk one step further to v2 and record (v1, v2). After
    collecting all such pairs, the cluster nodes and the unique v1 nodes are
    removed and the new node is wired to every unique v2 still in G.

    Returns the new node id, or None if no outgoing branches were found.
    """
    cluster_set = set(cluster)
    branches = []
    for u in cluster:
        for v in G.neighbors(u):
            if v in cluster_set:
                continue
            prev, cur = u, v
            walk = [u, v]
            while len(walk) < 3:
                nxts = [n for n in G.neighbors(cur) if n != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                walk.append(cur)
            if len(walk) >= 3:
                branches.append((walk[1], walk[2]))
            else:
                # length-2 walk: no v2 deeper than v1 -- treat v1 itself as
                # the endpoint and don't remove it.
                branches.append((walk[1], walk[1]))

    if not branches:
        return None

    new_id = max(G.nodes) + 1
    G.add_node(new_id, pos=tuple(point))

    G.remove_nodes_from(cluster_set)
    # Only remove v1 stubs that are distinct from their v2 (length >= 3 walks).
    G.remove_nodes_from({v1 for v1, v2 in branches if v1 != v2 and v1 in G})

    for v2 in {v2 for _, v2 in branches}:
        if v2 in G and v2 != new_id:
            G.add_edge(new_id, v2)

    return new_id


def detect_pattern(img, kernel, return_centers=False):
    """Locate all kernel placements where every pixel under a -1 entry is
    strictly less than every pixel under a +1 entry. (0 entries are ignored.)

    Implementation: with a footprint of the +1 cells, ``minimum_filter`` gives
    the min of foreground values centered at each pixel; with the -1 footprint,
    ``maximum_filter`` gives the max of background values. The match condition
    is ``max_bg < min_fg``. Off-image neighbors are forced to fail (cval = -inf
    for the min, +inf for the max), so patterns can't poke past the borders.

    Returns a (H, W) bool mask whose True entries are the union of the -1
    pixel positions across all matching placements. Pass ``return_centers=True``
    to instead receive a bool mask of the matching kernel-center positions.
    """
    img_f = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel)
    fg_mask = kernel == 1
    bg_mask = kernel == -1
    if not fg_mask.any():
        raise ValueError("Kernel must contain at least one foreground (1) entry.")
    if not bg_mask.any() and not return_centers:
        raise ValueError("Kernel must contain at least one background (-1) entry.")

    min_fg = minimum_filter(img_f, footprint=fg_mask, mode="constant", cval=-np.inf)
    if bg_mask.any():
        max_bg = maximum_filter(img_f, footprint=bg_mask, mode="constant", cval=np.inf)
        valid = max_bg < min_fg
    else:
        valid = np.isfinite(min_fg)

    if return_centers:
        return valid
    return binary_dilation(valid, structure=bg_mask)


def detect_pattern_tresh(img, kernel, thresh=15, return_centers=False):
    """Like detect_pattern, but a kernel placement only counts when
    ``min(fg values) - max(bg values) >= thresh`` — i.e. the foreground/
    background gap meets a strict minimum margin (default 15)."""
    img_f = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel)
    fg_mask = kernel == 1
    bg_mask = kernel == -1
    if not fg_mask.any():
        raise ValueError("Kernel must contain at least one foreground (1) entry.")
    if not bg_mask.any() and not return_centers:
        raise ValueError("Kernel must contain at least one background (-1) entry.")

    min_fg = minimum_filter(img_f, footprint=fg_mask, mode="constant", cval=-np.inf)
    if bg_mask.any():
        max_bg = maximum_filter(img_f, footprint=bg_mask, mode="constant", cval=np.inf)
        valid = (min_fg - max_bg) >= thresh
    else:
        valid = np.isfinite(min_fg)

    if return_centers:
        return valid
    return binary_dilation(valid, structure=bg_mask)


def all_symmetries(kernel):
    """All 8 D4 symmetries (rotations + flips) of `kernel`, deduplicated."""
    k = np.asarray(kernel)
    seen = set()
    out = []
    for r in range(4):
        kr = np.rot90(k, r)
        for flip in (False, True):
            kf = np.fliplr(kr) if flip else kr
            key = (kf.shape, kf.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(kf)
    return out


def all_symmetry_pairs(kernel, replace):
    """All 8 D4 symmetries of (kernel, replace) transformed in tandem,
    deduplicated by the (kernel, replace) byte pair so that a kernel with its
    own symmetry can still pair with multiple distinct replace orientations."""
    k = np.asarray(kernel)
    r = np.asarray(replace)
    seen = set()
    out = []
    for rot in range(4):
        kr = np.rot90(k, rot)
        rr = np.rot90(r, rot)
        for flip in (False, True):
            kf = np.fliplr(kr) if flip else kr
            rf = np.fliplr(rr) if flip else rr
            key = (kf.shape, kf.tobytes(), rf.shape, rf.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((kf, rf))
    return out


# Patterns whose -1 pixels mark forbidden node positions.
FORBIDDEN_PATTERNS = [
    # the following 2 are causing problems
    # 4-cross saddle: center darker than every cross arm
    # np.array([
    #     [0,  1, 0],
    #     [1, -1, 1],
    #     [0,  1, 0],
    # ]),
    # zigzag pair: two -1 pixels each sandwiched between three brighter
    # neighbors, offset diagonally by one
    # np.array(
    #     [
    #         [ 1, 1, -1],
    #         [-1,-1, -1],
    #         [-1, 1,  1],
    #     ]
    # ),
    np.array(
        [
            [1, -1, 1, 0],
            [0, 1, -1, 1],
        ]
    ),
    np.array(
        [
            [ 1, 1, 1,  0],
            [ -1,-1,-1,-1],
            [ 0, 1, 1,  1],
        ]
    ),
    np.array(
        [
            [0,  1,  0, 0],
            [1, -1, -1, 0],
            [0, -1, -1, 1],
            [0,  0,  1, 0],
        ]
    ),
    np.array(
        [
            [1, -1,  1, 0],
            [1, -1,  1, 0],
            [1, -1,  1, -1],
            [0,  1,  -1, 1],
        ]
    ),
    np.array(
        [
            [0,  0,  1, -1, -1],
            [0,  1, -1, -1,  1],
            [1, -1, -1,  1,  0],
            [0,  0,  1,  0,  0],
        ]
    ),
    # np.array(
    #     [
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 1],
    #         [0, 0,-1, 1],
    #         [1, 1, 0, 0],
    #     ]
    # ),
    np.array(
        [
            [0, 0, 1, 1],
            [0, 0,-1, 0],
            [1,-1, 0, 0],
            [1, 0, 0, 0],
        ]
    ),
    np.array(
        [
            [1, -1, -1, 1],
            [1, -1, -1, 1],
            [1, -1, -1, 1],
        ]
    ),
    np.array(
        [
            [1,-1, -1,  1, 0, 0],
            [0, 1, -1, -1, 1, 0],
            [0, 0,  1, -1,-1, 1],
        ]
    ),
    np.array(
        [
            [0, 0, -1,  1, 1, 0],
            [1, 1, -1, -1, 1, 1],
            [0, 1,  1,  0, 0, 0],
        ]
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0,-1,-1],
            [0, 1,-1,-1, -1],
            [0, 0,-1,-1, -1],
        ]
    ),
    np.array(
        [
            [0,  0, 0, 0,  0],
            [0,  1, 1, 0,  0],
            [-1,-1,-1, 1,  1],
            [ 1, 1,-1,-1, -1],
            [0,  0, 1, 1,  0],
        ]
    ),
    np.array(
        [
            [ 0,  1,  1,  1,  1],
            [ 1, -1, -1, -1, -1],
            [-1, -1, -1,  1,  1],
            [0,  0,   0,  0,  0],
        ]
    ),
    np.array(
        [
            [1, -1,-1, 0,  0],
            [1, -1,-1, 1,  1],
            [1,  0,-1, 0,  0],
            [0,  1,-1, 0,  0],
        ]
    ),
]


KERNELS = [
     {
        "pattern": np.array(
            [
                [ 0, -1, -1,  0],
                [ -1, 1,  0, -1],
                [-1, 0,  1,  -1],
                [0, -1,  -1,  0],
            ]
        ),
         "bad edges":  [
             np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )]
         
     },
     {
        "pattern": np.array(
            [
                [0,  -1, -1,  0],
                [ -1, 1,  -1, -1],
                [-1, -1,  1,  1],
                [0,  -1,   0,  0],
            ]
        ),
         "bad edges":  [
             np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )]
         
     },
     {
        "pattern": np.array(
            [
                [-1, -1, -1,  0],
                [ 1,  1, -1, -1],
                [-1, -1,  1,  1],
                [0,   0, 0,  0],
            ]
        ),
         "bad edges":  [
             np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )]
         
     },
     {
        "pattern": np.array(
            [
                [ 0, 0, 0, -1,  1],
                [ 0, 0, -1, 1, -1],
                [ 0,-1, 1, -1,  0],
                [-1, 1,-1,  0,  0],
            ]
        ),
         "bad edges":  [
             np.array(
            [
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 1, 0],
                [ 0, 0, 0, 0, 0],
            ]
        ),
            np.array(
            [
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 1, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
            ]
        ),
            np.array(
            [
                [ 0, 0, 0, 1, 0],
                [ 0, 0, 0, 0, 1],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
            ]
        )]
         
     },
    #   {
    #     "pattern": np.array(
    #         [
    #             [-1,   0, -1,  0],
    #             [-1,   1, -1,  1],
    #             [ 1,  -1,  1, -1],
    #             [ 0,  -1,  0, -1],
    #         ]
    #     ),
    #      "bad edges":  [
    #          np.array(
    #         [
    #             [0, 0, 0, 0],
    #             [1, 0, 0, 0],
    #             [0, 1, 0, 0],
    #             [0, 0, 0, 0],
    #         ]
    #     ),
    #          np.array(
    #         [
    #             [0, 0, 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1],
    #             [0, 0, 0, 0],
    #         ]
    #     )]
         
    #  },
      {
        "pattern": np.array(
            [
                [0, -1,  1, -1,  1, -1],
                [0,  0,  1, -1,  1, -1],
                [-1, 1, -1,  1,  0,  0],
                [-1, 1, -1,  1, -1,  0],
            ]
        ),
         "bad edges":  [
             np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
            np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )]
         
     },
    {
        "pattern": np.array(
            [
                [-1, -1,  0,   0, -1, -1],
                [ 1,  1, -1,   0, -1, -1],
                [-1, -1,  0,  -1,  1,  1],
                [-1, -1,  0,   0, -1, -1],
            ]
        ),
        "bad edges": [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ],
    },
    {
        "pattern": np.array(
            [
                [-1, -1,  -1,  -1, -1],
                [ 1,  1,  -1, -1, -1],
                [-1, -1,  -1,  1, 1],
                [-1, -1,  -1, -1, -1],
            ]
        ),
        "bad edges": [
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ]
    },
    {
        "pattern": np.array(
            [
                [0,   0,  0, -1],
                [-1, -1,  -1,  1],
                [-1,  1,   1, -1],
                [1,  -1,  -1, -1],
            ]
        ),
        "bad edges": [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ])
        ]
    },
    {
        "pattern": np.array(
            [
                [-1, -1,   -1,  0],
                [ 1,  1,  -1, -1],
                [-1, -1,   1,  1],
                [ 0, -1,  -1, -1],
            ]
        ),
        "bad edges": [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ])
        ]
    }
]



def find_forbidden_nodes(img):
    """Bool (H, W) mask of pixel positions that fall under a -1 cell of any
    forbidden pattern (or one of its D4 symmetries), then refined by
    ``detect_and_mark`` with all symmetries of (KERNEL, REPLACE_KERNEL) so
    near-misses get their REPLACE_KERNEL footprint folded into the mask."""
    H, W = img.shape
    mask = np.zeros((H, W), dtype=bool)
    for base in FORBIDDEN_PATTERNS:
        for k in all_symmetries(base):
            mask |= detect_pattern_tresh(img, k)

    binary = mask.astype(np.uint8) * 255
    # for k, rk in all_symmetry_pairs(KERNEL, REPLACE_KERNEL):
    #     added, _ = gimg.detect_and_mark(
    #         binary, k.astype(np.int8), rk.astype(np.uint8)
    #     )
    #     binary = np.maximum(binary, added)
    return binary > 0




def find_disallowed_edges_from_kernel(forbidden_mask):
    """For each entry in KERNELS (a dict with ``pattern`` and ``bad edges``),
    iterate every (pattern, bad_edge_arr) D4 symmetry pair: each detection of
    the kernel pattern contributes exactly ONE forbidden edge, between the two
    pixels marked by bad_edge_arr placed at the detection location."""
    binary = forbidden_mask.astype(np.uint8) * 255
    disallowed = set()
    for entry in KERNELS:
        pattern = entry["pattern"]
        for bad_edge_arr in entry["bad edges"]:
            for k, be in all_symmetry_pairs(pattern, bad_edge_arr):
                ones = list(zip(*np.where(be == 1)))
                if len(ones) != 2:
                    continue
                (lr1, lc1), (lr2, lc2) = ones
                anchor_r = k.shape[0] // 2
                anchor_c = k.shape[1] // 2
                detected = cv2.morphologyEx(
                    binary, cv2.MORPH_HITMISS, k.astype(np.int8)
                )
                ii, jj = np.where(detected > 0)
                for i, j in zip(ii.tolist(), jj.tolist()):
                    p1 = (int(i) + int(lr1) - anchor_r, int(j) + int(lc1) - anchor_c)
                    p2 = (int(i) + int(lr2) - anchor_r, int(j) + int(lc2) - anchor_c)
                    disallowed.add(_canon(p1, p2))
    return disallowed


def disallowed_edges_from_forbidden(forbidden_mask):
    """For every 2x2 window where one diagonal pair is fully forbidden, the
    OTHER diagonal of the same 2x2 window is a disallowed edge.
    Returns canonical pixel-pair tuples ((i1, j1), (i2, j2))."""
    tl = forbidden_mask[:-1, :-1]
    tr = forbidden_mask[:-1, 1:]
    bl = forbidden_mask[1:, :-1]
    br = forbidden_mask[1:, 1:]

    disallowed = set()
    # main-diagonal pair (TL, BR) forbidden -> disallow anti-diagonal edge (TR, BL)
    for ii, jj in zip(*np.where(tl & br)):
        a, b = (int(ii), int(jj) + 1), (int(ii) + 1, int(jj))
        disallowed.add(_canon(a, b))
    # anti-diagonal pair (TR, BL) forbidden -> disallow main-diagonal edge (TL, BR)
    for ii, jj in zip(*np.where(tr & bl)):
        a, b = (int(ii), int(jj)), (int(ii) + 1, int(jj) + 1)
        disallowed.add(_canon(a, b))
    return disallowed


def build_max_neighbor_graph(img, disallowed_nodes, disallowed_edges):
    """Spawn a node at every nonzero pixel above threshold (skipping cross
    centers) and connect each to the highest-valued non-disallowed 8-neighbor.
    """
    H, W = img.shape
    G = nx.Graph()

    ys, xs = np.nonzero(img)
    node_id = {}
    counter = 0
    for i, j in zip(ys.tolist(), xs.tolist()):
        if int(img[i, j]) < 15:
            continue
        if (i, j) in disallowed_nodes:
            continue
        node_id[(i, j)] = counter
        G.add_node(counter, pos=(j + 0.5, i + 0.5), pixel=(i, j), value=int(img[i, j]))
        counter += 1

    for (i, j), nid in node_id.items():
        candidates = []
        for di, dj in OFFSETS_8:
            ni, nj = i + di, j + dj
            if not (0 <= ni < H and 0 <= nj < W) or (ni, nj) not in node_id:
                continue
            if _canon((i, j), (ni, nj)) in disallowed_edges:
                continue
            candidates.append((ni, nj, int(img[ni, nj])))
        if not candidates:
            continue
        max_val = max(v for _, _, v in candidates)
        for ni, nj, v in candidates:
            if v == max_val:
                G.add_edge(nid, node_id[(ni, nj)])
                break

    return G, node_id


def bridge_dangling_endpoints(G, node_id, disallowed_edges):
    """Collect every (degree-1 endpoint -> existing 8-neighbor node) candidate
    edge, sort descending by the minimum image value of the two pixels, and
    add them in that order whenever they connect different components.
    """
    uf = UnionFind(G.nodes)
    for u, v in G.edges:
        uf.union(u, v)

    degree1 = [n for n, d in G.degree() if d == 1]

    candidates = []
    seen = set()
    for nid in degree1:
        i, j = G.nodes[nid]["pixel"]
        v_nid = G.nodes[nid]["value"]
        for di, dj in OFFSETS_8:
            other_pix = (i + di, j + dj)
            other = node_id.get(other_pix)
            if other is None or other == nid:
                continue
            if _canon((i, j), other_pix) in disallowed_edges:
                continue
            edge = _canon(nid, other)
            if edge in seen:
                continue
            seen.add(edge)
            axis_aligned = 1 if (di == 0 or dj == 0) else 0
            candidates.append(
                (axis_aligned, min(v_nid, G.nodes[other]["value"]), nid, other)
            )

    candidates.sort(key=lambda c: c[1], reverse=True)

    for axa, _, a, b in candidates:

        if uf[a] != uf[b]:
            G.add_edge(a, b)
            uf.union(a, b)

        else:
            path = nx.shortest_path(G, a, b)
            if len(path) < 8:
                continue
            else:
                poly = np.array([G.nodes[nid]["pos"] for nid in path])
                if shapeutils.get_max_inscribed_circle(poly)[1] > 1.25:
                    G.add_edge(a, b)


def regions(binary: np.ndarray):
    # 3x3 with one corner forbidden, plus its 3 rotations.
    corner_cut = np.array(
        [
            [1, 1, 1, -1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int8,
    )

    #all_ones = np.ones((3, 3), dtype=np.int8)

    kernels = [np.rot90(corner_cut, r) for r in range(4)]

    union = np.zeros_like(binary)
    for k in kernels:
        mask, _ = gimg.detect_and_mark(binary, k)
        union = np.maximum(union, mask)
    return union


def process(img_or_path, debug_dir: str = "", stem: str | None = None):
    """Build a path-graph from a centerline image.

    `img_or_path` may be a filesystem path (str / pathlib.Path) or a
    grayscale numpy image (H, W) — useful when called inline from main.py
    where the centerline is already in memory.
    `stem` overrides the filename stem used in debug output paths;
    defaults to Path(img_or_path).stem for path inputs and "image" otherwise.
    """
    if isinstance(img_or_path, (str, Path)):
        path = img_or_path
        if stem is None:
            stem = Path(path).stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = img_or_path
        if stem is None:
            stem = "image"
    assert img is not None
    binary = (img > 0).astype(np.uint8) * 255
    #cv2.imwrite(f"binary_{stem}.png", binary)
    # adjusted = gimg.adaptive(img, size=7)
    # adjusted = 255-adjusted
    # cv2.imwrite(f"adaptive_{stem}.png", adjusted)
    # imgmed = cv2.medianBlur(img, 5)
    # cv2.imwrite(f"medblur_{stem}.png", imgmed)
    # img2 = cv2.subtract(img, imgmed)
    # img4 = cv2.add(img, imgmed)
    # cv2.imwrite(f"medblursub_{stem}.png", img)
    # cv2.imwrite(f"medbluradd_{stem}.png", img4)
    # # img2 = cv2.subtract(img, adjusted)
    # img2 = img.copy()
    # img2[img2>50] = 0
    # cv2.imwrite(f"noise_{stem}.png", img2)
    
    forbidden_mask = find_forbidden_nodes(img).astype(np.uint8)
    fy, fx = np.nonzero(forbidden_mask)
    disallowed_nodes = set(zip(fy.tolist(), fx.tolist()))
    print(f"[{stem}] forbidden nodes: {len(disallowed_nodes)}")

    binary = cv2.subtract(binary, forbidden_mask * 255)


    disallowed = find_disallowed_edges_from_kernel(forbidden_mask)
    print(f"[{stem}] disallowed edges: {len(disallowed)}")
    G, node_id = build_max_neighbor_graph(img, disallowed_nodes, disallowed)
    disallowed_segments = [
        ((j1 + 0.5, i1 + 0.5), (j2 + 0.5, i2 + 0.5))
        for (i1, j1), (i2, j2) in disallowed
    ]

    bridge_dangling_endpoints(G, node_id, disallowed)

    if debug_dir:
        print(f"[{stem}] plotting")
        with plu.ImageOverlay(f"{debug_dir}/1_{stem}.svg", cv2_img=img) as ax:
            plu.plot_graph(ax, G, color="red")
            ax.scatter(fx + 0.5, fy + 0.5, c="burlywood", s=0.5,
                    edgecolors="none", zorder=2)
            ax.add_collection(
                LineCollection(
                    disallowed_segments, colors="burlywood", linewidths=0.3, zorder=2
                )
            )
        print(f"[{stem}] plotted overlay")

    
    

    remove_short_branches(G, 2)
    # print("closing short loops")
    # n_closed = close_short_loops(G, img, disallowed_nodes, disallowed)
    # print(f"[{stem}] short loops closed: {n_closed}")

    contract_adjacent_junctions(G)
    clusters = color_junction_clusters(G, k=4)
    print(f"[{stem}] junction clusters: {len(clusters)}")
    if debug_dir:
        print(f"[{stem}] plotting")
        with plu.ImageOverlay(f"{debug_dir}/2_{stem}.svg", cv2_img=img) as ax:
            plu.plot_graph(ax, G, color="red")
        print(f"[{stem}] plotted overlay2")


    
    paths = planargraph_core.decompose_into_paths(G)
    pos = nx.get_node_attributes(G, "pos")
    for path in paths:
        if len (paths) <=5:
            continue
        path = path[2: -2]
        coords = np.array([pos[n] for n in path])
        smoothed = shapeutils.smooth_polyline_constrained(coords, d=0.5)
        for node, new_pos in zip(path, smoothed):
            G.nodes[node]["pos"] = tuple(new_pos)

    background_pixels = set(map(tuple, np.argwhere(img == 0).tolist()))
    palette = plu.COLORS_ON_BLACK[1:]
    collapsed = 0
    for i, cluster in enumerate(clusters):
        # add_band_debug(G, cluster)
        p = cluster_best_collapse_point_band(G, cluster, img, background_pixels)
        if p is None:
            continue
        annotate_cluster_v2_dots(G, cluster, p)
        new_id = collapse_cluster_to_point(G, cluster, p)
        if new_id is None:
            continue
        G.nodes[new_id]["color"] = palette[i % len(palette)]
        collapsed += 1
    print(f"[{stem}] clusters collapsed: {collapsed}/{len(clusters)}")
    
    if debug_dir:
        print(f"[{stem}] plotting")
        with plu.ImageOverlay(f"{debug_dir}/3_{stem}.svg", cv2_img=img) as ax:
            plu.plot_graph(ax, G, color="red")
        print(f"[{stem}] plotted overlay3")

    # Collapse every remaining degree-3 node as its own trivial cluster. This
    # also catches degree-3 nodes from clusters that did not collapse.
    deg3 = [n for n, d in G.degree() if d == 3]
    deg3_collapsed = 0
    for u in deg3:
        if u not in G or G.degree(u) != 3:
            continue
        singleton = {u}
        p = cluster_best_collapse_point(G, singleton, img, background_pixels)
        if p is None:
            continue
        if collapse_cluster_to_point(G, singleton, p) is not None:
            deg3_collapsed += 1
    print(f"[{stem}] degree-3 nodes collapsed: {deg3_collapsed}/{len(deg3)}")

    n_pair_contracted = contract_close_junction_pairs(G, max_len=1.25)
    print(f"[{stem}] close junction pairs contracted: {n_pair_contracted}")

    # remove_short_branches(G, 1)
    G.remove_nodes_from(list(nx.isolates(G)))
    if debug_dir:
        print(f"[{stem}] plotting")
        with plu.ImageOverlay(f"{debug_dir}/4_{stem}.svg", cv2_img=img) as ax:
            plu.plot_graph(ax, G, color="red")
        print(f"[{stem}] plotted overlay3")


    total_length = 0.0
    for a, b in G.edges:
        pa = np.array(G.nodes[a]["pos"], dtype=float)
        pb = np.array(G.nodes[b]["pos"], dtype=float)
        total_length += float(np.linalg.norm(pb - pa))
    print(f"[{stem}] total edge length: {total_length:.2f}")

    return G