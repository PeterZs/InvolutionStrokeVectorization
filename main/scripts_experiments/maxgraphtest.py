import cv2
import numpy as np
import networkx as nx
from networkx.utils import UnionFind
from scipy.ndimage import minimum_filter, maximum_filter, binary_dilation
import plotutils.plotutils as plu

path = "debug_files/hat/hat_centerline.png"
path = "debug_files/balloon/balloon_centerline.png"
# path = "debug_files/star/star_centerline.png"
# path = "debug_files/star/star_centerline2.png"

OFFSETS_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def _canon(a, b):
    return (a, b) if a < b else (b, a)


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
    fg_mask = (kernel == 1)
    bg_mask = (kernel == -1)
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


def find_cross_centers(img):
    """Pixel positions (i, j) where every 4-cross-arm value is strictly greater
    than img[i, j]. The center itself may be 0."""
    H, W = img.shape
    centers = set()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            v = int(img[i, j])
            arms = (int(img[i - 1, j]), int(img[i + 1, j]),
                    int(img[i, j - 1]), int(img[i, j + 1]))
            if min(arms) > v:
                centers.add((i, j))
    return centers


def find_disallowed_diagonals(cross_centers):
    """Diagonal pixel-pair edges to disallow around each cross center.
    Returns canonical ((i1, j1), (i2, j2)) tuples between adjacent cross arms."""
    disallowed = set()
    for (i, j) in cross_centers:
        n, s, w, e = (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
        for u, v in [(n, w), (n, e), (s, w), (s, e)]:
            disallowed.add(_canon(u, v))
    return disallowed


def build_max_neighbor_graph(img, cross_centers, disallowed):
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
        if (i, j) in cross_centers:
            continue
        node_id[(i, j)] = counter
        G.add_node(counter, pos=(j + 0.5, i + 0.5), pixel=(i, j),
                   value=int(img[i, j]))
        counter += 1

    for (i, j), nid in node_id.items():
        candidates = []
        for di, dj in OFFSETS_8:
            ni, nj = i + di, j + dj
            if not (0 <= ni < H and 0 <= nj < W) or (ni, nj) not in node_id:
                continue
            if _canon((i, j), (ni, nj)) in disallowed:
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


def _parallel_pixel_pairs(p1, p2):
    """Pixel pairs that would form a parallel edge to (p1, p2): shifted by
    +/-1 perpendicular to the edge direction (vertical -> +/-1 x, horizontal
    -> +/-1 y, diagonal -> both)."""
    i1, j1 = p1
    i2, j2 = p2
    di, dj = i2 - i1, j2 - j1
    if dj == 0:
        offsets = [(0, -1), (0, 1)]
    elif di == 0:
        offsets = [(-1, 0), (1, 0)]
    else:
        offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for oi, oj in offsets:
        yield (i1 + oi, j1 + oj), (i2 + oi, j2 + oj)


def _has_parallel_edge(G, node_id, pa, pb):
    for q1, q2 in _parallel_pixel_pairs(pa, pb):
        n1 = node_id.get(q1)
        n2 = node_id.get(q2)
        if n1 is not None and n2 is not None and G.has_edge(n1, n2):
            return True
    return False


def _count_busy_pixel_neighbors(G, node_id, pixel):
    """Count 8-pixel-neighbors of `pixel` that are nodes with degree >= 2."""
    i, j = pixel
    count = 0
    for di, dj in OFFSETS_8:
        nbr = node_id.get((i + di, j + dj))
        if nbr is not None and G.degree(nbr) >= 2:
            count += 1
    return count


def _count_multidegree_pixel_neighbors(G, node_id, pixel):
    """Number of 8-pixel-neighbors of `pixel` that are nodes with degree >= 2."""
    i, j = pixel
    count = 0
    for di, dj in OFFSETS_8:
        nid = node_id.get((i + di, j + dj))
        if nid is not None and G.degree[nid] >= 2:
            count += 1
    return count


def bridge_dangling_endpoints(G, node_id, disallowed):
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
            # if _canon((i, j), other_pix) in disallowed:
            #     continue
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
        pa = G.nodes[a]["pixel"]
        pb = G.nodes[b]["pixel"]
        if not axa:
            continue
        # if _has_parallel_edge(G, node_id, pa, pb):
        #     continue
        # if _count_busy_pixel_neighbors(G, node_id, pa) > 1:
        #     continue
        # if _count_busy_pixel_neighbors(G, node_id, pb) > 1:
        #     continue
        # G.add_edge(a, b)

        if uf[a] != uf[b]:
            G.add_edge(a, b)
            uf.union(a, b)

        #elif max(G.nodes[a]["value"], G.nodes[b]["value"]) >15:
        else:
            path = nx.shortest_path(G, a, b)
            if len(path) >15:
                G.add_edge(a, b)



img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cross_centers = find_cross_centers(img)
disallowed = find_disallowed_diagonals(cross_centers)
disallowed = set()
G, node_id = build_max_neighbor_graph(img, cross_centers, disallowed)
print("plotting")
with plu.ImageOverlay("maxgraphtest_overlay1.svg", cv2_img=img) as ax:
    plu.plot_graph(ax, G, color="red")
    # plu.plot_graph(ax, H, color="white")
bridge_dangling_endpoints(G, node_id, disallowed)

H = nx.Graph()
for pa, pb in disallowed:
    a = node_id.get(pa)
    b = node_id.get(pb)
    if a is None or b is None:
        continue
    H.add_node(a, pos=G.nodes[a]["pos"])
    H.add_node(b, pos=G.nodes[b]["pos"])
    H.add_edge(a, b)
degree1 = [n for n, d in G.degree() if d == 1]
G.remove_nodes_from(degree1)

print("plotting")
with plu.ImageOverlay("maxgraphtest_overlay.svg", cv2_img=img) as ax:
    plu.plot_graph(ax, G, color="red")
    # plu.plot_graph(ax, H, color="white")
