import numpy as np
import networkx as nx
import transform
import cv2
from skimage.draw import line
import itertools
import utils.utils as utils
import torch
import os
from scipy.ndimage import gaussian_filter1d
import plotutils.plotutils as plu
from scipy.interpolate import interp1d


MAX_OBJECTS = 1000
IMAGE_SIZE = 500

def compute_matrix_for_rescaling(in_w: float, in_h: float, target: float, align: str = "center") -> np.ndarray:
    """
    Compute matrix (a, b, c, d, e, f) for uniform scaling to fit (no stretching)
    into a square of size `target`. `align` can be "center" or "top-left".
    - a = scale_x = scale
    - d = scale_y = scale
    - b = c = 0
    - e = tx, f = ty (translation)
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input width and height must be positive numbers")

    s = min(target / in_w, target / in_h)
    scaled_w = in_w * s
    scaled_h = in_h * s

    if align == "center":
        tx = (target - scaled_w) / 2.0
        ty = (target - scaled_h) / 2.0
    elif align == "top-left":
        tx = 0.0
        ty = 0.0
    else:
        raise ValueError("invalid align")

    # matrix(a b c d e f) with no skew/rotation
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def apply_transform(matrix: np.ndarray, points: np.ndarray):
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack([points, ones])
    transformed = hom @ matrix.T
    return transformed[:, :2]


def parabolic_profile(t):
        return 4 * t * (1 - t)
    
def noise_2d(N, sigma):
    raw_noise = np.random.randn(N, 2)
    smoothed_noise = gaussian_filter1d(raw_noise, sigma=sigma, axis=0)

    smoothed_noise = smoothed_noise / np.std(smoothed_noise)
    return smoothed_noise

def add_profile_noise(polyline: np.ndarray, profile_func = parabolic_profile, max_noise_std: float = 1.0, sigma: float = 5.0) -> np.ndarray:
    """
    Adds noise to an (N, 2) polyline based on a programmable intensity profile.
    
    :param polyline: (N, 2) numpy array representing the polyline.
    :param profile_func: A function that takes an array of floats and returns an array of scales.
    :param max_noise_std: Maximum standard deviation of the Gaussian noise.
    :return: A new (N, 2) numpy array with the applied noise.
    """
    N = polyline.shape[0]
    
    # 1. Parameterize the polyline indices from 0.0 to 1.0
    t = np.linspace(0, 1, N)
    
    # 2. Sample the noise intensity profile
    # Reshape to (N, 1) so it broadcasts correctly against the (N, 2) noise array
    intensity_profile = profile_func(t).reshape(-1, 1)
    
    # 3. Generate raw 2D Gaussian noise (mean=0, std=1)
    noise = noise_2d(N, sigma=sigma)
    
    # 4. Scale the raw noise by our profile and the max noise intensity
    scaled_noise = noise * intensity_profile * max_noise_std
    
    # 5. Add noise to the original polyline
    noisy_polyline = polyline + scaled_noise
    
    # 6. Explicitly preserve endpoints to prevent floating-point precision issues
    noisy_polyline[0] = polyline[0]
    noisy_polyline[-1] = polyline[-1]
    
    return noisy_polyline

def add_gaussian_noise(image, mean=0, std_dev=25):
    image = image.astype(np.float32)
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def densify_polyline_maxdist(polyline: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Returns a densified polyline where every segment has length at most max_dist.
    Segments shorter than max_dist are left unchanged.

    Args:
        polyline: np.ndarray of shape (N, 2)
        max_dist: float, maximum allowed segment length

    Returns:
        np.ndarray of shape (M, 2) where M >= N
    """
    # 1. Handle edge cases
    if len(polyline) < 2:
        return polyline.copy()
    if max_dist <= 0:
        raise ValueError("max_dist must be positive")

    # 2. Calculate lengths of existing segments
    # shape: (N-1, 2)
    diffs = polyline[1:] - polyline[:-1]
    # shape: (N-1,)
    lengths = np.linalg.norm(diffs, axis=1)

    # 3. Determine how many segments each existing segment needs to be split into
    # We use ceil to ensure the new sub-segments are <= max_dist.
    # We enforce at least 1 segment (max with 1) to handle 0-length segments gracefully.
    num_segments = np.ceil(lengths / max_dist).astype(int)
    num_segments = np.maximum(num_segments, 1)

    # 4. Generate the new points
    new_points = []

    for i in range(len(num_segments)):
        p_start = polyline[i]
        p_end = polyline[i+1]
        n = num_segments[i]

        if n == 1:
            # If the segment is already short enough, keep the start point.
            # We treat the list as [start, end), appending the final end point later.
            new_points.append(p_start.reshape(1, 2))
        else:
            # Generate n+1 points (including start and end), then exclude the last one
            # so we don't duplicate points when moving to the next segment.
            # shape: (n, 2)
            segment_points = np.linspace(p_start, p_end, n + 1)[:-1]
            new_points.append(segment_points)

    # 5. Add the very last point of the original polyline
    new_points.append(polyline[-1].reshape(1, 2))

    # 6. Concatenate all parts
    return np.vstack(new_points)



def plot_half_edge_connections(out_path, half_edges: dict, seg_img: np.ndarray, R):
    def midpoint(l):
        return l[len(l)//2] + 0.5
    positions = {k: midpoint(v[1]) for k, v in half_edges.items()}
    labels = {k: k for k in half_edges.keys()}
    with plu.ImageOverlay(output_path=out_path, cv2_img=seg_img) as ax:
        plu.plot_graph_from_M(ax, R, positions, node_labels=labels)


def verify_seg_img(seg_img: np.ndarray, half_edges: dict):
    
    counts = len(np.unique(seg_img))
    print("counted on image:", counts)
    print("n half eddges", len(half_edges))
    assert counts == len(half_edges)
    
        
    

def involutive_assignment(C: np.ndarray, maximize: bool = True):
    """
    Find a maximum- or minimum-cost involutive assignment.

    This function constructs a graph of size 2*N to transform the problem
    into a Maximum Weight Perfect Matching problem. 
    
    Nodes 0 to n-1 represent the 'Real' layer.
    Nodes n to 2n-1 represent the 'Shadow' layer.
    
    - Fixed points (C[i,i]) are modeled as edges between Real i and Shadow i.
    - Swaps (C[i,j]) are modeled as edges (i, j) in Real and (i+n, j+n) in Shadow.
    
    The solver finds a perfect matching that maximizes the sum of weights.
    Due to the symmetric construction, the optimal matching in the Real layer 
    mirrors the Shadow layer, providing the correct total cost for swaps 
    (C[i,j] + C[j,i]) without manually summing them beforehand.

    Parameters
    ----------
    C : np.ndarray (n x n)
        Symmetric cost matrix. C[i,j] == 0 means forbidden.
    maximize : bool
        If True, maximize total cost. If False, minimize.

    Returns
    -------
    perm : np.ndarray
        Involutive permutation array of length n.
    total_cost : float
        Total assignment cost.
    """
    n = C.shape[0]
    G = nx.Graph()
    

    def get_weight(val):
        return val if maximize else -val

    # Construct the 2N graph
    for i in range(n):
        # 1. Edge for Fixed Point: Connect Real i to Shadow i (i <-> i+n)
        # Weight is C[i, i]
        w_fixed = get_weight(C[i, i])
        G.add_edge(i, i + n, weight=w_fixed)
        
        # 2. Edges for Swaps: Connect i <-> j within layers
        for j in range(i + 1, n):
            if C[i, j] == 0.0 or not np.isfinite(C[i, j]):
                continue
            
            w_swap = get_weight(C[i, j])
            
            # Add edge in Real layer
            G.add_edge(i, j, weight=w_swap)
            
            # Add edge in Shadow layer
            G.add_edge(i + n, j + n, weight=w_swap)

    # Solve Maximum Weight Perfect Matching
    # maxcardinality=True ensures the matching covers as many nodes as possible 
    # (conceptually a Perfect Matching for this construction).
    with utils.timer("max matching"):
        matching = nx.max_weight_matching(G, maxcardinality=True)

    # Reconstruct the permutation
    perm = np.arange(n, dtype=int)
    
    # We only need to parse the matching relative to the Real layer (0 to n-1)
    for u, v in matching:
        # Normalize so u is smaller
        if u > v:
            u, v = v, u
            
        # Case 1: Fixed Point Edge (u in Real, v in Shadow)
        # u < n and v >= n. Specifically v should be u + n.
        if v == u + n:
                perm[u] = u
        
        # Case 2: Swap Edge in Real Layer (u < n and v < n)
        # Note: We ignore Swap Edges in Shadow Layer (u >= n and v >= n)
        elif u < n and v < n:
            perm[u] = v
            perm[v] = u

    # Calculate exact total cost based on the resulting permutation
    total_cost = 0.0
    for i in range(n):
        total_cost += C[i, perm[i]]

    return perm, total_cost


def involutive_assignment2(C: np.ndarray, maximize: bool = True):
    n = C.shape[0]
    diag = np.diag(C).copy()

    # Build n-node graph where matching = swaps, unmatched = fixed points
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] == 0.0 or not np.isfinite(C[i, j]):
                continue
            # Gain of swapping (i,j) vs fixing both
            gain = 2 * C[i, j] - diag[i] - diag[j]
            if maximize and gain > 0:
                G.add_edge(i, j, weight=gain)
            elif not maximize and gain < 0:
                G.add_edge(i, j, weight=-gain)

    matching = nx.max_weight_matching(G, maxcardinality=False)

    perm = np.arange(n, dtype=int)
    for u, v in matching:
        perm[u] = v
        perm[v] = u

    total_cost = sum(C[i, perm[i]] for i in range(n))
    return perm, total_cost

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        # If they are already in the same component, do nothing
        if root_i == root_j:
            return False

        # Union by Rank: attach smaller tree to larger tree
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            # If ranks are same, attach one to other and increment rank
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
            
        return True

def reconstruct_paths(half_edge_pairs: np.ndarray, perm: np.ndarray, debug = False):
    #print("input dim", M.shape, len(half_edge_pairs))
    assert len(half_edge_pairs) == perm.shape[0]
    N = len(half_edge_pairs) - 1
    uf = UnionFind(perm.shape[0])
    #skip 0
    for idx in range(1, N+1):
        #correct because we know there is only 1 in the row
        col_idx = perm[idx]
        uf.union(idx, col_idx)
        uf.union(idx, half_edge_pairs[idx])
        if debug:
            print("union", idx, col_idx, half_edge_pairs[idx])
    
    components = {}
    [components.setdefault(uf.find(k), []).append(k) for k in range(1, N+1)]
    return components


def indexed_color(i: int) -> tuple[int, int, int]:
    """
    Encode an integer index into an RGB color using base-256.

    Parameters
    ----------
    i : int
        Layer index (0 <= i < 256**3)

    Returns
    -------
    (r, g, b) : tuple[int, int, int]
        RGB color encoding the index
    """
    if i < 0 or i >= 16_777_216:
        raise ValueError("Index out of range (must be 0 <= i < 256^3)")

    r = i % 256
    g = (i // 256) % 256
    b = (i // (256**2)) % 256

    return r, g, b

def draw_polyline_cv2(image, polyline, color, aa=False):
    # 1. Ensure the polyline is in the correct format (int32)
    # OpenCV requires a list of arrays with shape (number_of_points, 1, 2)
    pts = polyline.astype(np.int32).reshape((-1, 1, 2))
    
    # 2. Draw the polyline
    # lineType=cv2.LINE_8 ensures no anti-aliasing
    # isClosed=False ensures it's an open path (polyline) rather than a polygon
    cv2.polylines(
        image, 
        [pts], 
        isClosed=False, 
        color=color, 
        thickness=1, 
        lineType=cv2.LINE_AA if aa else cv2.LINE_8
    )



def get_polyline_pixels(polyline, dim=None):
    """
    Returns the ordered (N, 2) array of [y, x] pixel coordinates for a polyline.
    
    Args:
        polyline: np.array of shape (N, 2) containing [x, y] coordinates.
        dim: Tuple (height, width) to clip pixels. Pixels outside are removed.
             Order is preserved.
    """
    all_segments = []
    
    for i in range(len(polyline) - 1):
        p0 = polyline[i]
        p1 = polyline[i+1]
        
        # 1. Generate line pixels for this segment
        # skimage uses (row, col) which is (y, x)
        rr, cc = line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
        segment = np.column_stack((rr, cc)) # shape (M, 2) as [x, y]
        
        # 2. Filter by dimensions if provided
        if dim is not None:
            h, w = dim
            # mask: 0 <= x < width AND 0 <= y < height
            mask = (segment[:, 0] >= 0) & (segment[:, 0] < w) & \
                   (segment[:, 1] >= 0) & (segment[:, 1] < h)
            segment = segment[mask]
        
        if len(segment) > 0:
            all_segments.append(segment)

    if not all_segments:
        return np.empty((0, 2), dtype=np.int32)

    # 3. Concatenate all valid segments
    pixels = np.concatenate(all_segments, axis=0)
    
    # 4. Remove consecutive duplicates (vertices shared by segments)
    # This ensures a clean 1px stroke order without "double-stepping" on joints
    if len(pixels) > 1:
        # Keep pixel if it is different from the previous pixel
        diff_mask = np.ones(len(pixels), dtype=bool)
        diff_mask[1:] = np.any(pixels[1:] != pixels[:-1], axis=1)
        pixels = pixels[diff_mask]
    
        
    return pixels



def split_in_half(polylines: list[np.ndarray], 
                                    connections: list[tuple[int, int]],):
    
    res = {}
    half_edge_to_node = {}
    c = 1
    for (l_node_idx, r_node_idx), pl in zip(connections, polylines):
        #special case
        if len(pl) ==2:
            mid = (pl[0]+pl[1])/2
            left_points = np.array([pl[0], mid])
            right_points = np.array([mid, pl[1]])
        else:
            line = densify_polyline_maxdist(pl, 0.5)
            k = len(line)//2
            # we want the connection point to overlap, so k+1
            left_points = line[:k+1]
            right_points =line[k:][::-1]
        half_edge_to_node[c] = (l_node_idx, left_points)
        half_edge_to_node[c+1] = (r_node_idx, right_points)
        res.setdefault(l_node_idx, []).append(c)
        res.setdefault(r_node_idx, []).append(c+1)
        c+=2
    
    assert len(half_edge_to_node) == 2*len(polylines)
    return res, half_edge_to_node

def resample_polyline(points, N):
    """resamples a single polyline of shape (V, 2) to (N, 2). """
    points = np.asarray(points)
    if len(points) == 1:
        return np.repeat(points, N, axis=0)

    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    
    total_length = cum_dists[-1]
    if total_length == 0:
        return np.repeat(points[:1], N, axis=0)
    
    cum_dists_norm = cum_dists / total_length
    interpolator = interp1d(cum_dists_norm, points, axis=0, kind='linear')
    t = np.linspace(0, 1, N)
    resampled = interpolator(t)
    
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled
    

def build_intersection_segmentation(polylines: list[np.ndarray], 
                                    connections: list[tuple[int, int]],
                                    source_size, 
                                    target_size= 500,
                                    vis = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
    
    M = transform.compute_matrix_for_rescaling(source_size, source_size, target_size)
        
    res, half_edge = split_in_half(polylines, connections)
    N = len(half_edge)+1
    # step 1: build constraint matrix R
    R = np.zeros((N, N), dtype=np.int32)
    for row, v in half_edge.items():
        nidx = v[0]
        for col in res[nidx]:
            R[row, col] = 1
    R[0, 0] = 1
    
    # step 2: build segmentation image
    seg_img = np.zeros((target_size, target_size), dtype=np.int32)
    
    for idx, v in half_edge.items():
        pl = v[1]
        coords = get_polyline_pixels(pl, (target_size, target_size))
        seg_img[coords[:, 0], coords[:, 1]] = idx
            
    
    # step 3: line drawing
    img_dr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    for pl in polylines:
        draw_polyline_cv2(img_dr, pl, (255, 255, 255), aa=True)
        
    
    # step 4: segmentation image with visualisation colors (not strictly needed)
    outvis = None
    if vis:
        outvis = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        colors = itertools.cycle(utils.PAIRED_COLORS)
        for i in range(1, N, 2):
            pl1 = half_edge[i][1]
            pl2 = half_edge[i+1][1]
            c1, c2 = next(colors)
            coords1 = get_polyline_pixels(pl1, (target_size, target_size))
            coords2 = get_polyline_pixels(pl2, (target_size, target_size))
            outvis[coords1[:, 0], coords1[:, 1]] = utils.color_to_bgr(c1)
            outvis[coords2[:, 0], coords2[:, 1]] = utils.color_to_bgr(c2)
        
    return img_dr, seg_img, R, half_edge, outvis

def get_line_tensor(half_edges: dict):
    mat =  compute_matrix_for_rescaling(768, 2024, 512)
    res = torch.zeros(len(half_edges)+1, 100, 2)
    for k, v in half_edges.items():
        res[k] = torch.tensor(resample_polyline(v[1], 100))
    return res
        

def prepare_batch(img_dr: np.ndarray, seg_img: np.ndarray, R: np.ndarray, half_edges: dict):
    
    print(img_dr.shape)
    assert img_dr.ndim == 3
    assert seg_img.ndim == 2
    #assert img_dr.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)
   # assert seg_img.shape == (IMAGE_SIZE, IMAGE_SIZE)
    N = R.shape[0]
    assert N == len(half_edges) +1
    img_tensor = utils.cv2_to_torch(img_dr)
    print("img dtype", img_tensor.dtype)
    seg = torch.from_numpy(seg_img).long()
    # padded_matrix = torch.zeros((MAX_OBJECTS, MAX_OBJECTS), dtype=torch.float32)
    # padded_matrix[:N, :N] = torch.from_numpy(R).to(torch.float32)
    padded_matrix = torch.from_numpy(R).to(torch.float32)
    lines = get_line_tensor(half_edges)
    return {"img": img_tensor.unsqueeze(0),
            "seg": seg.unsqueeze(0),
            "lines": lines.unsqueeze(0),
            "R":padded_matrix.unsqueeze(0),
            "n_objects": torch.tensor([N], dtype=torch.int64)
            }

def scan_module(module):
    print("---- PARAMETERS ----")
    for n, p in module.named_parameters():
        if p.dtype == torch.float64:
            print("PARAM:", n)

    print("---- BUFFERS ----")
    for n, b in module.named_buffers():
        if b.dtype == torch.float64:
            print("BUFFER:", n)
    
    for k, v in module.state_dict().items():
        if v.dtype == torch.float64:
            print("STATE:", k)


def load_model(device : torch.device = torch.device("cpu")):
    model_path = os.environ["INTERSECTION_MODEL"]
    # This loads the logic AND the weights
    # device = torch.device("mps")
    model = torch.jit.load(model_path, map_location=device)
    utils.get_param_count(model)
    return model


def predict(model: torch.ScriptModule, img_dr: np.ndarray, seg_img: np.ndarray, R: np.ndarray, half_edges: dict) -> np.ndarray:
    N = R.shape[0]
    batch = prepare_batch(img_dr, seg_img, R, half_edges)
    d = torch.device("cpu")
    if torch.backends.mps.is_available():
        d = torch.device("mps")
    batch = {k: v.to(d) for k, v in batch.items()}
    model = model.to(d)
    with torch.no_grad():
        pred = model(batch)
    mat = pred[0][:N, :N].cpu().numpy()
    
    return mat

def reconstruct_ordered_paths(nodes, m1, m2):
    visited = set()
    paths = []

    def next_node(curr, prev):
        """Return the next node from curr that is not prev (if possible)."""
        a, b = m1[curr], m2[curr]

        # Choose endpoint that is not prev
        if a != prev and a != curr:
            return a
        if b != prev and b != curr:
            return b
        return None

    for start in nodes:
        if start in visited:
            continue

        # Build path starting from 'start'
        path = [start]
        visited.add(start)

        prev = None
        curr = start
        while True:
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        prev = None
        curr = start
        while True:
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.insert(0, nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        paths.append(path)

    return paths



def reorient_sublines(sublines):
    """
    Ensure all sub-polylines are consistently oriented.

    Parameters
    ----------
    sublines : list of np.ndarray
        Each array is shape (N, D) representing a polyline.

    Returns
    -------
    list of np.ndarray
        Reoriented sublines.
    """

    if not sublines:
        return []

    oriented = [sublines[0]]

    for line in sublines[1:]:
        prev = oriented[-1]

        prev_end = prev[-1]
        curr_start = line[0]
        curr_end = line[-1]

        if np.array_equal(prev_end, curr_start):
            # already correctly oriented
            oriented.append(line)

        elif np.array_equal(prev_end, curr_end):
            # needs reversing
            oriented.append(line[::-1])

        else:
            raise ValueError("Sublines do not connect exactly.")

    return oriented

def get_matched_lines(rawmat: np.ndarray, half_edges: dict):
    print(rawmat.shape)
    N = len(half_edges)
    
    #print({k: v[0] for k, v in half_edges.items()})
    assert rawmat.ndim == 2
    assert rawmat.shape[0] == len(half_edges)+1
    # with utils.timer("computing assignment"):
    perm, _ = involutive_assignment2(rawmat)
    # with utils.timer("computing assignment slow"):
    #     perm2, _ = involutive_assignment(rawmat)
    # print("result equal: ", np.array_equal(perm, perm2))
        
    print("perm:", perm.shape)
    pairs = np.array([
        (x+1 if x%2 else x-1) for x in range(0, len(half_edges)+1)])
    assert len(pairs) == len(half_edges) +1
    nodes = np.arange(1, N+1)
    components = reconstruct_ordered_paths(nodes, pairs, perm)
    paths = []
    for idx_set in components:
        paths.append(
            reorient_sublines([half_edges[i][1] for i in idx_set])
        )
    
    return paths
            
    