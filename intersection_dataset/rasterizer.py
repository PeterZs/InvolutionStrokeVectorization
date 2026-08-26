import cv2
import numpy as np
from skimage.draw import line
import math
from typing import NamedTuple, Any
import sampler.intersections as intersections
import itertools
import sampler.randompath as randompath


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



def maybe_intersection(real_intersection_map: dict, yx_coord: tuple[int, int], pl_idx, visited: set[tuple[int, int]]):
    
    if yx_coord in real_intersection_map:
        if real_intersection_map[yx_coord][0] == pl_idx and yx_coord not in visited:
            return yx_coord , True
    for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]:
        neighbor = (yx_coord[0] + dy, yx_coord[1] + dx)
        if neighbor in real_intersection_map:
            if real_intersection_map[neighbor][0]:
                return neighbor, False
    
    return None, False



def split_segments(arr, include_trapped_reds=False):
    """
    Splits a numpy array of 0, 1, 2 into segments.
    
    Returns a list of 3-tuples: (slice_object, (left_bound_idx, right_bound_idx), is_pure_red)
    """
    # Create a boolean mask where 0 is True
    is_zero = (arr == 0)
    N = len(arr)
    
    # --- Step 1: Identify Core Segments (Runs of 0s) ---
    padded_mask = np.concatenate(([False], is_zero, [False]))
    diffs = np.diff(padded_mask.astype(np.int8))
    
    run_starts = np.flatnonzero(diffs == 1)
    run_ends = np.flatnonzero(diffs == -1)
    
    # We use a mutable dictionary for segments initially to update start/ends easily
    # Structure: {'start': int, 'end': int, 'is_red': bool}
    core_segments = [{'start': s, 'end': e, 'is_red': False} 
                     for s, e in zip(run_starts, run_ends)]
    
    if not core_segments:
        # If there are no zeros, technically there are no segments by the main definition.
        # If specific handling for "only trapped reds" without zeros is needed, 
        # it would go here, but usually this returns empty.
        return []

    # --- Step 2: Handle Outer Bounds (Start/End of Array) ---
    
    # 2a. Expand First Segment Leftwards
    # Look at gap from 0 to seg[0].start
    first_gap_idx = np.arange(0, core_segments[0]['start'])
    if len(first_gap_idx) > 0:
        gap_vals = arr[first_gap_idx]
        black_locs = np.flatnonzero(gap_vals == 2)
        if len(black_locs) > 0:
            # Extend left to the node immediately following the LAST black in this pre-gap
            last_black_idx = first_gap_idx[black_locs[-1]]
            core_segments[0]['start'] = last_black_idx + 1
        else:
            # No blacks, extend all the way to start
            core_segments[0]['start'] = 0

    # 2b. Expand Last Segment Rightwards
    # Look at gap from seg[-1].end to len(arr)
    last_gap_idx = np.arange(core_segments[-1]['end'], N)
    if len(last_gap_idx) > 0:
        gap_vals = arr[last_gap_idx]
        black_locs = np.flatnonzero(gap_vals == 2)
        if len(black_locs) > 0:
            # Extend right to the node immediately before the FIRST black in this post-gap
            first_black_idx = last_gap_idx[black_locs[0]]
            core_segments[-1]['end'] = first_black_idx
        else:
            # No blacks, extend all the way to end
            core_segments[-1]['end'] = N

    # --- Step 3: Process Internal Gaps ---
    
    final_segments_data = [core_segments[0]]

    for i in range(len(core_segments) - 1):
        left_seg = final_segments_data[-1]  # The segment to the left of the gap
        right_seg = core_segments[i+1]      # The segment to the right of the gap
        
        # Identify Gap
        gap_start = left_seg['end']
        gap_end = right_seg['start']
        gap_indices = np.arange(gap_start, gap_end)
        gap_values = arr[gap_indices]
        
        black_locs = np.flatnonzero(gap_values == 2)
        
        if len(black_locs) == 0:
            # Case A: Only Reds (or empty) -> Split Evenly
            n_reds = len(gap_indices)
            n_left = (n_reds + 1) // 2
            
            # Update bounds
            # add -1 because we don't want to include the node itself
            left_seg['end'] += n_left-1
            
            right_seg['start'] -= (n_reds - n_left)
        else:
            # Case B: Contains Blacks
            
            # 1. Left segment absorbs reds up to the FIRST black node
            first_black_global_idx = gap_indices[black_locs[0]]
            left_seg['end'] = first_black_global_idx
            
            # 2. Right segment absorbs reds after the LAST black node
            last_black_global_idx = gap_indices[black_locs[-1]]
            right_seg['start'] = last_black_global_idx + 1
            
            # 3. Handle Trapped Reds (Independent Segments)
            if include_trapped_reds and len(black_locs) > 1:
                # Iterate between black pairs
                for b_i in range(len(black_locs) - 1):
                    # Indices of the black nodes themselves
                    b_idx_curr = gap_indices[black_locs[b_i]]
                    b_idx_next = gap_indices[black_locs[b_i+1]]
                    
                    # If there is space between them
                    if b_idx_next > b_idx_curr + 1:
                        new_seg = {
                            'start': b_idx_curr + 1,
                            'end': b_idx_next,
                            'is_red': True
                        }
                        final_segments_data.append(new_seg)

        # Add the next core segment to the list
        final_segments_data.append(right_seg)

    # --- Step 4: Format Output ---
    
    formatted_results = []
    for seg in final_segments_data:
        start, end = seg['start'], seg['end']
        
        # Create Slice
        slc = slice(int(start), int(end))
        
        # Calculate Bound Indices
        # Left Bound: The index before start. If start is 0, bound is -1 (array edge)
        left_bound = int(start - 1) if start > 0 else 0
        
        # Right Bound: The index at end (since slice is exclusive, 'end' is the first outside item). 
        # If end is N, bound is -1 (array edge)
        right_bound = int(end) if end < N else N-1
        # if left_bound >=0:
        #     assert arr[left_bound] > 0
        # if right_bound >=0:
            # print({i: int(x) for i, x in zip(range(right_bound-5,right_bound+5),
            #     arr[right_bound-5: right_bound+5])}, right_bound)
            # assert arr[right_bound] > 0
        
        formatted_results.append({"slice": slc, "left": left_bound, "right": right_bound, "is_red": seg['is_red']})
        
    return formatted_results


class SplitInfo(NamedTuple):
    coords: np.ndarray
    idx: int
    left_neighbors: set[int]
    right_neighbors: set[int]
    vertex_left: int | None
    vertex_right: int | None
    
    def __repr__(self) -> str:
        return f'''(idx {self.idx},
        ln {self.left_neighbors if len(self.left_neighbors ) else {}}, 
        rn {self.right_neighbors if len(self.right_neighbors ) else {}}, 
        vl {self.vertex_left}, 
        vr {self.vertex_right})'''.replace("  ", "").replace("\n", "")
    
def rasterized_split_seqs(polylines, 
                                 real_intersections: list[np.ndarray], dim=None) -> list[list[SplitInfo]]:
    pixel_sequences = []
    
    # 1. Generate pixels
    for pl_idx, pl in enumerate(polylines):
        px = get_polyline_pixels(pl, dim)
        pixel_sequences.append(px)
       
    real_intersections_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pl_idx, inters in enumerate(real_intersections):
        for k, (x, y) in enumerate(inters):
            real_intersections_map.setdefault(
                (math.floor(y), math.floor(x)), []).append((pl_idx, k))
            
    results = []
    
    # 2. Process each polyline sequence
    for pl_idx, px in enumerate(pixel_sequences):
        if len(px) == 0:
            results.append([])
            continue
        
        
        detect_lst = []
        
        #maps quantized intersections -> index in the pixel sequence
        point_mp = {0: (int(px[0][0]), int(px[0][1])), 
                    len(px)-1: (int(px[-1][0]), int(px[-1][1]))}
        visited_intersections = set()
        for i, p in enumerate(px):
            coord = (int(p[0]), int(p[1])) # y, x
            inter_point, true_intersection = maybe_intersection(
                real_intersections_map, coord, pl_idx, visited_intersections)
            if inter_point is not None:
                point_mp[i] = inter_point
                detect_lst.append(2 if true_intersection else 1)
                if true_intersection:
                    visited_intersections.add(inter_point)
            else:
                detect_lst.append(0)
            
        detect_array = np.array(detect_lst)
        # print("num of black nodes", len(detect_array[detect_array==2]))
        # print("num of red nodes", len(detect_array[detect_array==1]))
        sub_idx = split_segments(detect_array, include_trapped_reds=True)
        splits_tup = [(x["slice"], point_mp[x["left"]], point_mp[x["right"]]) for x in sub_idx]
        results.append(splits_tup)
    
    
    #3. Generate graph information
    #maps point tuple -> neighbor edges
    mp = {}
    edge_idx = 0
    for splits in results: 
        for (subline_slice, left_p, right_p) in splits:
            # the point acts as the key to search for neighbors
            if left_p is not None:
                mp.setdefault(left_p, []).append(edge_idx)
            if right_p is not None:
                mp.setdefault(right_p, []).append(edge_idx)
            edge_idx+=1
    edge_idx = 0
    output = []
    global_vertex_id = {None: None}
    for i, p in enumerate(mp.keys()):
        global_vertex_id[p] = i
    for splits, px in zip(results, pixel_sequences): 
        curr_splits: list[SplitInfo] = []
        for subline_slice, left_p, right_p in splits:
            neighbors_left = mp[left_p] if left_p is not None else []
            neighbors_right = mp[right_p] if right_p is not None else []
            coords = px[subline_slice]
            curr_splits.append(SplitInfo(coords=coords,
                                         idx=edge_idx,
                                         left_neighbors=set(neighbors_left),
                                         right_neighbors=set(neighbors_right),
                                         vertex_left=global_vertex_id[left_p],
                                         vertex_right=global_vertex_id[right_p]))
            # curr_splits.append((coords, edge_idx, set(neighbors_left), set(neighbors_right), 
            #              global_vertex_id[left_p], 
            #              global_vertex_id[right_p]))
            edge_idx+=1
        output.append(curr_splits)
        
    return output


def get_half_edge_splits(sequences: list[list[SplitInfo]]) ->list[list[tuple]]:
    
    #offset for half-edge indexes
    n = sum(len(x) for x in sequences)
    
    nodes = {}
    for pl_idx, line_splits in enumerate(sequences):
        for k, split in enumerate(line_splits):
            nodes[split.idx] = (pl_idx, k, split.idx, split.idx+n)
            
    
    result = []
    for line_splits in sequences:
        half_edges_curr = []
        for split in line_splits:
            mid = len(split.coords)//2
            coords_half_a = split.coords[:mid]
            coords_half_b = split.coords[mid:]
            #problem: for the neighboring edges at both sides, 
            # we don't know in which direction they're oriented
            # (if they're adjacent on the "left" or the "right")
            # so we have to check
            nbrs_half_a = []
            for i in split.left_neighbors:
                if i == split.idx:
                    continue
                nbr_info = nodes[i]
                other_left_id = sequences[nbr_info[0]][nbr_info[1]][4]
                other_right_id = sequences[nbr_info[0]][nbr_info[1]][5]
                if other_left_id == split.vertex_left:
                     nbrs_half_a.append(nbr_info[2])
                if other_right_id == split.vertex_left:
                     nbrs_half_a.append(nbr_info[3])
            
            #same for the right end
            nbrs_half_b = []
            for i in split.right_neighbors:
                if i == split.idx:
                    continue
                nbr_info = nodes[i]
                other_left_id = sequences[nbr_info[0]][nbr_info[1]][4]
                other_right_id = sequences[nbr_info[0]][nbr_info[1]][5]
                if other_left_id == split.vertex_right:
                     nbrs_half_b.append(nbr_info[2])
                if other_right_id == split.vertex_right:
                     nbrs_half_b.append(nbr_info[3])
            
            #append half edge a and b
            half_edges_curr.append((coords_half_a, split.idx, nbrs_half_a))
            half_edges_curr.append((coords_half_b, split.idx+n, nbrs_half_b))
            
        result.append(half_edges_curr)
        
    assert verify_indices(result)
    
    return result

def split_up(pl):
    if len(pl) ==2:
        mid = (pl[0]+pl[1])/2
        left_points = np.array([pl[0], mid])
        right_points = np.array([mid, pl[1]])
    else:
        line = pl
        k = len(line)//2
        # we want the connection point to overlap, so k+1
        left_points = line[:k+1]
        right_points =line[k:]
    return left_points, right_points

def polyline_half_edges(polylines: list[np.ndarray], 
                        inters: list[list[intersections.IntersectionPoint]], n) ->dict[int, np.ndarray]:

    result ={}
    idx =0
    for pl, it in zip(polylines, inters):
        sub = intersections.split_up(pl, it)
        for subline in sub:
            left, right = split_up(randompath.densify_polyline_maxdist(subline, 0.5))
            result[idx] = left
            #we want the half-edge lines pointing away from their intersection point
            result[idx+n] = right[::-1]
            idx+=1
        print("index", idx)
    return result

def half_edges_from_sublines(polylines:  list[np.ndarray]):
    result = []
    for pl in polylines:
        left, right = split_up(randompath.densify_polyline_maxdist(pl, 0.5))
        result.append(left)
                #we want the half-edge lines pointing away from their intersection point
        result.append(right[::-1])
    return result
    
    

def polyline_half_edges_full(polylines: list[np.ndarray], 
                        inters: list[list[intersections.IntersectionPoint]]) ->tuple[dict[int, np.ndarray], dict, dict]:

    sublines = []
    for pl, it in zip(polylines, inters):
        sub = intersections.split_up(pl, it)
        # print("num sub", len(sub))
        sublines.append(sub)
    

    n = sum(len(x) for x in sublines)
    result ={}
    pairs = {}
    nxt = {}
    idx =0
    for sub in sublines:
        for i, subline in enumerate(sub):
            if i >0:
                nxt[idx+n-1] = idx
            pairs[idx] = idx+n
            pairs[idx+n] = idx
            left, right = split_up(randompath.densify_polyline_maxdist(subline, 0.5))
            result[idx] = left
            #we want the half-edge lines pointing away from their intersection point
            result[idx+n] = right[::-1]
            idx+=1
        # print("index", idx) 
    return result, pairs, nxt

def get_matrix_M(polylines: dict[int, np.ndarray], nxt: dict):
    n = len(polylines)
    M = np.zeros((n, n), dtype=np.int32)
    
    #not the fastest but fair
    for i in range(n):
        for j in range(i, n):
            # Two polylines are adjacent iff their first point matches exactly
            if np.array_equal(polylines[i][0], polylines[j][0]):
                M[i, j] = 1
                M[j, i] = 1
    
    for k, v in nxt.items():
        M[k, v] +=1
        M[v, k] +=1
    
    for row in range(len(M)):
        if (M[row]==2).sum() != 1:
            M[row, row] +=1
        # if M[row][M[row]==2].sum():
                
    # Assert that the matrix is symmetric
    assert np.array_equal(M, M.T), "Adjacency matrix is not symmetric"
    
    return M
    
 
def verify_indices(half_edges: list[list[tuple]]):
    
    unique = set()
    
    for splits in half_edges:
        for he in splits:
            if he[1] in unique:
                return False
            unique.add(he[1])
    
    return True


def are_half_edges_valid(half_edges: list[list[tuple]]):
    for splits in half_edges:
        for he in splits:
            if len(he[0]) < 2:
                return False
    return True
    

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

def adjacency_mat(half_edges):
    n = sum(len(x) for x in half_edges)
    M = np.zeros((n, n), dtype=int)
    #all diags must be 1 because any half-edge might end there
    np.fill_diagonal(M, 1)
    for half_edges_curr in half_edges:
        for i in range(len(half_edges_curr)):
            neighbors = half_edges_curr[i][2]
            h_idx = half_edges_curr[i][1]
            for v in neighbors:
                M[h_idx, v] += 1
            
            
            if (i%2==1) and (i+1 < len(half_edges_curr)):
                nxt =  half_edges_curr[i+1][1]
                #these are the target edges, so they are =2 in the adjacency matrix
                M[h_idx, nxt] +=1
                M[nxt, h_idx] +=1
                

    for i, row in enumerate(M):
        if (row==2).sum() == 0:
            # this means that it is a path-terminating half-edge 
            # that may or may not have neighbors
            M[i, i] += 1
    
    assert np.array_equal(M, M.T)
    #all rows must have exactly 1 2
    assert all((row==2).sum() ==1 for row in M)
    return M
    

def color_half_edges_for_model(img, half_edges):
    
    for half_edges_curr in half_edges:
        for i, (coords, idx, _) in enumerate(half_edges_curr, ):
            #add +1 everywhere so that we don't get black
            r, g, b= indexed_color(idx+1)
            #stay consistent with cv2's color ordering
            img[coords[:, 0], coords[:, 1]] = (b, g, r)
            

def count_colors(img: np.ndarray):
    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)


def check_numbers(img, intersections, lines):
    # in each polyline: if there are k intersections, there are 2k+2 half-edges, 
    # UNLESS an intersection is an endpoint
    n_expected = sum(2*len(i)+2 for i in intersections)
    n_expected = 0
    for ip, ln in zip(intersections, lines):
        k = 2*len(ip)+2
        if len(ip):
            k -= 2*np.array_equal(ip[0], ln[0])
            k -= 2*np.array_equal(ip[-1], ln[-1])
        n_expected +=k
            
    print("n expected", n_expected)
    # -1 because black is the background
    print("n counted", count_colors(img)-1)
    return count_colors(img)-1 == n_expected


def half_edge_pairs(half_edges) -> dict[int, int]:
    res = {}
    for he_curr in half_edges:
        #print([a[1] for a in he_curr])
        for i in range(0, len(he_curr), 2):
            a = he_curr[i]
            b = he_curr[i+1]
            res[a[1]] = b[1]
            res[b[1]] = a[1]
    return res
             

def half_edge_centers(half_edges):
    pos={}
    for splits_curr in half_edges:
        for i, (coords, idx, _) in enumerate(splits_curr):
            mid = len(coords)//2
            pos[idx] = coords[mid][1]+0.5, coords[mid][0]+0.5
    return pos

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

def reconstruct_paths(half_edge_pairs: dict[int, int], M: np.ndarray):
    
    uf = UnionFind(M.shape[0])
    for idx, row in enumerate(M):
        #correct because we know there is exactly 1 2 in the row
        col_idx = np.flatnonzero(row==2)[0]
        uf.union(idx, col_idx)
        uf.union(idx, half_edge_pairs[idx])
    
    components = {}
    [components.setdefault(uf.find(k), []).append(k) for k in range(len(M))]
    return components


def reconstruct_image(components: dict[int, list[int]], seg_img: np.ndarray, color_iter):
    print(components)
    for k, v in components.items():
        new_color = next(color_iter)
        for idx in v:
            #invert because we need (b, g, r)
            orig = indexed_color(idx+1)[::-1]
            mask = np.all(seg_img == orig, axis=-1)
            seg_img[mask] = np.array(new_color)
    
    

def obtain_edge_split_graph_for_viz(sequences):
    
    nodes = {}
    labels= {}

    for line_splits in sequences:
        for i, (coords, edge_idx, neighbors_left, neighbors_right, id_left, id_right) in enumerate(line_splits):
            pos = tuple(coords[len(coords)//2])
            x, y = int(pos[1])+0.5, int(pos[0])+0.5
            nodes[edge_idx]=  ((x, y), [])
            labels[edge_idx] = f"{id_left}->{id_right}"
    for line_splits in sequences:
        for i, (coords, edge_idx, neighbors_left, neighbors_right, _, _) in enumerate(line_splits):
            nodes[edge_idx][1].extend(neighbors_left)
            nodes[edge_idx][1].extend(neighbors_right)
    return nodes, labels


def obtain_half_edge_graph_for_viz(sequences):
    
    nodes = {}


    for line_splits in sequences:
        for i, (coords, edge_idx, neighbors) in enumerate(line_splits):
            pos = tuple(coords[len(coords)//2])
            x, y = int(pos[1])+0.5, int(pos[0])+0.5
            nodes[edge_idx]=  ((x, y), [])
    for line_splits in sequences:
        for i, (coords, edge_idx, neighbors) in enumerate(line_splits):
            nodes[edge_idx][1].extend(neighbors)
    return nodes