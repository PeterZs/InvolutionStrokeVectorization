import numpy as np
from scipy.spatial import cKDTree
from collections import namedtuple, defaultdict, deque
from shapely.geometry import Polygon
from shapely import maximum_inscribed_circle



def lerp(a, b, t):
    return (1-t)*a + t*b

# point: np.ndarray (x, y)
# segment_index: int (index of the segment within the polyline)
# t: float (normalized distance along the segment, 0.0 to 1.0)
IntersectionPoint = namedtuple('IntersectionPoint', ['point', 'segment_index', 't'])

def _extract_segments(polylines: list[np.ndarray]):
    """
    Splits a list of polylines into a flattened array of segments 
    with associated bookkeeping data.
    
    Returns:
        starts (N, 2): Start points of all segments
        ends (N, 2): End points of all segments
        poly_indices (N,): Which polyline the segment belongs to
        seg_indices (N,): The index of the segment within its polyline
    """
    if not polylines:
        return np.zeros((0,2)), np.zeros((0,2)), np.array([]), np.array([])

    # We need to stack all segments. 
    # If a polyline has P points, it has P-1 segments.
    
    p_starts = []
    p_ends = []
    p_poly_idx = []
    p_seg_idx = []

    for i, poly in enumerate(polylines):
        if len(poly) < 2:
            continue
            
        # Segment starts are points [0 ... N-1]
        # Segment ends are points [1 ... N]
        starts = poly[:-1]
        ends = poly[1:]
        num_segs = len(starts)
        
        p_starts.append(starts)
        p_ends.append(ends)
        
        # Bookkeeping
        p_poly_idx.append(np.full(num_segs, i, dtype=np.int32))
        p_seg_idx.append(np.arange(num_segs, dtype=np.int32))

    # Concatenate everything into big arrays for vectorization
    if not p_starts:
        return np.zeros((0,2)), np.zeros((0,2)), np.array([]), np.array([])
        
    all_starts = np.vstack(p_starts)
    all_ends = np.vstack(p_ends)
    all_poly_indices = np.concatenate(p_poly_idx)
    all_seg_indices = np.concatenate(p_seg_idx)

    return all_starts, all_ends, all_poly_indices, all_seg_indices



def compute_intersections(polylines: list[np.ndarray]) -> list[list[IntersectionPoint]]:
    """
    Computes all self-intersections and mutual intersections between polylines.
    Uses cKDTree for spatial pruning and vectorized numpy algebra for intersection checks.
    """
    
    # 1. Extract Segments
    # A = Starts, B = Ends
    A, B, poly_ids, seg_ids = _extract_segments(polylines)
    n_segments = len(A)
    
    if n_segments < 2:
        return [[] for _ in polylines]

    # 2. Spatial Indexing
    # We use midpoints for the KDTree
    midpoints = (A + B) * 0.5
    
    # We need a search radius. 
    # To guarantee we catch all intersections, radius must be >= max_segment_length / 2.
    # Note: If segments vary wildly in length (e.g. 0.1 vs 1000.0), this is inefficient.
    # However, standard KDTree requires a fixed radius for query_pairs.
    segment_lengths = np.linalg.norm(B - A, axis=1)
    max_radius = np.max(segment_lengths) * 0.5001 # slightly larger for float safety
    
    # Build Tree
    tree = cKDTree(midpoints)
    
    # Query pairs within sum of radii (approximate as 2 * max_radius to be safe)
    # This returns a set of (i, j) where i < j
    candidates = tree.query_pairs(r=max_radius * 2.0, output_type='ndarray')
    
    if len(candidates) == 0:
        return [[] for _ in polylines]

    idx_i = candidates[:, 0]
    idx_j = candidates[:, 1]

    # 3. Vectorized Intersection Math
    # Retrieve coordinates for candidates
    A1, B1 = A[idx_i], B[idx_i] # Segment 1
    A2, B2 = A[idx_j], B[idx_j] # Segment 2

    # Fast AABB (Axis Aligned Bounding Box) Pruning
    # Even if midpoints are close, boxes might not overlap.
    min1 = np.minimum(A1, B1)
    max1 = np.maximum(A1, B1)
    min2 = np.minimum(A2, B2)
    max2 = np.maximum(A2, B2)

    # Overlap check: min1_x <= max2_x AND max1_x >= min2_x ...
    aabb_overlap = (
        (min1[:, 0] <= max2[:, 0]) & (max1[:, 0] >= min2[:, 0]) &
        (min1[:, 1] <= max2[:, 1]) & (max1[:, 1] >= min2[:, 1])
    )
    
    # Filter arrays based on AABB
    if not np.any(aabb_overlap):
        return [[] for _ in polylines]
        
    # Apply filter
    idx_i = idx_i[aabb_overlap]
    idx_j = idx_j[aabb_overlap]
    A1, B1 = A1[aabb_overlap], B1[aabb_overlap]
    A2, B2 = A2[aabb_overlap], B2[aabb_overlap]

    # 4. Detailed Intersection Calculation
    # Represent segments as P + t * D
    # Seg1: P1 + t * D1
    # Seg2: P2 + u * D2
    D1 = B1 - A1
    D2 = B2 - A2
    
    # 2D Cross Product (Determinant)
    # det = x1 * y2 - y1 * x2
    def cross_2d(v1, v2):
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    det = cross_2d(D1, D2)
    
    # Filter Parallel lines (det ~ 0)
    # Using a small epsilon
    EPS = 1e-9
    non_parallel = np.abs(det) > EPS
    
    if not np.any(non_parallel):
        return [[] for _ in polylines]

    # Apply filter
    idx_i = idx_i[non_parallel]
    idx_j = idx_j[non_parallel]
    A1, A2 = A1[non_parallel], A2[non_parallel]
    D1, D2 = D1[non_parallel], D2[non_parallel]
    det = det[non_parallel]

    # Solve for t and u using Cramer's Rule
    # t = ( (P2 - P1) x D2 ) / det
    # u = ( (P2 - P1) x D1 ) / det
    delta_p = A2 - A1
    t_vals = cross_2d(delta_p, D2) / det
    u_vals = cross_2d(delta_p, D1) / det

    # 5. Check if intersection is strictly within segments
    # 0 <= t <= 1 and 0 <= u <= 1
    # We use a small epsilon for tolerance if needed, but strict inequality usually fine
    valid_intersections = (
        (t_vals >= -EPS) & (t_vals <= 1.0 + EPS) &
        (u_vals >= -EPS) & (u_vals <= 1.0 + EPS)
    )
    
    # Apply Final Filter
    final_indices = np.where(valid_intersections)[0]
    
    if len(final_indices) == 0:
        return [[] for _ in polylines]

    # Extract final data
    final_i = idx_i[final_indices]
    final_j = idx_j[final_indices]
    final_t = t_vals[final_indices]
    final_u = u_vals[final_indices]
    
    # Clamp t/u to [0,1] to handle float epsilon overshoots
    final_t = np.clip(final_t, 0.0, 1.0)
    final_u = np.clip(final_u, 0.0, 1.0)

    # Retrieve Poly and Seg indices
    p_idx_i = poly_ids[final_i]
    s_idx_i = seg_ids[final_i]
    p_idx_j = poly_ids[final_j]
    s_idx_j = seg_ids[final_j]

    # 6. Filter Adjacent Segments (Shared Vertices)
    # If segments belong to same polyline and indices diff by 1, they share a vertex.
    # We usually exclude these unless strict touching is required.
    # Check: Same poly AND abs(seg_idx_diff) <= 1
    is_neighbor = (p_idx_i == p_idx_j) & (np.abs(s_idx_i - s_idx_j) <= 1)
    
    # Keep only non-neighbors (true intersections)
    keep_mask = ~is_neighbor
    
    final_i = final_i[keep_mask]
    final_t = final_t[keep_mask]
    final_u = final_u[keep_mask]
    
    # Recalculate helper arrays after neighbor pruning
    p_idx_i = poly_ids[final_i]
    s_idx_i = seg_ids[final_i]
    p_idx_j = poly_ids[idx_j[final_indices][keep_mask]]
    s_idx_j = seg_ids[idx_j[final_indices][keep_mask]]

    # 7. Compute Points and Format Output
    # P = A + t*D
    # We calculate intersection points once to ensure consistency
    pts = A[final_i] + final_t[:, None] * (B[final_i] - A[final_i])

    # Initialize result structure
    result = [[] for _ in range(len(polylines))]

    # Iterate and fill
    for pt, pi, si, t, pj, sj, u in zip(pts, p_idx_i, s_idx_i, final_t, p_idx_j, s_idx_j, final_u):
        
        # Create Point Object (shared for both)
        # Note: pt is a numpy array view, we usually copy it or keep as array
        # tuple(pt) ensures it's hashable/clean
        
        # Add to Poly I list
        result[pi].append(IntersectionPoint(point=pt, segment_index=si, t=t))
        
        # Add to Poly J list
        result[pj].append(IntersectionPoint(point=pt, segment_index=sj, t=u))

    return result




def compute_intersections_deduped(polylines: list[np.ndarray]) -> list[list[IntersectionPoint]]:
    """
    Computes intersections ensuring consistency at segment endpoints.
    Intersections at endpoints are normalized to t=0 of the subsequent segment.
    """
    
    # 1. Extract Segments
    A, B, poly_ids, seg_ids = _extract_segments(polylines)
    n_segments = len(A)
    
    if n_segments < 2:
        return [[] for _ in polylines]

    # 2. Spatial Indexing (Broad Phase)
    # Using midpoints and max segment radius
    midpoints = (A + B) * 0.5
    segment_lengths = np.linalg.norm(B - A, axis=1)
    if len(segment_lengths) == 0:
        return [[] for _ in polylines]
        
    max_radius = np.max(segment_lengths) * 0.500001
    print(max_radius)
    tree = cKDTree(midpoints)
    # 2.05 factor ensures we catch connections where midpoints are far but endpoints touch
    candidates = tree.query_pairs(r=max_radius * 2.05, output_type='ndarray')
    print("number of candidates", len(candidates))
    if len(candidates) == 0:
        return [[] for _ in polylines]

    idx_i = candidates[:, 0]
    idx_j = candidates[:, 1]

    # 3. Vectorized Intersection Math
    A1, B1 = A[idx_i], B[idx_i]
    A2, B2 = A[idx_j], B[idx_j]

    # AABB Pruning
    min1 = np.minimum(A1, B1)
    max1 = np.maximum(A1, B1)
    min2 = np.minimum(A2, B2)
    max2 = np.maximum(A2, B2)

    aabb_overlap = (
        (min1[:, 0] <= max2[:, 0]) & (max1[:, 0] >= min2[:, 0]) &
        (min1[:, 1] <= max2[:, 1]) & (max1[:, 1] >= min2[:, 1])
    )
    
    # Filter by AABB
    if not np.any(aabb_overlap):
        return [[] for _ in polylines]
        
    idx_i = idx_i[aabb_overlap]
    idx_j = idx_j[aabb_overlap]
    A1, B1 = A1[aabb_overlap], B1[aabb_overlap]
    A2, B2 = A2[aabb_overlap], B2[aabb_overlap]

    # Detailed Intersection (Cramer's Rule)
    D1 = B1 - A1
    D2 = B2 - A2
    
    def cross_2d(v1, v2):
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    det = cross_2d(D1, D2)
    
    # Filter Parallel lines
    EPS = 1e-9
    non_parallel = np.abs(det) > EPS
    
    if not np.any(non_parallel):
        return [[] for _ in polylines]

    idx_i = idx_i[non_parallel]
    idx_j = idx_j[non_parallel]
    A1, A2 = A1[non_parallel], A2[non_parallel]
    D1, D2 = D1[non_parallel], D2[non_parallel]
    det = det[non_parallel]

    # Solve for t and u
    delta_p = A2 - A1
    t_vals = cross_2d(delta_p, D2) / det
    u_vals = cross_2d(delta_p, D1) / det

    # 4. Snap and Filter
    # Check bounds with epsilon tolerance
    valid_intersections = (
        (t_vals >= -EPS) & (t_vals <= 1.0 + EPS) &
        (u_vals >= -EPS) & (u_vals <= 1.0 + EPS)
    )
    
    final_indices = np.where(valid_intersections)[0]
    
    if len(final_indices) == 0:
        return [[] for _ in polylines]

    # Extract valid data
    final_i = idx_i[final_indices]
    final_j = idx_j[final_indices]
    final_t = t_vals[final_indices]
    final_u = u_vals[final_indices]
    
    # --- CRITICAL FIX: Snapping and Shifting ---

    # 1. Snap tiny epsilons to 0.0 or 1.0
    final_t[np.abs(final_t) < EPS] = 0.0
    final_t[np.abs(final_t - 1.0) < EPS] = 1.0
    final_u[np.abs(final_u) < EPS] = 0.0
    final_u[np.abs(final_u - 1.0) < EPS] = 1.0

    # Retrieve Poly and Seg indices
    p_idx_i = poly_ids[final_i]
    s_idx_i = seg_ids[final_i]
    p_idx_j = poly_ids[final_j]
    s_idx_j = seg_ids[final_j]

    # 2. Filter Neighbors (Shared Vertices)
    # We do this BEFORE shifting indices. If seg 0 and seg 1 intersect at vertex 1,
    # it's a structural connection, not a self-intersection to report.
    # Logic: Same polyline AND segments are adjacent.
    is_neighbor = (p_idx_i == p_idx_j) & (np.abs(s_idx_i - s_idx_j) <= 1)
    keep_mask = ~is_neighbor
    
    if not np.any(keep_mask):
        return [[] for _ in polylines]

    final_i = final_i[keep_mask]
    final_t = final_t[keep_mask]
    final_u = final_u[keep_mask]
    p_idx_i = p_idx_i[keep_mask]
    s_idx_i = s_idx_i[keep_mask]
    p_idx_j = p_idx_j[keep_mask]
    s_idx_j = s_idx_j[keep_mask]
    print(s_idx_i, final_t)
    print(s_idx_j, final_u)
    # 3. Shift t=1.0 to t=0.0 of the next segment
    # This satisfies the requirement: 0 <= t < 1, even for mutual endpoint hits.
    # If t=1.0, we increment segment index and set t=0.0. 
    # This also handles the "last point" case by pushing it to index N.
    
    mask_shift_i = (final_t == 1.0)
    s_idx_i[mask_shift_i] += 1
    final_t[mask_shift_i] = 0.0
    
    mask_shift_j = (final_u == 1.0)
    s_idx_j[mask_shift_j] += 1
    final_u[mask_shift_j] = 0.0

    # 4. Compute Exact Points for consistency
    # Instead of P + t*D (which introduces float noise), we use A[idx] if t=0
    # Because we normalized t=1 -> t=0, we only need to check t=0.
    
    # We need the Original A arrays for the shifted indices.
    # However, s_idx might now be N (out of bounds for A). 
    # But note: A[seg_i + 1] is mathematically B[seg_i].
    # So if we didn't shift, we'd use B. Since we shifted, we conceptually use A of next.
    # To avoid array bounds issues with the vectorized check, we calculate pure float first
    # then snap coordinates.
    
    # Re-fetch A and B for the calculation (using original indices `final_i`)
    # We calculate based on the un-shifted geometry (segment i, t=1) == point B[i]
    # We effectively want: if t_original was 1, use B[final_i]. Else A[final_i] + t*D.
    
    # Recalculate original T for point generation to avoid index lookup complexity
    # (Since s_idx_i is now modified).
    
    points = np.zeros_like(A[final_i])
    
    # Vectorized Point Calculation
    # Note: final_t is currently 0.0 if it was shifted.
    # We can rely on the fact that we use A[original_idx] + original_t * D
    # But easier: Use the already computed 'final_t' (0 or fraction).
    # If we shifted, s_idx increased. We need the coordinates of the NEW s_idx start.
    # But determining coordinates for s_idx=N is hard without looking up the polyline.
    # Simpler approach: Compute point using UN-SHIFTED indices and UN-SHIFTED t.
    # But we overwrote final_t.
    
    # Let's compute points using simple vector algebra on the original segments
    # and then snap the resulting coordinates to the segment starts A if t was snapped to 0.
    
    # Re-derive unshifted T for calculation only? No, use the mask.
    raw_points = A[final_i] + (B[final_i] - A[final_i]) * np.where(mask_shift_i[:, None], 1.0, final_t[:, None])
    
    # If final_t is 0 (either snapped or shifted), we ideally want the exact vertex coordinate
    # to ensure dictionary keys match perfectly.
    # Logic: If mask_shift_i is True, we are at B[final_i].
    # If final_t is 0 and NOT shifted, we are at A[final_i].
    
    # Update raw_points to be exact vertices where applicable
    # Case 1: Was t=1 (Shifted) -> Use B[final_i]
    np.copyto(raw_points, B[final_i], where=mask_shift_i[:, None])
    
    # Case 2: Was t=0 (Not shifted, just 0) -> Use A[final_i]
    # Mask is where t=0 AND not shifted
    mask_at_start = (final_t == 0.0) & (~mask_shift_i)
    np.copyto(raw_points, A[final_i], where=mask_at_start[:, None])

    # 5. Result Formatting
    result_dicts: list[dict[tuple, IntersectionPoint]] = [{} for _ in range(len(polylines))]

    # Zip everything. 
    # Note: raw_points contains the coords. 
    # s_idx_i / final_t contain the normalized (seg+1, 0) values.
    result = [[] for _ in range(len(polylines))]

    # Iterate and fill
    # for pt, pi, si, t, pj, sj, u in zip(raw_points, p_idx_i, s_idx_i, final_t, p_idx_j, s_idx_j, final_u):
        
    #     # Create Point Object (shared for both)
    #     # Note: pt is a numpy array view, we usually copy it or keep as array
    #     # tuple(pt) ensures it's hashable/clean
        
    #     # Add to Poly I list
    #     result[pi].append(IntersectionPoint(point=pt, segment_index=si, t=t))
        
    #     # Add to Poly J list
    #     result[pj].append(IntersectionPoint(point=pt, segment_index=sj, t=u))

    # return result
    iterator = zip(
        raw_points, 
        p_idx_i, s_idx_i, final_t, 
        p_idx_j, s_idx_j, final_u
    )
    prev = None
    for pt, pi, si, ti, pj, sj, tj in iterator:
        # Create hashable key for deduplication
        #can still happen due to floating point issues
        if (pi == pj) and abs(si - sj) ==1:
            continue
        # Add for Poly I
        # If the point exists, we overwrite. 
        # Because we normalized boundary conditions, (Seg 0, t=1) became (Seg 1, t=0).
        # Intersection with (Seg 1, t=0) is now an identical record. Dict handles dedup.
        result_dicts[pi][(si, ti)] = IntersectionPoint(point=pt, segment_index=si, t=ti)
        
        # Add for Poly J
        result_dicts[pj][(sj, tj)] = IntersectionPoint(point=pt, segment_index=sj, t=tj)

    result: list[list[IntersectionPoint]] = []
    for d in result_dicts:
        # Convert dict values to list
        poly_intersects = list(d.values())
        # Sort by segment index, then t
        poly_intersects.sort(key=lambda x: (x.segment_index, x.t))
        result.append(poly_intersects)
    
    return result


def compute_intersections_deduped_fast(polylines: list[np.ndarray]) -> list[list[IntersectionPoint]]:
    """
    Computes intersections ensuring consistency at segment endpoints.
    Intersections at endpoints are normalized to t=0 of the subsequent segment.
    """
    
    # 1. Extract Segments
    A, B, poly_ids, seg_ids = _extract_segments(polylines)
    n_segments = len(A)
    
    if n_segments < 2:
        return [[] for _ in polylines]

    # 2. Spatial Indexing (Broad Phase)
    midpoints = (A + B) * 0.5
    segment_lengths = np.linalg.norm(B - A, axis=1)
    if len(segment_lengths) == 0:
        return [[] for _ in polylines]
        
    #max_radius = np.max(segment_lengths) * 0.500001
    tree = cKDTree(midpoints)
    
    #query_radius = max_radius * 2.05

    # Result accumulators across all chunks
    result_dicts: list[dict[tuple, IntersectionPoint]] = [{} for _ in range(len(polylines))]

    EPS = 1e-9
    CHUNK = 5000  # tune based on available memory

    for chunk_start in range(0, n_segments, CHUNK):
        # print(chunk_start, n_segments)
        chunk_end = min(chunk_start + CHUNK, n_segments)
        batch_midpoints = midpoints[chunk_start:chunk_end]
        
        batch_radii = segment_lengths[chunk_start:chunk_end] * 1.05
        neighbors_list = tree.query_ball_point(batch_midpoints, r=batch_radii)
        
        lengths = np.array([len(n) for n in neighbors_list])
        if lengths.sum() == 0:
            continue
        idx_j = np.concatenate(neighbors_list).astype(np.intp)
        idx_i = np.repeat(np.arange(chunk_start, chunk_end, dtype=np.intp), lengths)
        
        # Remove self-pairs only (no i < j — asymmetric radii need both directions)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        
        if len(idx_i) == 0:
            continue

        # 3. AABB Pruning
        A1, B1 = A[idx_i], B[idx_i]
        A2, B2 = A[idx_j], B[idx_j]

        min1 = np.minimum(A1, B1)
        max1 = np.maximum(A1, B1)
        min2 = np.minimum(A2, B2)
        max2 = np.maximum(A2, B2)

        aabb_overlap = (
            (min1[:, 0] <= max2[:, 0]) & (max1[:, 0] >= min2[:, 0]) &
            (min1[:, 1] <= max2[:, 1]) & (max1[:, 1] >= min2[:, 1])
        )
        
        if not np.any(aabb_overlap):
            continue
            
        idx_i = idx_i[aabb_overlap]
        idx_j = idx_j[aabb_overlap]
        A1, B1 = A1[aabb_overlap], B1[aabb_overlap]
        A2, B2 = A2[aabb_overlap], B2[aabb_overlap]

        # Detailed Intersection (Cramer's Rule)
        D1 = B1 - A1
        D2 = B2 - A2
        
        def cross_2d(v1, v2):
            return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        det = cross_2d(D1, D2)
        
        non_parallel = np.abs(det) > EPS
        
        if not np.any(non_parallel):
            continue

        idx_i = idx_i[non_parallel]
        idx_j = idx_j[non_parallel]
        A1, A2 = A1[non_parallel], A2[non_parallel]
        D1, D2 = D1[non_parallel], D2[non_parallel]
        det = det[non_parallel]

        delta_p = A2 - A1
        t_vals = cross_2d(delta_p, D2) / det
        u_vals = cross_2d(delta_p, D1) / det

        # 4. Snap and Filter
        valid_intersections = (
            (t_vals >= -EPS) & (t_vals <= 1.0 + EPS) &
            (u_vals >= -EPS) & (u_vals <= 1.0 + EPS)
        )
        
        final_indices = np.where(valid_intersections)[0]
        
        if len(final_indices) == 0:
            continue

        final_i = idx_i[final_indices]
        final_j = idx_j[final_indices]
        final_t = t_vals[final_indices]
        final_u = u_vals[final_indices]
        
        # Snap to 0/1
        final_t[np.abs(final_t) < EPS] = 0.0
        final_t[np.abs(final_t - 1.0) < EPS] = 1.0
        final_u[np.abs(final_u) < EPS] = 0.0
        final_u[np.abs(final_u - 1.0) < EPS] = 1.0

        p_idx_i = poly_ids[final_i]
        s_idx_i = seg_ids[final_i]
        p_idx_j = poly_ids[final_j]
        s_idx_j = seg_ids[final_j]

        # Filter Neighbors
        is_neighbor = (p_idx_i == p_idx_j) & (np.abs(s_idx_i - s_idx_j) <= 1)
        keep_mask = ~is_neighbor
        
        if not np.any(keep_mask):
            continue

        final_i = final_i[keep_mask]
        final_t = final_t[keep_mask]
        final_u = final_u[keep_mask]
        p_idx_i = p_idx_i[keep_mask]
        s_idx_i = s_idx_i[keep_mask]
        p_idx_j = p_idx_j[keep_mask]
        s_idx_j = s_idx_j[keep_mask]

        # Shift t=1.0 -> t=0.0 of next segment
        mask_shift_i = (final_t == 1.0)
        s_idx_i[mask_shift_i] += 1
        final_t[mask_shift_i] = 0.0
        
        mask_shift_j = (final_u == 1.0)
        s_idx_j[mask_shift_j] += 1
        final_u[mask_shift_j] = 0.0

        # Compute exact points
        raw_points = A[final_i] + (B[final_i] - A[final_i]) * np.where(mask_shift_i[:, None], 1.0, final_t[:, None])
        np.copyto(raw_points, B[final_i], where=mask_shift_i[:, None])
        mask_at_start = (final_t == 0.0) & (~mask_shift_i)
        np.copyto(raw_points, A[final_i], where=mask_at_start[:, None])

        # Accumulate into result_dicts
        iterator = zip(
            raw_points, 
            p_idx_i, s_idx_i, final_t, 
            p_idx_j, s_idx_j, final_u
        )

        for pt, pi, si, ti, pj, sj, tj in iterator:
            # Create hashable key for deduplication
            #can still happen due to floating point issues
            if (pi == pj) and abs(si - sj) ==1:
                continue
            result_dicts[pi][(si, ti)] = IntersectionPoint(point=pt, segment_index=si, t=ti)
            result_dicts[pj][(sj, tj)] = IntersectionPoint(point=pt, segment_index=sj, t=tj)

    # Final sort
    result: list[list[IntersectionPoint]] = []
    for d in result_dicts:
        poly_intersects = list(d.values())
        poly_intersects.sort(key=lambda x: (x.segment_index, x.t))
        result.append(poly_intersects)
    
    return result

def all_ipoints(res: list[list[IntersectionPoint]]):
    return  np.array(list(set(tuple(x.point) for i in res for x in i)))

def points_flattened(points):
    return np.array(list(set(tuple(x) for i in points for x in i)))

def all_ipoints_set(res: list[list[IntersectionPoint]]):
    return  set(tuple(x.point) for i in res for x in i)

def points_to_line_mp(res: list[list[IntersectionPoint]]):
    mp = {}
    for i, r in enumerate(res):
        for ip in r:
            mp.setdefault(tuple(ip.point), []).append(i)
    return mp

def filter_endpoint_connections(
    intersections: list[list[IntersectionPoint]],
    polylines: list[np.ndarray],
) -> list[list[IntersectionPoint]]:
    """
    Removes intersections that are simple endpoint-to-endpoint connections
    between exactly 2 polylines.
    """
    
    point_to_entries: dict[tuple, list[tuple[int, IntersectionPoint]]] = defaultdict(list)
    
    for poly_idx, poly_ints in enumerate(intersections):
        for ip in poly_ints:
            key = tuple(ip.point)
            point_to_entries[key].append((poly_idx, ip))
    
    # Find which points to remove
    remove_points: set[tuple] = set()
    
    for pt_key, entries in point_to_entries.items():
        poly_indices = list(e[0] for e in entries)
        if len(poly_indices) != 2:
            continue
        
        is_endpoint_connection = True
        for poly_idx, ip in entries:
            n_segs = len(polylines[poly_idx]) - 1
            at_start = (ip.segment_index == 0 and ip.t == 0.0)
            at_end = (ip.segment_index == n_segs)
            if not (at_start or at_end):
                is_endpoint_connection = False
                break
        
        if is_endpoint_connection:
            remove_points.add(pt_key)
    
    # Filter
    result = []
    for poly_ints in intersections:
        result.append([ip for ip in poly_ints if tuple(ip.point) not in remove_points])
    
    return result

def split_up(polyline, ipoints: list[IntersectionPoint]):
    
    ipoints.sort(key=lambda x: (x.segment_index, x.t))
    sublines = []
    prev = 0
    prevpoint=None
    for ip in ipoints:
        if  (tuple(ip.point) ==  tuple(polyline[0]) and ip.segment_index==0) or (
            tuple(ip.point) ==  tuple(polyline[-1]) and ip.segment_index == len(polyline)-1):
            continue
        tup = (np.expand_dims(prevpoint, axis=0), 
               polyline[prev:ip.segment_index+1], 
               np.expand_dims(ip.point, axis=0)) if prevpoint is not None else (
            polyline[prev:ip.segment_index+1], 
            np.expand_dims(ip.point, axis=0))
        subline = np.concat(tup)
        sublines.append(subline)
        prev = ip.segment_index+1
        prevpoint= ip.point
    
    tup = (np.expand_dims(prevpoint, axis=0), 
               polyline[prev:]) if prevpoint is not None else (
            polyline[prev:],)
        #print([d.shape for d in tup])
    subline = np.concat(tup)
    sublines.append(subline)
    return sublines



def consecutive_dist(polyline: np.ndarray) -> np.ndarray:
    diffs = polyline[1:] - polyline[:-1] 
    # shape: (N-1,)
    return np.linalg.norm(diffs, axis=1)





def propagate_intersections_multi(
    polylines: list[np.ndarray], 
    intersections: list[list['IntersectionPoint']], 
    d: float,
    no_endpoints = True
) -> list[np.ndarray]:
    """
    Walks along multiple polylines starting from intersection points.
    Marks a point if there is a 'foreign' part of ANY polyline within distance d.
    
    A 'foreign' part is defined as:
    1. A point belonging to a DIFFERENT polyline within distance d.
    2. A point on the SAME polyline within distance d that cannot be reached 
       geodesically (walking along segments) without leaving the sphere of radius d.
    
    Returns:
        A list of lists. result[i] contains the list of marked index chunks 
        (np.ndarrays) for polylines[i].
    """
    
    # 1. Preprocessing & Global Indexing
    if not polylines:
        return []

    # Stack all points to build a single global spatial tree
    # We need to map global_index -> (line_index, point_index)
    all_points_list = []
    global_to_line_idx = []
    global_to_local_idx = []
    
    # Pre-calculate lengths and intersection sets for fast lookup
    Ns = []
    inter_sets = []
    
    for i, poly in enumerate(polylines):
        n_points = len(poly)
        Ns.append(n_points)
        
        all_points_list.append(poly)
        
        # Create mapping arrays
        global_to_line_idx.append(np.full(n_points, i, dtype=int))
        global_to_local_idx.append(np.arange(n_points, dtype=int))
        
        # Store intersection segment indices for this line in a set for O(1) lookup
        # intersections[i] contains IntersectionPoints for polylines[i]
        inter_sets.append({x.segment_index for x in intersections[i]})

    all_points = np.vstack(all_points_list)
    global_line_map = np.concatenate(global_to_line_idx)
    global_local_map = np.concatenate(global_to_local_idx)

    # 2. Build Spatial Tree
    tree = cKDTree(all_points)
    
    # 3. BFS Initialization
    # Visited state: list of boolean arrays matching input structure
    visited = [np.zeros(n, dtype=bool) for n in Ns]
    # Marked indices: list of lists to store results per polyline
    marked_indices = [[] for _ in range(len(polylines))]
    
    queue = deque()
    # Seed with intersection points for all polylines
    for line_idx, inters_in_line in enumerate(intersections):
        N = Ns[line_idx]
         
        for inter in inters_in_line:
            if no_endpoints and ((inter.segment_index ==0 and inter.t ==0) or inter.segment_index ==N-1):
                continue
            seeds = [inter.segment_index, inter.segment_index + 1]
            for seed in seeds:
                if 0 <= seed <= N -1 and not visited[line_idx][seed]:
                    visited[line_idx][seed] = True
                    queue.append((line_idx, seed))

    # 4. Process Queue
    while queue:
        curr_line_idx, curr_pt_idx = queue.popleft()
        
        current_poly = polylines[curr_line_idx]
        center_pt = current_poly[curr_pt_idx]
        N = Ns[curr_line_idx]
        current_inters = inter_sets[curr_line_idx]

        # --- STEP A: Determine Local Reachability (Same Polyline) ---
        # "Geodesic" walk on the current polyline
        L = curr_pt_idx
        R = curr_pt_idx
        
        # Walk Left
        while L > 0 and L not in current_inters:
            dist = np.linalg.norm(current_poly[L - 1] - center_pt)
            if dist <= d:
                L -= 1
            else:
                break
                
        # Walk Right
        while R < N - 1 and R not in current_inters:
            dist = np.linalg.norm(current_poly[R + 1] - center_pt)
            if dist <= d:
                R += 1
            else:
                break
        
        # L = max(0, L-1)
        # R = min(N-1, R+1)

        # --- STEP B: Check for Aliens (Global Tree Query) ---
        neighbor_global_indices = tree.query_ball_point(center_pt, r=d)
        
        is_foreign = False
        
        if neighbor_global_indices:
            neighbor_global_indices = np.array(neighbor_global_indices)
            
            # Map global indices back to (line, local)
            neigh_lines = global_line_map[neighbor_global_indices]
            
            # Condition 1: distinct polyline point within region
            # If any neighbor is on a different line, mark immediately
            if np.any(neigh_lines != curr_line_idx):
                is_foreign = True
            else:
                # Condition 2: Same polyline, but disjoint part
                # Filter to get local indices of neighbors on the SAME line
                mask_same_line = (neigh_lines == curr_line_idx)
                neigh_locals = global_local_map[neighbor_global_indices[mask_same_line]]
                
                min_neighbor = np.min(neigh_locals)
                max_neighbor = np.max(neigh_locals)
                
                if min_neighbor < L or max_neighbor > R:
                    is_foreign = True

        if is_foreign:
            marked_indices[curr_line_idx].append(curr_pt_idx)
            
            # Propagate BFS along the current polyline
            next_steps = []
            if curr_pt_idx > 0: next_steps.append(curr_pt_idx - 1)
            if curr_pt_idx < N - 1: next_steps.append(curr_pt_idx + 1)
            
            for step in next_steps:
                if not visited[curr_line_idx][step]:
                    visited[curr_line_idx][step] = True
                    queue.append((curr_line_idx, step))

    # 5. Format Output
    final_output = []
    for line_indices in marked_indices:
        if not line_indices:
            final_output.append(np.array([], dtype=int))
            continue
            
        arr = np.array(line_indices)
        arr.sort()
        # split_locs = np.where(np.diff(arr) != 1)[0] + 1
        # subarrays = np.split(arr, split_locs)
        final_output.append(arr)

    return final_output



def find_proximity_points(
    polylines: list[np.ndarray], 
    ignore_indices: list[np.ndarray], 
    d: float
) -> list[list[int]]:
    """
    Identifies points across multiple polylines that are close to 'non-local' geometry.
    
    Args:
        polylines: List of (N, 2) arrays.
        ignore_indices: List of lists, where ignore_indices[i] contains indices 
                        on the i-th polyline that should be treated as invisible.
        d: The search radius.
        
    Returns:
        A list of lists, where result[i] contains the sorted indices of marked points 
        for the i-th polyline.
    """
    
    # 1. Flatten all points for the KDTree
    # We need to map the flat index back to (poly_index, point_index)
    
    all_points = []
    flat_to_poly = []  # Maps flat_idx -> poly_index
    flat_to_local = [] # Maps flat_idx -> point_index within poly
    
    # Convert ignore lists to sets for O(1) lookup
    ignore_sets = [set(idxs) for idxs in ignore_indices]
    
    current_flat_idx = 0
    # Store ranges to map back later if needed, or just iterate structure
    poly_starts = []
    
    for p_idx, poly in enumerate(polylines):
        n_points = len(poly)
        all_points.append(poly)
        
        # Create mapping arrays
        # np.full is faster than list comprehensions for large arrays
        flat_to_poly.append(np.full(n_points, p_idx, dtype=np.int32))
        flat_to_local.append(np.arange(n_points, dtype=np.int32))
        
        poly_starts.append(current_flat_idx)
        current_flat_idx += n_points

    if not all_points:
        return []

    # Concatenate for vectorized operations
    points_flat = np.vstack(all_points)
    map_poly = np.concatenate(flat_to_poly)
    map_local = np.concatenate(flat_to_local)
    
    # 2. Build Spatial Tree
    tree = cKDTree(points_flat)
    
    results = [[] for _ in polylines]
    
    # 3. Iterate through every point of every polyline
    # We query the tree for each point.
    
    # query_ball_point can be vectorized (pass all points at once), 
    # returning a customized array of lists (object array).
    # This is usually faster than a Python loop for the query part.
    neighbors_list = tree.query_ball_point(points_flat, r=d)
    
    for flat_idx, neighbors in enumerate(neighbors_list):
        if not neighbors:
            continue
            
        curr_polyline_idx = map_poly[flat_idx]
        curr_l_idx = map_local[flat_idx]
        
        if curr_l_idx in ignore_sets[curr_polyline_idx]:
            continue
        
        # Determine strict local neighbors
        # We need to filter neighbors based on the 'ignore' list and 'local path' logic
        
        same_line_neighbors = []
        found_foreign = False
        
        for n_flat in neighbors:
            n_p_idx = map_poly[n_flat]
            n_l_idx = map_local[n_flat]
            
            # Check Ignore List
            if n_l_idx in ignore_sets[n_p_idx]:
                continue
                
            if n_p_idx != curr_polyline_idx:
                # Found a valid point on a DIFFERENT polyline
                found_foreign = True
                break
            else:
                # Same polyline
                same_line_neighbors.append(n_l_idx)
        
        if found_foreign:
            results[curr_polyline_idx].append(int(curr_l_idx))
            continue
            
        # If we are here, we only have neighbors on the same polyline.
        # Logic: "ignore points that form an +1 increasing sequence that includes the query point index"
        
        if not same_line_neighbors:
            # Only happened if the point itself was in the ignore list (unlikely based on usage)
            # or simply no neighbors found after filtering.
            continue
            
        # Sort indices to find the sequence
        same_line_neighbors.sort()
        
        # Find the query point in this list
        # It should be there unless it was ignored, but let's handle safety
        try:
            center_pos = same_line_neighbors.index(curr_l_idx)
        except ValueError:
            print("problem: query point itself not in result")
            exit()

        # Expand outwards from center_pos to define the "Local Path"
        # We count how many points form the continuous sequence
        seq_start = center_pos
        seq_end = center_pos
        
        # Walk Left
        while seq_start > 0:
            if same_line_neighbors[seq_start - 1] == same_line_neighbors[seq_start] - 1:
                seq_start -= 1
            else:
                break
        
        # Walk Right
        while seq_end < len(same_line_neighbors) - 1:
            if same_line_neighbors[seq_end + 1] == same_line_neighbors[seq_end] + 1:
                seq_end += 1
            else:
                break
                
        # The indices from seq_start to seq_end (inclusive) are the "local path".
        # If the list contains ANY index outside this range, mark the point.
        
        local_path_count = (seq_end - seq_start + 1)
        total_neighbors = len(same_line_neighbors)
        
        if total_neighbors > local_path_count:
            results[curr_polyline_idx].append(int(curr_l_idx))

    return results


def find_proximity_points_detailed(
    polylines: list[np.ndarray],
    ignore_indices: list[np.ndarray],
    d: float
) -> list[dict[int, dict[int, int]]]:
    """
    Like find_proximity_points, but returns for each polyline a dict mapping
    point_index -> dict of {line_index: closest_point_index}
    for the nearby non-local geometry.

    Args:
        polylines: List of (N, 2) arrays.
        ignore_indices: List of arrays, where ignore_indices[i] contains indices
                        on the i-th polyline that should be treated as invisible.
        d: The search radius.

    Returns:
        A list of dicts, where result[i] maps point_index -> {line_index: closest_point_index}.
    """

    all_points = []
    flat_to_poly = []
    flat_to_local = []

    ignore_sets = [set(idxs) for idxs in ignore_indices]

    current_flat_idx = 0
    poly_starts = []

    for p_idx, poly in enumerate(polylines):
        n_points = len(poly)
        all_points.append(poly)
        flat_to_poly.append(np.full(n_points, p_idx, dtype=np.int32))
        flat_to_local.append(np.arange(n_points, dtype=np.int32))
        poly_starts.append(current_flat_idx)
        current_flat_idx += n_points

    if not all_points:
        return []

    points_flat = np.vstack(all_points)
    map_poly = np.concatenate(flat_to_poly)
    map_local = np.concatenate(flat_to_local)

    tree = cKDTree(points_flat)

    results = [dict() for _ in polylines]

    neighbors_list = tree.query_ball_point(points_flat, r=d)

    for flat_idx, neighbors in enumerate(neighbors_list):
        if not neighbors:
            continue

        curr_polyline_idx = map_poly[flat_idx]
        curr_l_idx = map_local[flat_idx]
        curr_point = points_flat[flat_idx]

        if curr_l_idx in ignore_sets[curr_polyline_idx]:
            continue

        # For foreign lines, track the closest point per line
        foreign_closest = {}  # line_index -> (dist, point_index)
        same_line_neighbors = []

        for n_flat in neighbors:
            n_p_idx = int(map_poly[n_flat])
            n_l_idx = int(map_local[n_flat])

            if n_l_idx in ignore_sets[n_p_idx]:
                continue

            if n_p_idx != curr_polyline_idx:
                dist = np.linalg.norm(points_flat[n_flat] - curr_point)
                if n_p_idx not in foreign_closest or dist < foreign_closest[n_p_idx][0]:
                    foreign_closest[n_p_idx] = (dist, n_l_idx)
            else:
                same_line_neighbors.append(n_l_idx)

        nearby_dict = {li: info[1] for li, info in foreign_closest.items()}

        # Check same-line non-local neighbors
        if same_line_neighbors:
            same_line_neighbors.sort()
            try:
                center_pos = same_line_neighbors.index(curr_l_idx)
            except ValueError:
                print("problem: query point itself not in result")
                exit()

            seq_start = center_pos
            seq_end = center_pos

            while seq_start > 0:
                if same_line_neighbors[seq_start - 1] == same_line_neighbors[seq_start] - 1:
                    seq_start -= 1
                else:
                    break

            while seq_end < len(same_line_neighbors) - 1:
                if same_line_neighbors[seq_end + 1] == same_line_neighbors[seq_end] + 1:
                    seq_end += 1
                else:
                    break

            local_path_count = (seq_end - seq_start + 1)
            if len(same_line_neighbors) > local_path_count:
                # Find closest non-local same-line point
                non_local = [idx for idx in same_line_neighbors
                             if idx < same_line_neighbors[seq_start] or idx > same_line_neighbors[seq_end]]
                if non_local:
                    dists = [np.linalg.norm(polylines[curr_polyline_idx][idx] - curr_point) for idx in non_local]
                    best = non_local[np.argmin(dists)]
                    nearby_dict[int(curr_polyline_idx)] = best

        if nearby_dict:
            results[curr_polyline_idx][int(curr_l_idx)] = nearby_dict

    return results



def get_self_loop_indices_stack(polyline, intersections: list[IntersectionPoint]):
    """
    Computes indices of points inside self-loops using a linear traversal and stack.
    
    Args:
        polyline: N x 2 numpy array.
        intersections: List of IntersectionPoint namedtuples.
        
    Returns:
        list[tuple[np.ndarray, bool]]: List of (indices, is_complex) tuples.
    """

    # We create a list of events: (linear_position, intersection_id)
    i_sorted = sorted(intersections, key=lambda x : (x.segment_index, x.t))
    
    pt_map = {}
    results = []
    n_pushes = 0
    n_pops = 0
    for ip in i_sorted:
        
        if tuple(ip.point) not in pt_map:
            n_pushes+=1
            pt_map[tuple(ip.point)] = (ip.segment_index, ip.t, len(results), n_pushes, n_pops)
        else:
            prev, t, size, prev_pushes, prev_pops = pt_map[tuple(ip.point)]
            count = len(results)-size
            left, right = np.ceil(prev+t), np.floor( ip.segment_index +ip.t)
            results.append((np.arange(left, right+1, dtype=int), 
                            count, n_pushes-prev_pushes))
            n_pops+=1
            # pt_map[tuple(ip.point)] = (ip.segment_index, ip.t, len(results), n_pushes, n_pops)
    
    return results
        




def get_max_inscribed_circle(poly_array, tolerance=None):
    """
    Finds the maximum inscribed circle of a polygon.
    
    Parameters:
    - poly_array: N, 2 numpy array of vertex coordinates.
    - tolerance: (Optional) The algorithm stops when the search area is 
                 smaller than this. Defaults to max(width, height) / 1000.
                 
    Returns:
    - center: numpy array [x, y]
    - radius: float
    """
    # 1. Create the Shapely polygon
    poly = Polygon(poly_array)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    # 2. Compute the maximum inscribed circle
    # Returns a LineString(center_point, boundary_point)
    result_line = maximum_inscribed_circle(poly, tolerance=tolerance)
    
    # 3. Extract points and calculate radius
    center = np.array(result_line.coords[0])
    
    radius = result_line.length # Length of the line connecting center to boundary
    
    return center, radius

def get_red_black_map(pl, indexes: np.ndarray, intersections: list[IntersectionPoint]) -> dict[int, int]:
    
    rb = np.zeros(len(pl), dtype=int)
    rb[indexes] = 1
    inter_indexes = np.array([x.segment_index for x in intersections], dtype=int)
    rb[inter_indexes] = 2
    return {i: int(val) for i, val in enumerate(rb)}
    
def get_trapped_red_nodes(nodes: dict[int, int]) -> set[int]:
    '''
    Remove a contiguous run of Red nodes (1) if it is immediately bounded by Black nodes (2)
    or the sequence boundaries on both sides, without encountering a 0.
    '''
    # 0 = Neutral (Stops removal)
    # 1 = Red (Target for removal)
    # 2 = Black (Boundaries)

    
    indices_to_remove = set()
    current_red_run = []
    
    # We treat the "Start" of the sequence as a Black node (2).
    previous_boundary_was_black = True
    
    for idx in nodes:
        val = nodes[idx]
        
        if val == 1:
            # Accumulate red nodes in the current run
            current_red_run.append(idx)
        
        elif val == 2:
            # Found a Black node (Right Boundary)
            # If the Left Boundary was also Black (or Start), remove the trapped reds.
            if previous_boundary_was_black:
                indices_to_remove.update(current_red_run)
            
            # Reset run
            current_red_run = []
            # This node becomes the new Left Boundary for the next segment
            previous_boundary_was_black = True
            
        elif val == 0:
            # Any reds currently accumulated are safe because they touch a 0.
            current_red_run = []
            # This node becomes the Left Boundary, but it is NOT Black.
            previous_boundary_was_black = False

    # Check End of List condition
    # We treat the "End" of the sequence as a Black node.
    if current_red_run and previous_boundary_was_black:
        indices_to_remove.update(current_red_run)

    # Return a new dictionary without the removed nodes
    # return {k: v for k, v in nodes.items() if k not in indices_to_remove}
    return indices_to_remove

def detect_trapped_red_nodes(nodes: dict[int, int]) -> bool:
    '''
    detects if there is a contiguous run of Red nodes (1) if it is immediately bounded by Black nodes (2)
    or the sequence boundaries on both sides, without encountering a 0.
    '''
    # 0 = Neutral (Stops removal)
    # 1 = Red (Target for removal)
    # 2 = Black (Boundaries)

    
    indices_to_remove = set()
    current_red_run = []
    
    # We treat the "Start" of the sequence as a Black node (2).
    previous_boundary_was_black = True
    
    for idx in nodes:
        val = nodes[idx]
        
        if val == 1:
            # Accumulate red nodes in the current run
            current_red_run.append(idx)
        
        elif val == 2:
            # Found a Black node (Right Boundary)
            # If the Left Boundary was also Black (or Start), remove the trapped reds.
            if previous_boundary_was_black:
                print("true 1")
                return True
            
            # Reset run
            current_red_run = []
            # This node becomes the new Left Boundary for the next segment
            previous_boundary_was_black = True
            
        elif val == 0:
            # Any reds currently accumulated are safe because they touch a 0.
            current_red_run = []
            # This node becomes the Left Boundary, but it is NOT Black.
            previous_boundary_was_black = False

    # Check End of List condition
    # We treat the "End" of the sequence as a Black node.
    if current_red_run and previous_boundary_was_black:
        print("true 2")
        return True
    return False


def get_trapping_black_indices_old(nodes: dict[int, int]) -> list[int]:
    '''
    Identifies Black nodes (2) that are bounding a contiguous run of Red nodes (1).
    Returns a sorted list of indices of these Black nodes.
    
    Logic:
    - 0 (Neutral): Breaks the trap.
    - 1 (Red): The items being trapped.
    - 2 (Black): The walls.
    - Start/End of list: Treated as implicit Black nodes.
    '''
    
    bounding_indices = set()
    
    # Ensure we iterate in spatial order based on index
    sorted_indices = sorted(nodes.keys())
    
    # State tracking
    has_red_nodes = False
    
    # We treat the "Start" of the sequence as a Black node (2).
    # Since it's virtual, the index is None.
    left_boundary_is_black = True
    left_boundary_index = None
    
    for idx in sorted_indices:
        val = nodes[idx]
        
        if val == 1:
            # We have found red nodes in this segment
            has_red_nodes = True
            
        elif val == 2:
            # Found a Black node (Right Boundary)
            # If the Left Boundary was also Black (or Start) AND we saw reds...
            if left_boundary_is_black and has_red_nodes:
                # Add the current node (The Right Wall)
                bounding_indices.add(idx)
                # Add the previous node (The Left Wall), if it exists.
                # If left_boundary_index is None, the wall was the "Start" of list (virtual).
                if left_boundary_index is not None:
                    bounding_indices.add(left_boundary_index)
            
            # Reset state: This node now becomes the Left Boundary for the next segment
            left_boundary_is_black = True
            left_boundary_index = idx
            has_red_nodes = False
            
        elif val == 0:
            # Found a Neutral node. The trap is broken.
            # Any subsequent reds are not trapped by the previous black node.
            left_boundary_is_black = False
            left_boundary_index = idx
            has_red_nodes = False

    # Check End of List condition
    # We treat the "End" of the sequence as a Black node.
    # If we have an open run of reds bounded by a Black node on the left...
    if left_boundary_is_black and has_red_nodes:
        # The left boundary effectively trapped reds against the end of the list.
        if left_boundary_index is not None:
            bounding_indices.add(left_boundary_index)
            
    return sorted(list(bounding_indices))

def get_trapping_black_indices(nodes: dict[int, int]) -> list[list[int]]:
    '''
    Identifies Black nodes (2) that are bounding a contiguous run of Red nodes (1).
    Also reports consecutive Black nodes (2, 2) as boundaries.

    The start and end of the sequence act as implicit Black boundaries
    (i.e., red nodes at the very beginning or end are considered trapped
    against the border).

    Returns a list of groups, where each group is a sorted list of trapping Black
    node indices. Groups are separated by Neutral nodes (0): when a neutral node
    is encountered, the current group is flushed and a new one begins.
    '''

    sorted_indices = sorted(nodes.keys())

    segments = []  # list of groups of trapping black indices
    current_segment_blacks = []  # black indices accumulated in current group
    has_red_nodes = False
    # Start of sequence is an implicit Black boundary
    left_boundary_is_black = True
    left_boundary_index = None

    def flush_segment():
        nonlocal current_segment_blacks
        if current_segment_blacks:
            segments.append(sorted(current_segment_blacks))
        current_segment_blacks = []

    for idx in sorted_indices:
        val = nodes[idx]

        if val == 1:
            has_red_nodes = True

        elif val == 2:
            # Consecutive if previous boundary was adjacent, or if this is
            # the first node (consecutive with implicit start boundary)
            is_consecutive = (left_boundary_index is not None and left_boundary_index == idx - 1) \
                          or (left_boundary_index is None and idx == sorted_indices[0])

            if left_boundary_is_black and (has_red_nodes or is_consecutive):
                current_segment_blacks.append(idx)
                if left_boundary_index is not None:
                    if left_boundary_index not in current_segment_blacks:
                        current_segment_blacks.append(left_boundary_index)

            left_boundary_is_black = True
            left_boundary_index = idx
            has_red_nodes = False

        elif val == 0:
            # Neutral node: reds are NOT trapped on this side, so do NOT
            # add the left boundary. Just flush the current group and reset.
            flush_segment()
            left_boundary_is_black = False
            left_boundary_index = idx
            has_red_nodes = False

    # End of sequence acts as an implicit Black boundary.
    # If there are red nodes trapped between the last black node and the end, mark it.
    # Also, if the last node is black, it's consecutive with the implicit end boundary.
    if left_boundary_is_black:
        is_last_black = (left_boundary_index is not None
                         and left_boundary_index == sorted_indices[-1]
                         and nodes[left_boundary_index] == 2)
        if has_red_nodes or is_last_black:
            if left_boundary_index is not None:
                if left_boundary_index not in current_segment_blacks:
                    current_segment_blacks.append(left_boundary_index)

    flush_segment()

    return segments

def split_unit_progressions(arr: np.ndarray):
    split_locs = np.where(np.diff(arr) != 1)[0] + 1
    return np.split(arr, split_locs)

    
    
    