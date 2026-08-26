from typing import TypeVar, Callable, Optional
import numpy as np
import sampler.intersections as intersections
import sampler.randompath as randompath
import sampler.triangulations as triangulations
from pprint import pprint
from .stars import generate_random_star
# Define a generic type T for the samples
T = TypeVar('T')

def sample_with_backtracking(
    sampler_func: Callable[[], T],
    validate_sample_fn: Callable[[T], bool],
    validate_batch_fn: Callable[[list[T]], bool],
    n: int,
    max_tries: int = 100
) -> list[T]:
    """
    Generates a list of n samples using rejection sampling with backtracking.
    
    If the function cannot find a valid next sample within max_tries, it removes
    the previously added sample and tries to find a new path from there.

    Args:
        sampler_func: Function that returns a single candidate sample.
        validate_sample_fn: Function (sample) -> bool checking if a sample is valid in isolation.
        validate_batch_fn: Function (current_batch with new_sample) -> bool checking if 
                           adding the sample maintains batch validity (e.g., no duplicates).
        n: The target number of samples.
        max_tries: Maximum attempts to fill a specific slot before backtracking.

    Returns:
        A list of n valid samples.

    Raises:
        RuntimeError: If a valid batch cannot be generated within constraints.
    """
    
    # Stack to track how many attempts we've made for the current index (len(batch))
    # We initialize with 0 attempts for index 0.
    batch: list[T] = []
    attempts_stack: list[int] = [0]
    while len(batch) < n:
        current_index = len(batch)
        
        # Check if we have exhausted tries for the current slot
        if attempts_stack[current_index] >= max_tries:
            # If we are at the very first index and failed, we can't solve it.
            if current_index == 0:
                raise RuntimeError(
                    f"Failed to generate {n} samples. Exhausted max_tries at index 0."
                )
            # BACKTRACKING LOGIC:
            # 1. Pop the last successfully added item (it led to a dead end).
            batch.pop()
            print("backtracking to", len(batch))
            
            # 2. Pop the attempt counter for the level we are leaving.
            attempts_stack.pop()
            
            # 3. Increment the attempt counter for the PREVIOUS level.
            #    (We are treating the popped item as a 'failed attempt' for that previous level).
            attempts_stack[-1] += 1
            
            continue

        # Try to generate a candidate
        candidate = sampler_func()
        attempts_stack[current_index] += 1

        # 1. Validate the sample in isolation
        if not validate_sample_fn(candidate):
            continue

        batch.append(candidate)
        # 2. Validate the sample against the current batch
        if not validate_batch_fn(batch):
            batch.pop()
            continue

        # If successful:
        
        # Prepare the attempt counter for the NEXT slot (initialize to 0)
        attempts_stack.append(0)

    return batch



def get_min_distance(points: set[tuple[int, int]]) -> float:
    if len(points) < 2:
        return float('inf')

    coords = np.array(list(points))

    # 1. Create a distance matrix using broadcasting (N x N matrix)
    # shape: (N, 1, 2) - (1, N, 2) results in (N, N, 2)
    deltas = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # 2. Calculate Euclidean distance
    # shape: (N, N)
    dist_matrix = np.sqrt(np.sum(deltas**2, axis=-1))

    # 3. Handle the diagonal
    # The distance from a point to itself is 0. We want the min NON-ZERO distance.
    # We set the diagonal to infinity so it doesn't get picked as the minimum.
    np.fill_diagonal(dist_matrix, np.inf)

    return np.min(dist_matrix)


def validate(pl: np.ndarray,
                   min_dist=1.1) -> tuple[bool, dict]:

    inters = intersections.compute_intersections([pl])[0]

    self_loops = intersections.get_self_loop_indices_stack(pl, inters)
    for idx, counts1, counts2 in self_loops:
        if counts2 ==0:
            loop_points = pl[idx]
            if len(loop_points) < 3:
                return False, {}
            center, r = intersections.get_max_inscribed_circle(loop_points)
            if r <= min_dist/2:
                return False, {"small_loop": (center, r)}


    indexes = intersections.propagate_intersections_multi([pl], [inters], min_dist)[0]
    if len(indexes) and (indexes[0] == 0 or indexes[-1]==len(pl)-1):
        return False, {"invalid_endpoints": indexes}
    # rb = get_red_black_map(pl, indexes, inters)
    # if detect_trapped_red_nodes(rb):
    #     return False, {"close_lines": indexes}

    return True, {}


def validate_multiple(lines: list[np.ndarray], min_dist=1.1, exclude_set: set[tuple] | None = None) -> tuple[bool, dict]:
    if len(lines) <= 1:
        return True, {}
    res = intersections.compute_intersections_deduped(lines)
    all_p = intersections.all_ipoints_set(res)
    
    if exclude_set is not None:
        res = [[x for x in i if tuple(x.point) not in exclude_set] for i in res]
    # ipoint_set = set(tuple(x.point) for i in res for x in i)
    # if get_min_distance(ipoint_set) < min_dist:
    #     return False, {"reason": np.array(list(ipoint_set))}
    
    #it could still happen that we have short branches
    indexes_l = intersections.propagate_intersections_multi(lines, res, min_dist, no_endpoints=True)
    for i, indexes, line, inters in zip(range(len(lines)), indexes_l, lines, res):
        rb = intersections.get_red_black_map(line, indexes, inters)
        tmp= intersections.get_trapping_black_indices(rb)
        if len(tmp):
            return False, {"close_lines": (i, np.array(tmp))}
        # if len(indexes) and (indexes[0] == 0 or indexes[-1]==len(line)-1):
        #     return False, {"invalid_endpoints": indexes}
    
    return True, {}
     
     

def sample_curves(n_curves, n_steps, min_dist, dim=500, batches = list()):
    assert n_curves > 0
    
    sampler_fn = lambda : randompath.densify_polyline_maxdist(
        randompath.fractal_random_walk(n_steps, desired_dim=dim),
                                                         max_dist=min_dist/2)
    validator_fn = lambda x: validate(x, min_dist+0.001)[0]
    batch_validator = lambda x : validate_multiple(x, min_dist)[0]
    return sample_with_backtracking(sampler_fn, validator_fn, batch_validator,
                                    n_curves, max_tries=50)
    # while True:      
    #     res = False
    #     while not res:
    #         print("sampling curve...")
    #         pl = randompath.fractal_random_walk(n_steps, desired_dim=dim)
    #         pl_d = intersections.densify_polyline(pl, max_dist=min_dist/2)
    #         assert max(intersections.consecutive_dist(pl_d)) <= min_dist/2
    #         res, _ = validate(pl_d)
        
    #     lines.append(pl_d)
    #     r, _ = validate_multiple(lines, min_dist)
    #     if not r:
    #         lines.pop()
    #     if len(lines) == n_curves:
    #         break 
            
    # return lines


def detect_false_positives(lines, inters, pointset_start, min_dist):
    indexes_l = intersections.propagate_intersections_multi(lines, inters, min_dist)
    bad_points = set()
    all_p = intersections.all_ipoints_set(inters)
    mypoints = []
    red_points = []
    bad_point_counts = {}
    point_mp = intersections.points_to_line_mp(inters)
    for i, indexes, line, inters in zip(range(len(lines)), indexes_l, lines, inters):
        mp = {}
        [mp.setdefault(x.segment_index, []).append(x.point) for x in inters]
        rb = intersections.get_red_black_map(line, indexes, inters)
        bl_idx = intersections.get_trapping_black_indices(rb)
        # i_red = intersections.get_trapped_red_nodes(rb)
        for k in indexes:
            red_points.append(tuple(line[k]))
        for bi in bl_idx:
            # red_points.add(tuple(line[i]))
            for p in mp[bi]:
                # bad_points.add(tuple(p))
                bad_point_counts[tuple(p)] = bad_point_counts.setdefault(tuple(p), 0) + 1
        for u in inters:
            #print(u, len(line))
            mypoints.append(tuple(line[u.segment_index]))
            
    for p in bad_point_counts.keys():
        if len(point_mp[p]) == bad_point_counts[p]:
            bad_points.add(p)
    bad_points.difference_update(pointset_start)
    return bad_points

def sample_tri_method(width, height, n_points=50, closeness= 10.0, n_add=0, min_dist=2):
    MAX_TRIES = 1000
    lines = []
    debug_points = []
    red_points = []
    while True:
        print("sampling main lines")
        lines, pset, stats = triangulations.sample_splinepaths(
            n_points, width, height,  closeness, delete_edges_frac=0.3)
        # lines = lines[:2]
        res = intersections.compute_intersections_deduped(lines)
        indexes_l = intersections.propagate_intersections_multi(lines, res, d=min_dist)
        for pl, i in enumerate(indexes_l):
            for k in i: red_points.append(lines[pl][k])
        #due to the many tangent lines, some intersections may be false positives.
        # we need to remove them
        bad_points = detect_false_positives(lines, res, pset, min_dist)
        print("validating...")
        r, dbg = validate_multiple(lines, min_dist, exclude_set=bad_points)
        if r:
            break
        red_points = []

    for i in range(n_add):
        for _ in range(MAX_TRIES):
            for _ in range(MAX_TRIES):
                pl = randompath.densify_polyline_maxdist(
                randompath.fractal_random_walk(500, desired_dim=min(width, height)),
                                                                max_dist=min_dist/2)
                if validate(pl)[0]:
                    break
            lines.append(pl)
            r, dbg = validate_multiple(lines, min_dist, exclude_set=bad_points)
            if r:
                break
            lines.pop()
        
        
        
    res = intersections.compute_intersections_deduped(lines)
    res = [[x for x in i if tuple(x.point) not in bad_points] for i in res]
    # st = {p for p in mypoints}
    # print(len(st), len(mypoints))
    # mypoints = np.array(list(mypoints))
    red_points = np.array(list(red_points))
    bad_points = np.array(list(bad_points))
    return lines,  res, red_points, stats

    

def sample_star_method(dim, center_perturb, min_dist, n_add):
    MAX_TRIES = 1000
    polylines = generate_random_star(dim)
    #perturb center
    vec =  np.random.rand(2) * center_perturb
    polylines = [pl + vec for pl in polylines]
    print("trying to add ", n_add)
    for i in range(n_add):
        for _ in range(MAX_TRIES):
            for _ in range(MAX_TRIES):
                pl = randompath.densify_polyline_maxdist(
                randompath.fractal_random_walk(500, desired_dim=dim), max_dist=min_dist/2)
                if validate(pl)[0]:
                    break
            polylines.append(pl)
            r, dbg = validate_multiple(polylines, min_dist)
            if r:
                break
            polylines.pop()
    
    return polylines