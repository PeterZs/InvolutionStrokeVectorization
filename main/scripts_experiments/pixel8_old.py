import numpy as np
import cv2
from collections import defaultdict


def _signed_area(poly: list[tuple[int, int]]) -> float:
    """Shoelace signed area. Positive = CW in image coords (Y-down) = outer boundary."""
    n = len(poly)
    if n == 0:
        return 0.0
    s = 0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2


def component_pixel_polygon(
    mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """
    Given a binary mask (uint8, single 8-connected component, non-zero = foreground),
    returns (outer_polygon, holes) where:
      - outer_polygon: list of (x, y) corner coordinates tracing the outer boundary CW
      - holes: list of polygons (each CCW) tracing enclosed background regions

    With 8-connectivity, diagonal pixel neighbors are considered connected.
    At diagonal-junction corners (hourglass/X junctions), the tracer picks the
    most-clockwise outgoing edge to stay on the current polygon.

    Corner coordinate convention: pixel at grid position (col, row) occupies
    the unit square with corners (col, row) .. (col+1, row+1).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], []

    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    # -------------------------------------------------------------------------
    # Build the directed edge graph.
    #
    # Each boundary edge is a directed segment between two corner points,
    # oriented so that the foreground pixel is on the LEFT of the direction
    # of travel (clockwise exterior winding in image coords, Y-down).
    #
    # For pixel (x, y), its 4 axis-aligned sides produce boundary edges
    # when the axis-aligned neighbor is NOT in the pixel_set.
    #
    # The difference for 8-connectivity: a corner can be the START of
    # two different boundary edges (from two diagonally-touching foreground
    # pixels). We store all outgoing edges and resolve ambiguity during tracing.
    # -------------------------------------------------------------------------

    # Oriented CW (foreground on left):
    #   missing above  → top edge goes RIGHT:    (x,   y  ) → (x+1, y  )
    #   missing right  → right edge goes DOWN:   (x+1, y  ) → (x+1, y+1)
    #   missing below  → bottom edge goes LEFT:  (x+1, y+1) → (x,   y+1)
    #   missing left   → left edge goes UP:      (x,   y+1) → (x,   y  )
    edge_for_missing_neighbor = {
        ( 0, -1): lambda x, y: ((x,   y  ), (x+1, y  )),
        ( 1,  0): lambda x, y: ((x+1, y  ), (x+1, y+1)),
        ( 0,  1): lambda x, y: ((x+1, y+1), (x,   y+1)),
        (-1,  0): lambda x, y: ((x,   y+1), (x,   y  )),
    }

    outgoing: dict[tuple, list[tuple]] = defaultdict(list)

    for (x, y) in pixel_set:
        for (dx, dy), edge_fn in edge_for_missing_neighbor.items():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in pixel_set:
                start, end = edge_fn(x, y)
                outgoing[start].append(end)

    # -------------------------------------------------------------------------
    # Resolve ambiguous corners (diagonal junctions).
    #
    # Policy: at an ambiguous corner, pick the outgoing edge that turns most
    # CLOCKWISE relative to the incoming direction.
    # Most clockwise → most negative cross product: inc × (cand − cur).
    #
    # This correctly separates outer (CW) and hole (CCW) polygons because
    # the two edge types arrive at junction nodes from opposite winding
    # directions, so the most-CW choice consistently follows one polygon.
    # -------------------------------------------------------------------------

    def pick_next(cur: tuple, prev: tuple, candidates: list[tuple]) -> tuple:
        if len(candidates) == 1:
            return candidates[0]
        inc = (cur[0] - prev[0], cur[1] - prev[1])
        def cross(cand):
            d = (cand[0] - cur[0], cand[1] - cur[1])
            return inc[0] * d[1] - inc[1] * d[0]
        return min(candidates, key=cross)

    # -------------------------------------------------------------------------
    # Trace closed polygons by following the edge graph.
    # Each directed edge is consumed exactly once.
    # A junction node with two outgoing edges will be the start of two traces.
    # -------------------------------------------------------------------------
    visited_edges: set[tuple] = set()
    polygons = []

    work_list = list(outgoing.keys())
    work_idx = 0
    while work_idx < len(work_list):
        start_node = work_list[work_idx]
        work_idx += 1

        candidates = outgoing[start_node]
        unvisited = [c for c in candidates if (start_node, c) not in visited_edges]
        if not unvisited:
            continue

        first_next = unvisited[0]
        visited_edges.add((start_node, first_next))

        poly = [start_node]
        prev = start_node
        cur = first_next

        while True:
            if cur == start_node:
                still_avail = [c for c in outgoing[start_node]
                               if (start_node, c) not in visited_edges]
                if not still_avail:
                    break  # true closure
                # Pass through junction in a figure-8
                nxt = pick_next(cur, prev, still_avail)
                visited_edges.add((cur, nxt))
                poly.append(cur)
                prev = cur
                cur = nxt
            else:
                poly.append(cur)
                next_candidates = outgoing.get(cur, [])
                unvisited_next = [c for c in next_candidates if (cur, c) not in visited_edges]
                if not unvisited_next:
                    break  # shouldn't happen for a valid closed boundary
                nxt = pick_next(cur, prev, unvisited_next)
                visited_edges.add((cur, nxt))
                prev = cur
                cur = nxt

        polygons.append(poly)

        still_unvisited = [c for c in outgoing[start_node]
                           if (start_node, c) not in visited_edges]
        if still_unvisited:
            work_list.append(start_node)

    # -------------------------------------------------------------------------
    # Classify polygons by winding direction.
    # Positive shoelace area in Y-down coords = CW = outer boundary.
    # Negative = CCW = hole.
    # A single 8-connected component has exactly one outer polygon.
    # -------------------------------------------------------------------------
    outer = None
    holes = []
    for poly in polygons:
        if _signed_area(poly) >= 0:
            if outer is None or len(poly) > len(outer):
                outer = poly
        else:
            holes.append(poly)

    return (outer or [], holes)


def components_pixel_polygons(
    binary_image: np.ndarray,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Run connected components (8-connectivity) on a binary image and return
    the multipolygon for each component as (outer_polygon, holes).

    outer_polygon: ndarray of shape (N, 2) with (x, y) corner coordinates (CW).
    holes: list of ndarrays, each shape (M, 2), traced CCW.

    An island inside a hole is a separate connected component and appears
    as its own entry in the returned list.
    """
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    result = []
    num_labels = int(labels.max())
    for label in range(1, num_labels + 1):
        component_mask = (labels == label).astype(np.uint8)
        outer, holes = component_pixel_polygon(component_mask)
        outer_arr = np.array(outer) if outer else np.empty((0, 2), dtype=int)
        holes_arr = [np.array(h) for h in holes]
        result.append((outer_arr, holes_arr))
    return result


# =============================================================================
# Tests
# =============================================================================

def make_mask(rows) -> np.ndarray:
    """Build a binary mask from a list of strings ('.' = bg, '#' = fg)."""
    grid = [[1 if c == '#' else 0 for c in row] for row in rows]
    return np.array(grid, dtype=np.uint8)


def test_single_pixel():
    mask = make_mask(['#'])
    outer, holes = component_pixel_polygon(mask)
    assert set(outer) == {(0,0),(1,0),(1,1),(0,1)}, f"Got {outer}"
    assert holes == []
    print("PASS: single pixel")


def test_2x2_square():
    mask = make_mask(['##', '##'])
    outer, holes = component_pixel_polygon(mask)
    assert len(outer) == 8, f"Expected 8 boundary corners, got {len(outer)}: {outer}"
    assert set(outer) == {(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)}, f"Got {outer}"
    assert holes == []
    print("PASS: 2x2 square")


def test_diagonal_two_pixels_8conn():
    """
    Two pixels touching only diagonally — with 8-conn they form ONE component.
    """
    mask = make_mask(['#.', '.#'])
    outer, holes = component_pixel_polygon(mask)
    print(f"  diagonal two pixels polygon: {outer}")
    assert len(outer) == 8, f"Expected 8 corners, got {len(outer)}: {outer}"
    assert holes == []
    print("PASS: diagonal two pixels (8-conn)")


def test_anti_diagonal_two_pixels_8conn():
    mask = make_mask(['.#', '#.'])
    outer, holes = component_pixel_polygon(mask)
    print(f"  anti-diagonal polygon: {outer}")
    assert len(outer) == 8, f"Expected 8 corners, got {len(outer)}: {outer}"
    assert holes == []
    print("PASS: anti-diagonal two pixels (8-conn)")


def test_l_shape():
    mask = make_mask(['#.', '##'])
    outer, holes = component_pixel_polygon(mask)
    assert holes == []
    assert len(outer) > 0
    print("PASS: L-shape")


def test_z_shape_8conn():
    """Z-shape where pixels are only 8-connected (diagonal chain)."""
    mask = make_mask(['#.', '##', '.#'])
    outer, holes = component_pixel_polygon(mask)
    print(f"  Z-shape polygon (8-conn): {outer}")
    assert len(outer) > 0
    assert holes == []
    print("PASS: Z-shape 8-conn")


def test_ring_with_hole():
    """A ring-shaped component must produce exactly one hole polygon."""
    mask = make_mask([
        '#####',
        '#...#',
        '#...#',
        '#...#',
        '#####',
    ])
    outer, holes = component_pixel_polygon(mask)
    print(f"  ring outer length={len(outer)}, hole count={len(holes)}")
    assert len(outer) > 0, "Expected outer polygon"
    assert len(holes) == 1, f"Expected 1 hole, got {len(holes)}"
    assert _signed_area(outer) > 0, "Outer polygon must be CW (positive area)"
    assert _signed_area(holes[0]) < 0, "Hole polygon must be CCW (negative area)"
    print("PASS: ring with hole")


def test_island_inside_hole():
    """
    Outer ring → hole → island ring → inner hole.
    The island is a separate 8-connected component, so components_pixel_polygons
    returns two entries, each with their own outer polygon and hole.

    Gap of 2 pixels between outer ring and island ring ensures they are not
    8-connected.
    """
    mask = make_mask([
        '###########',
        '#.........#',
        '#.........#',
        '#..#####..#',
        '#..#...#..#',
        '#..#...#..#',
        '#..#####..#',
        '#.........#',
        '#.........#',
        '###########',
    ])
    result = components_pixel_polygons(mask)
    assert len(result) == 2, f"Expected 2 components, got {len(result)}"
    for i, (outer, holes) in enumerate(result):
        assert len(outer) > 0, f"Component {i} missing outer polygon"
        assert len(holes) == 1, f"Component {i}: expected 1 hole, got {len(holes)}"
        assert _signed_area(list(map(tuple, outer))) > 0
        assert _signed_area(list(map(tuple, holes[0]))) < 0
    print("PASS: island inside hole (two separate components)")
    
    
def test_close_regions():
    mask = make_mask([
        '###...',
        '####..',
        '.#.##.',
        '.##.##',
        '..##.#',
        '...#...',
    ])


if __name__ == '__main__':
    test_single_pixel()
    test_2x2_square()
    test_diagonal_two_pixels_8conn()
    test_anti_diagonal_two_pixels_8conn()
    test_l_shape()
    test_z_shape_8conn()
    test_ring_with_hole()
    test_island_inside_hole()
    print("\nAll tests passed.")
