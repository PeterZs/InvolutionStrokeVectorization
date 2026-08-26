import numpy as np
import cv2
from collections import defaultdict


def _signed_area(poly) -> float:
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


def _point_in_polygon(poly, point) -> bool:
    """Ray-casting point-in-polygon test. Works for self-touching polygons."""
    x, y = point
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _interior_sample(poly):
    """
    Return a point strictly inside the polygon.

    All edges are unit-length axis-aligned. The interior lies on the RIGHT of
    the direction of travel (fg on right for our CW/CCW convention):
      - CW polygon (outer): interior = fg pixel center just right of first edge.
      - CCW polygon (hole): interior = bg pixel center just right of first edge
        (which is bg since the polygon winds the OTHER way around bg).
    """
    v0 = poly[0]
    v1 = poly[1]
    dx = v1[0] - v0[0]
    dy = v1[1] - v0[1]
    mx = (v0[0] + v1[0]) / 2
    my = (v0[1] + v1[1]) / 2
    # Rotate direction 90° toward interior (right side in image coords).
    # (dx, dy) → (-dy, dx) is visual CCW rotation of direction.
    # In image Y-down coords, "right of direction" = visual CCW rotation.
    nx, ny = -dy, dx
    return (mx + 0.5 * nx, my + 0.5 * ny)


def component_pixel_polygon(
    mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """
    Given a binary mask (uint8, single 8-connected component, non-zero = foreground),
    returns one multipolygon as (outer, holes):
      - outer: list of (x, y) corner coordinates tracing the outer boundary CW
      - holes: list of polygons (each CCW) tracing enclosed background regions

    With 8-connectivity, diagonal pixel neighbors are considered connected. At
    diagonal-junction corners (X-junctions) the tracer has to decide how to
    pair incoming and outgoing edges. Two pairings are geometrically valid:

      - "same-pixel" pairing (tightest right turn / MAX cross product):
        each incoming edge exits via the other edge of the same fg pixel.
        This makes the trace detour around the bg pinch at the junction,
        absorbing an enclosed single-cell bg pocket into the outer polygon
        instead of producing a spurious hole.

      - "cross" pairing (tightest left turn / MIN cross product): the trace
        crosses the junction diagonally, producing a self-touching figure-8
        polygon. This is what we want when the bg at the junction is the
        exterior (e.g., two pixels touching only at a corner): the single
        8-CC is traced as one figure-8 polygon wrapping both pixels.

    We choose the pairing per X-junction based on whether BOTH diagonal bg
    cells at the junction are connected (via 4-connectivity through bg) to
    the image boundary. If yes → cross pairing (figure-8 through external
    pinch). If any bg cell is an enclosed pocket → same-pixel pairing
    (absorb the pocket into the outer polygon).

    A single 8-connected component therefore produces exactly one outer
    polygon (possibly self-touching) plus any fully enclosed holes.

    Corner coordinate convention: pixel at grid position (col, row) occupies
    the unit square with corners (col, row) .. (col+1, row+1).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], []

    pixel_set = set(zip(xs.tolist(), ys.tolist()))
    H, W = mask.shape

    # ---- Classify X-junctions ----------------------------------------------
    # 4-connected labeling of bg, padded with bg so the image boundary acts as
    # a single exterior bg region.
    bg = (mask == 0).astype(np.uint8)
    bg_padded = np.pad(bg, 1, constant_values=1)
    _, bg_labels = cv2.connectedComponents(bg_padded, connectivity=4)
    exterior_bg_label = bg_labels[0, 0]

    def bg_cell_is_exterior(px, py):
        if not (0 <= px < W and 0 <= py < H):
            return True
        if mask[py, px] > 0:
            return False
        return bg_labels[py + 1, px + 1] == exterior_bg_label

    # X-junction: corner where two diagonal fg pixels meet two diagonal bg
    # pixels. Map corner → 'cross' (MIN) or 'same' (MAX).
    x_junction_policy: dict[tuple[int, int], str] = {}
    for cx in range(W + 1):
        for cy in range(H + 1):
            nw = (cx - 1, cy - 1)
            ne = (cx,     cy - 1)
            sw = (cx - 1, cy    )
            se = (cx,     cy    )
            nw_fg = nw in pixel_set
            ne_fg = ne in pixel_set
            sw_fg = sw in pixel_set
            se_fg = se in pixel_set
            if nw_fg + ne_fg + sw_fg + se_fg != 2:
                continue
            if nw_fg and se_fg and not ne_fg and not sw_fg:
                bg_cells = [ne, sw]
            elif ne_fg and sw_fg and not nw_fg and not se_fg:
                bg_cells = [nw, se]
            else:
                continue
            both_exterior = all(bg_cell_is_exterior(*b) for b in bg_cells)
            x_junction_policy[(cx, cy)] = 'cross' if both_exterior else 'same'

    # ---- Build directed edge graph -----------------------------------------
    # Oriented so that foreground is on the RIGHT of direction of travel.
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

    # In image coords (Y-down), the cross product
    #     inc × (cand − cur)  =  inc.x * d.y − inc.y * d.x
    # is POSITIVE for CW (right) turns and NEGATIVE for CCW (left) turns.
    # MAX cross = tightest right turn = "same pixel" pairing at X-junction.
    # MIN cross = tightest left turn  = "cross" pairing at X-junction.
    def pick_next(cur, prev, candidates):
        if len(candidates) == 1:
            return candidates[0]
        inc = (cur[0] - prev[0], cur[1] - prev[1])
        def cross(cand):
            d = (cand[0] - cur[0], cand[1] - cur[1])
            return inc[0] * d[1] - inc[1] * d[0]
        policy = x_junction_policy.get(cur, 'same')
        return min(candidates, key=cross) if policy == 'cross' else max(candidates, key=cross)

    # Trace closed polygons by following the edge graph; each directed edge is
    # consumed exactly once. A junction node may be visited multiple times as
    # the trace weaves through bg pinches; we detect true closure by checking
    # that no outgoing edge from start_node remains unvisited.
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

    # Classify by winding direction: CW (positive area in Y-down) = outer, CCW = hole.
    # A single 8-connected component produces exactly one outer (possibly
    # self-touching, i.e. figure-8-shaped when X-junctions are present) plus
    # any fully-enclosed holes.
    outers = [p for p in polygons if _signed_area(p) > 0]
    holes = [p for p in polygons if _signed_area(p) < 0]

    # assert len(outers) == 1, (
    #     f"Expected exactly 1 outer polygon per 8-connected component, got {len(outers)}. "
    #     f"Is the input mask really a single 8-connected component?"
    # )
    outer = outers[0]
    return outer, holes


def components_pixel_polygons(
    binary_image: np.ndarray,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Run connected components (8-connectivity) on a binary image and return
    one multipolygon per component as (outer_polygon, holes).

    outer_polygon: ndarray of shape (N, 2) with (x, y) corner coordinates (CW).
    holes: list of ndarrays, each shape (M, 2), traced CCW.

    An island inside a hole is a separate connected component and appears as
    its own entry in the result.
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
    Two pixels touching only diagonally are ONE 8-connected component with an
    X-junction at the shared corner. The closest-CW rule weaves through the
    junction, producing ONE figure-8 polygon that wraps around both pixels
    (8 corners total, with a self-touch at the junction corner). No holes.
    """
    mask = make_mask(['#.', '.#'])
    outer, holes = component_pixel_polygon(mask)
    assert len(outer) == 8, f"Expected 8-corner figure-8 polygon, got {len(outer)}: {outer}"
    assert holes == []
    # Both pixels' four corners must appear in the trace.
    corners_expected = {(0,0),(1,0),(1,1),(0,1),(2,1),(2,2),(1,2)}
    assert corners_expected <= set(outer), f"Missing corners: {corners_expected - set(outer)}"
    print("PASS: diagonal two pixels (8-conn) → 1 figure-8 multipolygon")


def test_anti_diagonal_two_pixels_8conn():
    mask = make_mask(['.#', '#.'])
    outer, holes = component_pixel_polygon(mask)
    assert len(outer) == 8, f"Expected 8-corner figure-8 polygon, got {len(outer)}: {outer}"
    assert holes == []
    print("PASS: anti-diagonal two pixels (8-conn) → 1 figure-8 multipolygon")


def test_l_shape():
    mask = make_mask(['#.', '##'])
    outer, holes = component_pixel_polygon(mask)
    assert holes == []
    assert len(outer) > 0
    print("PASS: L-shape")


def test_z_shape_8conn():
    """Z-shape with a diagonal chain: still one 8-connected component, so one
    multipolygon. Depending on corner configurations, the outer may be
    figure-8-shaped if any X-junctions exist."""
    mask = make_mask(['#.', '##', '.#'])
    outer, holes = component_pixel_polygon(mask)
    total_fg_pixels = int((mask > 0).sum())
    print(f"  Z-shape (8-conn): outer={len(outer)} corners for {total_fg_pixels} fg pixels, holes={len(holes)}")
    assert len(outer) > 0
    assert holes == []
    print("PASS: Z-shape 8-conn")


def test_ring_with_hole():
    """A 5x5 ring: no X-junctions, one outer + one hole."""
    mask = make_mask([
        '#####',
        '#...#',
        '#...#',
        '#...#',
        '#####',
    ])
    outer, holes = component_pixel_polygon(mask)
    print(f"  ring outer length={len(outer)}, hole count={len(holes)}")
    assert len(outer) > 0
    assert len(holes) == 1, f"Expected 1 hole, got {len(holes)}"
    assert _signed_area(outer) > 0
    assert _signed_area(holes[0]) < 0
    print("PASS: ring with hole")


def test_island_inside_hole():
    """Outer ring with an island ring inside its hole. Two separate CCs, each
    one multipolygon with one outer and one hole."""
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
    assert len(result) == 2, f"Expected 2 multipolygons, got {len(result)}"
    for i, (outer, holes) in enumerate(result):
        assert len(outer) > 0
        assert len(holes) == 1, f"Component {i}: expected 1 hole, got {len(holes)}"
        assert _signed_area(list(map(tuple, outer))) > 0
        assert _signed_area(list(map(tuple, holes[0]))) < 0
    print("PASS: island inside hole (two separate components)")


def test_close_regions():
    """
    Shape with two X-junctions (at corners (3,3) and (4,4)) where the bg
    pinches are INTERIOR (absorbed by the single connected shape). With the
    closest-CW rule, the trace weaves through both pinches and produces ONE
    multipolygon with NO holes — the bg pinches are part of the outer boundary,
    not separate hole polygons.
    """
    mask = make_mask([
        '###...',
        '####..',
        '.#.##.',
        '.##.##',
        '..##.#',
        '...#..',
    ])
    outer, holes = component_pixel_polygon(mask)
    print(f"  close_regions: outer={len(outer)} corners, holes={len(holes)}")
    assert len(holes) == 0, f"Expected 0 holes, got {len(holes)}"
    assert len(outer) > 0
    print("PASS: close regions (weave through interior X-junctions)")

def test_close_regions2():
    """
    Shape with two X-junctions (at corners (3,3) and (4,4)) where the bg
    pinches are INTERIOR (absorbed by the single connected shape). With the
    closest-CW rule, the trace weaves through both pinches and produces ONE
    multipolygon with NO holes — the bg pinches are part of the outer boundary,
    not separate hole polygons.
    """
    mask = make_mask([
        '.#..',
        '.##.',
        '..#.',
        '...#',
        '...#',
    ])
    outer, holes = component_pixel_polygon(mask)
    print(f"  close_regions: outer={len(outer)} corners, holes={len(holes)}")
    assert len(holes) == 0, f"Expected 0 holes, got {len(holes)}"
    assert len(outer) > 0
    assert len(outer) == 16
    print(outer)
    print("PASS: close regions (weave through interior X-junctions)")


if __name__ == '__main__':
    test_single_pixel()
    test_2x2_square()
    test_diagonal_two_pixels_8conn()
    test_anti_diagonal_two_pixels_8conn()
    test_l_shape()
    test_z_shape_8conn()
    test_ring_with_hole()
    test_island_inside_hole()
    test_close_regions()
    test_close_regions2()
    print("\nAll tests passed.")
