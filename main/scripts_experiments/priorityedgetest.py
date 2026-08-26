import cv2
import numpy as np
import networkx as nx
import plotutils.plotutils as plu

path = "debug_files/hat/hat_centerline.png"
path = "debug_files/balloon/balloon_centerline.png"

# path = "debug_files/star/star_centerline.png"
# path = "debug_files/star/star_centerline2.png"

OFFSETS_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def _canon(a, b):
    return (a, b) if a < b else (b, a)


def build_priority_path_graph(img, threshold=15):
    """Spawn a node at every pixel >= threshold, then walk every 8-neighbor
    candidate edge in priority order (axis-aligned first, then min-pixel-value
    descending), accepting an edge only if neither endpoint already has
    degree 2. Result: every node ends up with degree 0, 1, or 2 — i.e. the
    graph is a disjoint union of simple paths and simple cycles.
    """
    H, W = img.shape
    G = nx.Graph()

    ys, xs = np.nonzero(img)
    node_id = {}
    counter = 0
    for i, j in zip(ys.tolist(), xs.tolist()):
        if int(img[i, j]) < threshold:
            continue
        node_id[(i, j)] = counter
        G.add_node(
            counter, pos=(j + 0.5, i + 0.5), pixel=(i, j), value=int(img[i, j])
        )
        counter += 1

    candidates = []
    seen = set()
    for (i, j), nid in node_id.items():
        v_nid = G.nodes[nid]["value"]
        for di, dj in OFFSETS_8:
            other = node_id.get((i + di, j + dj))
            if other is None:
                continue
            edge = _canon(nid, other)
            if edge in seen:
                continue
            seen.add(edge)
            axis_aligned = 1 if (di == 0 or dj == 0) else 0
            candidates.append(
                (axis_aligned, min(v_nid, G.nodes[other]["value"]), nid, other)
            )

    # axis-aligned first, then highest min-value first
    candidates.sort(key=lambda c: c[1], reverse=True)

    for _, _, a, b in candidates:
        if G.degree(a) >= 2 or G.degree(b) >= 2:
            continue
        G.add_edge(a, b)

    return G, node_id





img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
G, node_id = build_priority_path_graph(img)

print("plotting")
with plu.ImageOverlay("priorityedge_overlay.svg", cv2_img=img) as ax:
    plu.plot_graph(ax, G, color="red")
