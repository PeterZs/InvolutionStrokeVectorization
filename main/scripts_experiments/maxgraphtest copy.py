import cv2
import numpy as np
import networkx as nx
from networkx.utils import UnionFind
import plotutils.plotutils as plu

path = "debug_files/hat/hat_centerline.png"
path = "debug_files/star/star_centerline.png"
path = "debug_files/star/star_centerline2.png"

OFFSETS_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def build_max_neighbor_graph(img):
    """Spawn a node at every nonzero pixel and connect each to the
    8-neighbor(s) with the highest value (ties produce multiple edges).
    """
    H, W = img.shape
    G = nx.Graph()

    ys, xs = np.nonzero(img)
    node_id = {}
    for nid, (i, j) in enumerate(zip(ys.tolist(), xs.tolist())):
        node_id[(i, j)] = nid
        G.add_node(nid, pos=(j + 0.5, i + 0.5), pixel=(i, j), value=int(img[i, j]))

    for (i, j), nid in node_id.items():
        candidates = []
        for di, dj in OFFSETS_8:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) in node_id:
                candidates.append((ni, nj, int(img[ni, nj])))
        if not candidates:
            continue
        max_val = max(v for _, _, v in candidates)
        for ni, nj, v in candidates:
            if v == max_val:
                G.add_edge(nid, node_id[(ni, nj)])
                break

    return G, node_id


def bridge_dangling_endpoints(G, node_id):
    """For every degree-1 node (in descending image-value order), if any
    8-neighbor pixel-node lies in a different connected component, add an
    edge to the highest-valued such neighbor.
    """
    uf = UnionFind(G.nodes)
    for u, v in G.edges:
        uf.union(u, v)
    pos = nx.get_node_attributes(G, "pos")
    degree1 = [n for n, d in G.degree() if d == 1]
    degree1.sort(key=lambda n: G.nodes[n]["value"], reverse=True)
    print([pos[n] for n in degree1])
    for nid in degree1:
        i, j = G.nodes[nid]["pixel"]
        best = None
        best_val = -1
        for di, dj in OFFSETS_8:
            key = (i + di, j + dj)
            other = node_id.get(key)
            if other is None:
                continue
            if uf[nid] == uf[other]:
                continue
            v = G.nodes[other]["value"]
            if v > best_val:
                best_val = v
                best = other
        print(pos[nid], best)
        if best is not None:
            print("adding", pos[nid], pos[best])
            G.add_edge(nid, best)
            uf.union(nid, best)


img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
G, node_id = build_max_neighbor_graph(img)
bridge_dangling_endpoints(G, node_id)

print("plotting")
with plu.ImageOverlay("maxgraphtest_overlay.svg", cv2_img=img) as ax:
    plu.plot_graph(ax, G, color="red")
