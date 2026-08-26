import cv2
import numpy as np
import graph_extraction.image as image
import pixel8_old
import pixel8
import networkx as nx
from graph_extraction.build_linegraph import planargraph_core
import graph_extraction.shapeutils as shapeutils
import plotutils.plotutils as plu
import utils.utils as utils
path = "debug_files/hat/hat_centerline.png"
path = "debug_files/motobike/motobike_centerline2.png"
# path = "debug_files/test/test2_centerline.png"
# path = "debug_files/star/star_centerline.png"
# path = "graph_extraction/testcases/input/6.png"
# path = "graph_extraction/testcases/input/25.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
msk = image.simple_thresh(img, 10)
cv2.imwrite("hat.png", msk)

import networkx as nx

def remove_branches(G):
    """
    Remove all branches from a graph.
    Branches are nodes with degree 1 connected to nodes with degree > 2.
    
    Args:
        G: A NetworkX graph
    
    Returns:
        The modified graph (modified in place)
    """
    nodes_to_remove = set()
    
    # Find all leaf nodes (degree 1) and trace backwards
    for node in G.nodes():
        if G.degree(node) == 1:
            # Start from this leaf and traverse until we hit a node with degree > 2
            current = node
            path = [current]
            # print("--------")
            previous = current
            while G.degree(current) <= 2:
                #print(current, path)
                # Move to the next node (the one we haven't visited)
                next_nodel = [u for u in G.neighbors(current) if u != previous]
                
                if not len(next_nodel):
                    break
                next_node = int(next_nodel[0])
                
                # If we hit a node with degree > 2, stop (don't include it)
                if G.degree(next_node) > 2:
                    break
                previous = current
                current = next_node
                path.append(next_node)
            
            # Mark all nodes in path for removal
            nodes_to_remove.update(path)
    
    # Remove all collected nodes at the end
    G.remove_nodes_from(nodes_to_remove)
    


with utils.timer("pixel polygon"):
    polygons = pixel8_old.components_pixel_polygons(msk)

G = nx.Graph()
with utils.timer("voronoi"):
    for (ins, outs) in polygons:
        tmp = np.array(ins)
        holes = [np.array(i) for i in outs]
        print("do voronois")
        vor_vertices, vor_edges = shapeutils.get_multipolygon_voronoi(tmp, holes,
                                                                    densification_distance=0.2)
        
        H = nx.Graph()
        # Add initial Voronoi nodes
        for i, pos in enumerate(vor_vertices):
            H.add_node(i, pos=tuple(pos), accessible=None, type='voronoi')
        H.add_edges_from(vor_edges)
        planargraph_core.add_subgraph_to_G(G, H)

print("removing branches")
remove_branches(G)

paths = planargraph_core.decompose_into_paths(G)
pos = nx.get_node_attributes(G, "pos")
for path in paths:
    coords = np.array([pos[n] for n in path])
    smoothed = shapeutils.smooth_polyline_constrained(coords, d=2.5)
    for node, new_pos in zip(path, smoothed):
        G.nodes[node]["pos"] = tuple(new_pos)

print("plotting")
with plu.ImageOverlay("graph.svg", msk, max_gray="#BCBCBC") as ax:
    for (ins, outs) in polygons:
        plu.plot_polylines(ax, [ins], colors=["white"], linewidth=0.1)
        plu.plot_polylines(ax, outs, linewidth=0.1)
    
    plu.plot_graph(ax, G, color="white")
