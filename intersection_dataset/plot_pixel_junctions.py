import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

import sampler.intersections as intersections
import sampler.randompath as randompath
import plotutils
import rasterizer
np.random.seed(151)


    
N = 1000
DIM=500

lines = [
    randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim=DIM),
    randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim=DIM),
    randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim=DIM),
    randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim=DIM),
    randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim=DIM),
    np.array([(100, 100), (180, 180)]),
    np.array([(101, 180), (181, 100)]),
]

res = intersections.compute_intersections(lines)
all_ipoints = np.array([x.point for i in res for x in i])
ipoints_per_pl = [np.array([x.point for x in i]) for i in res]

vis_img = np.zeros((500, 500, 3), dtype=np.uint8)

sequences = rasterizer.rasterized_split_seqs(lines, ipoints_per_pl, vis_img.shape[:2])
plotutils.mark_pixels_alternating(vis_img, sequences, tup_idx=0)



with plotutils.ImageOverlay("debug_files/lines_pixelated.svg", vis_img) as ax:
    plotutils.plot_lines_and_points(ax, lines, all_ipoints)

    adj, labels = rasterizer.obtain_edge_split_graph_for_viz(sequences)
    plotutils.plot_graph(ax, adj, point_size=0.2, edge_width=0.05, 
                         node_labels=labels,
                         both_directions=True)

half_edges = rasterizer.get_half_edge_splits(sequences)

vis_img = np.zeros((500, 500, 3), dtype=np.uint8)
plotutils.mark_pixels_alternating(vis_img, sequences, tup_idx=0)

img_dataset = np.zeros((500, 500, 3), dtype=np.uint8)

rasterizer.color_half_edges_for_model(img_dataset, half_edges)
cv2.imwrite("debug_files/img_dataset_sample.png", img_dataset)
M = rasterizer.adjacency_mat(half_edges)

np.savetxt("debug_files/output.csv", M, delimiter=",", fmt="%d")

pos = rasterizer.half_edge_centers(half_edges)
with plotutils.ImageOverlay("debug_files/lines_half_edges.svg", vis_img) as ax:
    plotutils.plot_lines_and_points(ax, lines, all_ipoints)
    plotutils.plot_graph_from_M(ax, M, pos, point_size=0.2, edge_width=0.05, both_directions=False)
