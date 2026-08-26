import matplotlib.pyplot as plt
import matplotlib
import itertools
import sampler.line_sampler as line_sampler
import plotutils
import numpy as np
import cv2
import sampler.intersections as intersections
import rasterizer
matplotlib.rcParams['path.simplify'] = False
np.random.seed(42)
BASE_DIR = "samples"
N = 500
MIN_DIST = 4
DIM = 500
for n_curves in range(1, 8):
    
    lines = line_sampler.sample_curves(n_curves, N, MIN_DIST, dim=DIM)
    
    inter_l = intersections.compute_intersections(lines)
    ipoints_per_pl = [np.array([x.point for x in i]) for i in inter_l]
    sequences = rasterizer.rasterized_split_seqs(lines, ipoints_per_pl, (DIM, DIM))
    half_edges = rasterizer.get_half_edge_splits(sequences)

    #1. plot splitted lines
    img = np.zeros((DIM, DIM, 3), dtype=np.uint8) 
    plotutils.mark_pixels_alternating(img, sequences, tup_idx=0)
    
    path = f"{BASE_DIR}/valid_polyline_set/{n_curves}_splitted_lines.svg"
    
    all_ipoints = np.array(list(set(tuple(x.point) for i in inter_l for x in i)))
    with plotutils.ImageOverlay(path, img) as ax:
        plotutils.plot_lines_and_points(ax, lines, all_ipoints)
        plotutils.annotate_short_distances(ax, all_ipoints, d=2*MIN_DIST)
    
    #2 generate dataset version of image
    img_dataset = np.zeros((DIM, DIM, 3), dtype=np.uint8)
    rasterizer.color_half_edges_for_model(img_dataset, half_edges)
    cv2.imwrite(f"{BASE_DIR}/valid_polyline_set/{n_curves}_model_input.png", img_dataset)
    
    # verify: does the number of colors in the image match what we expect?
    print("verify: ", rasterizer.check_numbers(img_dataset, inter_l, lines))
    
    #3 get half-edge adjacency matrix
    M = rasterizer.adjacency_mat(half_edges)
    np.savetxt(f"{BASE_DIR}/valid_polyline_set/{n_curves}_Mat.csv", M, delimiter=",", fmt="%d")

    
    #4 plot half-edges
    img = np.zeros((DIM, DIM, 3), dtype=np.uint8)
    plotutils.mark_pixels_alternating(img, half_edges, tup_idx=0)
    
    path = f"{BASE_DIR}/valid_polyline_set/{n_curves}_half_edges_vis.svg"
    with plotutils.ImageOverlay(path, img) as ax:
        plotutils.plot_lines_and_points(ax, lines, all_ipoints)
        pos = rasterizer.half_edge_centers(half_edges)
        plotutils.plot_graph_from_M(ax, M, pos, point_size=0.2, edge_width=0.05, both_directions=False)

    
    
