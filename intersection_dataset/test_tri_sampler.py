import matplotlib.pyplot as plt
import matplotlib
import itertools
import sampler.line_sampler as line_sampler
import plotutils
import numpy as np
import cv2
import sampler.intersections as intersections
import rasterizer
import random

matplotlib.rcParams['path.simplify'] = False
np.random.seed(42)

BASE_DIR = "debug_files/vis"
DIM = 500
MIN_DIST = 2

NPOINTS = [15, 30, 50]
CLOSENESS = [10, 5]
SEEDS = [42, 47, 150, 99]

for n, cl, sd in itertools.product(NPOINTS, CLOSENESS, SEEDS):
    random.seed(sd)
    lines, ipoints_per_pl, red_points = line_sampler.sample_tri_method(DIM, DIM, n_points=n, 
                                                                  closeness=cl,
                                                                  min_dist=MIN_DIST)
    sequences = rasterizer.rasterized_split_seqs(lines, ipoints_per_pl, (DIM, DIM))
    half_edges = rasterizer.get_half_edge_splits(sequences)
    all_ipoints = intersections.points_flattened(ipoints_per_pl)
    # 1a
    path = f"{BASE_DIR}/{n}_{cl}_{sd}_lines.svg"
    fig, ax = plt.subplots(figsize=(10, 8))
    plotutils.plot_lines_and_points(ax, lines=lines, points = all_ipoints,
                                    color=plotutils.COLORS, lw=0.5, 
                                    linealpha=1, pointcolor="black")
    plotutils.plot_lines_and_points(ax, points =red_points, pointcolor="grey", psize=1)
    
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    # plotutils.plot_save_points_lines(path, lines, red_points, pointcolor="red", color=plotutils.COLORS)
    # plotutils.plot_save_points_lines(path, lines, all_ipoints, pointcolor="black", color=plotutils.COLORS)
    
    
    
    #1. plot splitted lines
    path = f"{BASE_DIR}/{n}_{cl}_{sd}_splitted_lines.svg"
    plotutils.plot_splitlines(DIM, lines, sequences, all_ipoints, path, d=MIN_DIST*2)
    
    #2 generate dataset version of image
    img_dataset = np.zeros((DIM, DIM, 3), dtype=np.uint8)
    rasterizer.color_half_edges_for_model(img_dataset, half_edges)
    cv2.imwrite(f"{BASE_DIR}/{n}_{cl}_{sd}_model_input.png", img_dataset)
    
    
    
    # verify: does the number of colors in the image match what we expect?
    r = rasterizer.check_numbers(img_dataset, ipoints_per_pl, lines)
    if not r: 
        print("INVALID SAMPLE\n")
        continue
    
    #3 get half-edge adjacency matrix
    M = rasterizer.adjacency_mat(half_edges)
    np.savetxt(f"{BASE_DIR}/{n}_{cl}_{sd}_Mat.csv", M, delimiter=",", fmt="%d")

    
    #4 plot half-edges from adjacency matrix
    pos = rasterizer.half_edge_centers(half_edges)
    path = f"{BASE_DIR}/{n}_{cl}_{sd}_half_edges_vis.svg"
    plotutils.plot_half_edges(DIM, M, half_edges, lines, all_ipoints, pos, path)
    
    
