import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random
from scipy.spatial import Delaunay
import matplotlib

import plotutils
from plotutils import plot_graph_nx
from plotutils import plot_paths
from plotutils import plot_tangent_vectors
import sampler.randompath as randompath
import sampler.intersections as intersections
import math

import sampler.triangulations as triangulations
from sampler.triangulations import compute_tangents_for_paths_all_tangent, get_path_coords, partition_graph_into_paths, remove_prioritized_non_cut_edges, sample_points_with_min_distance, triangulation_to_graph

matplotlib.rcParams['path.simplify'] = False




    
if __name__ == "__main__":
    WIDTH = 100
    HEIGHT = 100
    NUM_POINTS = 50
    MIN_DISTANCE = 5
    MAX_CYCLES=1
    # Constraints for partitioning
    # None = No limit (up to total nodes)
    MAX_PATH_LEN = None 
    # Minimum vertices to form a valid cycle (3 = triangle allowed)
    MIN_CYCLE_LEN = 5
    SEED = 499
    DELETE_EDGES = 3
    OUT_PATH = "debug_files/vis"
    np.random.seed(SEED)
    random.seed(SEED)

    print("1. Generating...")
    points = sample_points_with_min_distance(NUM_POINTS, WIDTH, HEIGHT, MIN_DISTANCE)
    # points = remove_extrema_points(points)
    
    if len(points) > 2:
        tri = Delaunay(points)
        graph = triangulation_to_graph(tri, points, ignore_chull=True)
        
        t = graph.number_of_edges()
        remove_prioritized_non_cut_edges(graph, int(t/DELETE_EDGES))
        
        print(f"2. Partitioning {graph.number_of_edges()} edges...")
        paths = partition_graph_into_paths(
            graph, 
            max_path_len=MAX_PATH_LEN, 
            min_cycle_len=MIN_CYCLE_LEN,
            max_cycles=MAX_CYCLES,
        )
        paths_coords = get_path_coords(graph, paths)
        tvectors = compute_tangents_for_paths_all_tangent(graph, paths, terminal_angles_strategy="random_random")
        
        # tvectors = apply_tangent_overshoot(tvectors, np.pi/8)
        
        middle_points = np.linspace(0, 1, num=50)
        t=middle_points
        # t = np.concatenate(([0], middle_points, [1]))
        
        lines = []
        for p, v in zip(paths_coords, tvectors):
            assert len(v) == len(p)
            sq = randompath.CubicBezierSequence(p, v, tension=0.33)
            curr = triangulations.rm_consecutive(np.concat([sq.get_points(i, t) for i in range(len(p)-1)]))
            assert curr.shape[1] ==2
            lines.append(curr)
        
        res = intersections.compute_intersections(lines)
        q = intersections.all_ipoints(res)
        
        
             
        
        # print("lines", len(lines))
        print("3. Visualizing...")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_graph_nx(ax, graph, draw_edges=2.0)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT_PATH}/graph_{SEED}_{DELETE_EDGES}_{MIN_DISTANCE}.svg", bbox_inches='tight')
        plt.close()
        
        # Draw background
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_graph_nx(ax, graph, draw_edges=0)
        
        # Draw paths
        plot_paths(ax, paths_coords)
        for p, v in zip(paths_coords, tvectors):
            assert len(v) == len(p)
            plot_tangent_vectors(ax, p, v, color="black")
        
        # plotutils.plot_lines_and_points(ax, lines=lines, color="black", lw=0.5)
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT_PATH}/result_{SEED}_{DELETE_EDGES}_{MIN_DISTANCE}.svg", bbox_inches='tight')
        plt.close()
        
        #make a plot with only the lines and their intersections
        fig, ax = plt.subplots(figsize=(10, 8))
        plotutils.plot_lines_and_points(ax, lines=lines,points = q,
                                        color=plotutils.COLORS, lw=1, 
                                        linealpha=1, pointcolor="black")
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.axis("off")
        plt.savefig(f"{OUT_PATH}/result_lines_{SEED}_{DELETE_EDGES}_{MIN_DISTANCE}.svg", bbox_inches='tight')

        
    else:
        print("Not enough points.")