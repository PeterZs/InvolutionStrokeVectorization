import sys
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotutils.plotutils as plu
from graph_extraction.build_linegraph import process, compute_polylines


if __name__ == "__main__":
    CASE = sys.argv[1]
    img = cv2.imread(f"graph_extraction/testcases/input/{CASE}.png", cv2.IMREAD_GRAYSCALE)

    OUT_DIR = f"graph_extraction/debug_files/{CASE}"
    os.makedirs(OUT_DIR, exist_ok=True)
    #with graph_clusters.timer("total"):
    g, success = process(img, invert=False, debug_dir=OUT_DIR)

    print("writing graph visualization svg")
    # out = f"graph_extraction/debug_files/{CASE}/result_overlay.svg"
    # plu.plot_img_graph(img, g, out, point_size=0.1, edge_width=0.05)
    out = f"graph_extraction/testcases/result/graph_overlay_{CASE}.svg"
    plu.plot_img_graph(img, g, out, point_size=0.1, edge_width=0.05)

    lines, connections = compute_polylines(g)
    # print("writing polyine svg")
    # out = f"graph_extraction/debug_files/{CASE}/result.svg"
    # plu.plot_lines_on_img(img, lines, out, linewidth=0.4)
