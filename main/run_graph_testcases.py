import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotutils.plotutils as plu
from graph_extraction.build_linegraph import process, compute_polylines
import utils.utils as utils
import os

CASES_DIR = "graph_extraction/testcases"
caselist = os.listdir(f"{CASES_DIR}/input")
bad = []
for case in caselist:
    if not case.endswith(".png") or case.endswith("mask.png"):
        continue
    print()
    print(case)
    img = cv2.imread(f"{CASES_DIR}/input/{case}", cv2.IMREAD_GRAYSCALE)
    try:
        with utils.timer("total"):
            g, success = process(img)
    except Exception as e:
        print("problem:", e)
        bad.append(case)
        continue

    print("writing graph visualization svg")
    out = f"{CASES_DIR}/result/graph_overlay_{case[:-4]}.svg"
    plu.plot_img_graph(img, g, out)
    lines, connections = compute_polylines(g)
    print("writing polyine svg")
    out = f"{CASES_DIR}/result/result_{case[:-4]}.svg"
    plu.plot_lines_on_img(img, lines, out, linewidth=0.4)

if bad:
    print("graph extraction failed for: ", bad)