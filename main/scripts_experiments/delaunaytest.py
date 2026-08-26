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
# path = "debug_files/motobike/motobike_centerline2.png"
# path = "debug_files/test/test2_centerline.png"
# path = "debug_files/star/star_centerline.png"
# path = "graph_extraction/testcases/input/6.png"
# path = "graph_extraction/testcases/input/25.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
msk = image.simple_thresh(img, 10)
cv2.imwrite("hat.png", msk)

import networkx as nx



with utils.timer("pixel polygon"):
    polygons = pixel8_old.components_pixel_polygons(msk)

