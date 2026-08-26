import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
import itertools
import uuid
from matplotlib.collections import LineCollection

COLORS = ["tomato", "mediumblue", "grey", 
          "yellowgreen", "plum", "burlywood", "lightseagreen", "cornflowerblue", "darkgreen", "darkorchid"]
PAIRED_COLORS = [("brown", "tomato"), ("mediumblue", "#85A2FF"), ("darkorchid", "plum"),  
                 ("teal", "#79D6D4"), ("olive", "khaki")]

COLORS_ON_BLACK = ['#FFA500', 'sienna', 'mediumblue', 'green', 'orchid']

def adequate_font_size(ax, h):
    fig = ax.figure
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((0, h))

    height_px = abs(p1[1] - p0[1])
    fontsize_pt = height_px * 72 / fig.dpi
    return fontsize_pt



class ImageOverlay:
    def __init__(self, output_path, cv2_img= None, dims = None, max_gray = "white", min_black="black"):
        """
        :param cv2_img: Numpy array (BGR from cv2.imread or Greyscale)
        :param output_path: String path (e.g., 'output.svg' or 'output.png')
        """
        assert (cv2_img is not None) or (dims is not None)
        self.path = output_path
        self.img = None
        self.max_gray = max_gray
        self.min_black = min_black

        if cv2_img is not None:
            self.img = cv2_img
            if len(self.img.shape) == 3:
                self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            else:
                self.img_rgb = self.img # Greyscale
                
        self.height, self.width = dims if dims is not None else self.img.shape[:2]
        # else:
        #     self.img = cv2.cvtColor(np.ones(dims, dtype=np.uint8)*255, cv2.COLOR_GRAY2RGB)
        
        # 1. Handle Color Conversion (OpenCV BGR -> Matplotlib RGB)
            
        
        # 2. Setup DPI and Figure Size
        # 72 DPI is the standard for SVG (1pt = 1/72in)
        # This makes 1 unit in the SVG viewBox == 1 pixel
        self.dpi = 72
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        
        # 3. Create axis covering 100% of the figure
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')


    def __enter__(self):
        # 4. Display image with inverted Y-axis logic
        # extent=[left, right, bottom, top] sets 0,0 at the very top-left edge
        
        if self.img is not None:
            if len(self.img.shape) ==2:
                custom_cmap = LinearSegmentedColormap.from_list('custom_gray', [self.min_black, self.max_gray])
                self.ax.imshow(self.img_rgb, extent=[0, self.width, self.height, 0], 
                            aspect='auto', cmap = custom_cmap)
            else:
                self.ax.imshow(self.img_rgb, extent=[0, self.width, self.height, 0],
                            aspect='auto', interpolation='nearest')
        
        # Set limits strictly to edges
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. Save with zero padding
        self.fig.patch.set_visible(False)
        self.fig.savefig(self.path, dpi=self.dpi, pad_inches=0, transparent=True)
        plt.close(self.fig)
 

def plot_points_and_edges(ax: Axes, points, edges, 
                          point_color: str | list[str] ='red', edge_color : str | list[str] ='blue', 
                          point_size=0.12, edge_width=0.11):
    """
    Plot points and edges on a given Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    points : list of tuple or np.ndarray
        List of (x, y) coordinates.
    edges : list of tuple
        List of (i, j) index pairs indicating edges between points.
    point_color : str, optional
        Color of the points.
    edge_color : str, optional
        Color of the edges.
    point_size : float, optional
        Size of the points.
    edge_width : float, optional
        Width of the edge lines.
    """
    # Unzip points into x and y coordinates
    xs, ys = zip(*points)
    
    if isinstance(point_color, str):
        point_color = [point_color for _ in range(len(xs))]
    if isinstance(edge_color, str):
        edge_color = [edge_color for _ in range(len(edges))]

    # Plot points
    ax.scatter(xs, ys, c=point_color, s=point_size, zorder=2, edgecolors=None, linewidths=0)

    # Plot edges
    for k, (i, j) in enumerate(edges):
        if i >= len(points):
            print(i, len(points))
            raise ValueError
        if j >= len(points):
            print(j, len(points))
            raise ValueError
            
        x_values = [points[i][0], points[j][0]]
        y_values = [points[i][1], points[j][1]]
        ax.plot(x_values, y_values, color = edge_color[k], linewidth = edge_width, zorder = 1)

    ax.set_aspect('equal', adjustable='box')


def plot_graph(ax: Axes, G_in: nx.Graph, color = "white", **kwargs):
    mapping = {old_label: i for i, old_label in enumerate(G_in.nodes())}
    G = nx.relabel_nodes(G_in, mapping, copy=True)
    points_list = [G.nodes[i]['pos'] for i in G.nodes]
    edges_list = list(G.edges())
    colors = [G.nodes[i].get('color', color) for i in G.nodes]
    plot_points_and_edges(ax, points_list, edges_list, colors, edge_color=color, **kwargs)
    for i in G.nodes:
        info = G.nodes[i].get('info')
        if info:
            x, y = G.nodes[i]['pos']
            ax.text(x, y, info, fontsize=1, color=color, ha='left', va='bottom')

def plot_polylines(ax: Axes, lines, colors=None, linewidth=1.0):
    """
    Plot multiple polylines on a given matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    lines : list of np.ndarray
        Each array is a polyline with shape (N, 2) or (2, N).
    colors : list of color specs, optional
        Colors to cycle through. Defaults to matplotlib color cycle.
    linewidth : float, optional
        Line width for the polylines.
    """
    if colors is None:
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    color_cycle = itertools.cycle(colors) if isinstance(colors, list) else colors
    lines_xy = [line.squeeze() for line in lines]
    tmp = [next(color_cycle) for _ in lines]
    lc = LineCollection(lines_xy, colors=[next(color_cycle) for _ in lines], linewidths=linewidth)
    #lc.set_gid(group_id)
    ax.add_collection(lc)
    # for line in lines:
    #     line = line.squeeze()
    #     x, y = line[:, 0], line[:, 1]

    #     artists = ax.plot(
    #         x,
    #         y,
    #         color=next(color_cycle),
    #         linewidth=linewidth,
    #         marker=None
    #     )
        
    #     for artist in artists:
    #         artist.set_gid(group_id)
    

def plot_polylines_rainbow(ax: Axes, lines: list[np.ndarray], cmapname="rainbow", linewidth=1):
    if not lines:
        return

    cmap = plt.get_cmap(cmapname)
    for line in lines:
        n = len(line)
        if n < 2:
            continue

        # Cumulative arc length for this polyline
        diffs = np.diff(line, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.concatenate(([0], np.cumsum(seg_lengths)))
        total_length = cum_length[-1] if cum_length[-1] > 0 else 1.0

        # Draw each segment
        for i in range(n - 1):
            t = cum_length[i] / total_length
            color = cmap(t)
            ax.plot(
                line[i:i+2, 0], line[i:i+2, 1],
                color=color, linewidth=linewidth, solid_capstyle="round",
            )
        
def plot_lines_on_img(img, lines, output_path,  hide_img=True, **kwargs):
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("black")
    if not hide_img:
        max_gray = "#5C5C5C" 
        custom_cmap = LinearSegmentedColormap.from_list('custom_gray', ['black', max_gray])
        ax.imshow(img, extent=[0, w, h, 0], cmap=custom_cmap)

    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # ensure origin top-left
    plot_polylines(ax, lines, **kwargs)

    plt.savefig(output_path, format='svg', dpi=100)
    plt.close(fig)


def plot_graph_from_M(ax, M: np.ndarray, positions: dict,
               point_size=1.0, edge_width=0.5, 
               color = "white", curved=True, both_directions=False, node_labels={}):
    """
    Plot a graph with slightly curved edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    adjacency_lst : dict[int, tuple]
        Mapping of node id -> (
            (x, y),          # node position
            [neighbor_ids]   # adjacency list
        )
    point_size : float
        Size of plotted points.
    edge_width : float
        Width of plotted edges.
    """
    assert M.shape[0]==M.shape[1]
    # Plot nodes
    for pos in positions.values():
        ax.scatter(pos[0], pos[1], s=point_size, color = color, zorder=2, edgecolors=None, linewidths=0,)

    if node_labels:
        sz = adequate_font_size(ax, 0.25)
        for k, txt in node_labels.items():
            pos = positions[k]
            x, y = pos[0] +1, pos[1] + 1
            ax.text(
                    x,
                    y,
                    txt,
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=sz,
                    zorder=3,
                    )

    # Plot curved edges
    drawn_edges = set()
    for u in range(len(M)):
        for v in range(len(M)):
            if M[u][v] >0 and u in positions:
                x1, y1 = positions[u]
                tup = tuple(sorted((u, v)))
                if tup in drawn_edges and not both_directions:
                    continue
                drawn_edges.add(tup)

                x2, y2 = positions[v]
                curr_color = color
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        color=curr_color,
                        shrinkA=0 if not both_directions else 0.02+point_size,
                        shrinkB=0 if not both_directions else 0.02+point_size,
                        arrowstyle="-" if not both_directions else "-|>, head_width=0.005, head_length=0.01",
                        linewidth=edge_width,
                        connectionstyle="arc3,rad=0.15",  # curvature
                    ),
                )

 


def plot_img_graph(img, G, output_path, max_gray = "#8B8B8B", min_black = "black", color="white", **kwargs):
    
    with ImageOverlay(output_path, cv2_img=img, max_gray=max_gray, min_black=min_black) as ax:
        plot_graph(ax, G, color=color, **kwargs)
    # h, w = img.shape
    # fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


    # custom_cmap = LinearSegmentedColormap.from_list('custom_gray', ['black', max_gray])
    # ax.imshow(img, extent=[0, w, h, 0], cmap=custom_cmap)
    # # ax.imshow(img, extent=[0, w, h, 0])

    # plot_graph(ax, G, **kwargs)
    # ax.axis('off')
    # ax.set_xlim(0, w)
    # ax.set_ylim(h, 0)  # ensure origin top-left


    # plt.savefig(output_path, format='svg', dpi=100)
    # plt.close(fig)
    
def plot_lines_on_img2(img, lines, output_path,  hide_img=False, dims= None, **kwargs):
    with ImageOverlay(output_path, cv2_img=img if not hide_img else None, dims = dims) as ax:
        plot_polylines(ax, lines, **kwargs)