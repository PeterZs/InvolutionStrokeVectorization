import matplotlib.pyplot as plt
import cv2
import os
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
import matplotlib
import itertools
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from typing import Any

COLORS = ["tomato", "mediumblue", "grey", "yellowgreen", "orchid", "burlywood", "lightseagreen", "cornflowerblue", "darkgreen"]
PAIRED_COLORS = [("brown", "tomato"), ("mediumblue", "dodgerblue"), ("darkorchid", "plum"),  
                 ("teal", "paleturquoise"), ("olive", "khaki")]


def color_to_bgr(color):
    """
    Convert a Matplotlib color to an OpenCV BGR tuple (uint8).

    Parameters
    ----------
    color : str or tuple
        Matplotlib color (e.g. 'red', '#ff0000', (0.5, 0.2, 0.1), etc.)

    Returns
    -------
    tuple
        (B, G, R) in range 0–255
    """
    # Convert to RGBA in 0–1 range
    rgba = mcolors.to_rgba(color)

    # Extract RGB, convert to 0–255, and reverse to BGR
    r, g, b = rgba[:3]
    return (int(b * 255), int(g * 255), int(r * 255))

def plot_lines_and_points(ax: Axes, lines: list[np.ndarray] | None = None, 
    points: np.ndarray | None = None, color: str| list="white", pointcolor : str = "white", lw = 0.1, linealpha=0.25, psize = None):
    if lines is not None:
        if isinstance(color, str):
            color = [color]
        colors = itertools.cycle(color)
        for pl in lines:
            c = next(colors)
            ax.plot(*pl.T, color=c, lw=lw, alpha = linealpha)
    if points is not None and len(points):
        plt.scatter(*points.T, c=pointcolor, s=4*lw if psize is None else psize, zorder=2, edgecolors=None, linewidths=0)
    

class ImageOverlay:
    def __init__(self, output_path, cv2_img= None, dims = None):
        """
        :param cv2_img: Numpy array (BGR from cv2.imread or Greyscale)
        :param output_path: String path (e.g., 'output.svg' or 'output.png')
        """
        assert (cv2_img is not None) or (dims is not None)
        if cv2_img is not None:
            self.img = cv2_img
        else:
            self.img = cv2.cvtColor(np.ones(dims, dtype=np.uint8)*255, cv2.COLOR_GRAY2RGB)
        self.path = output_path
        
        # 1. Handle Color Conversion (OpenCV BGR -> Matplotlib RGB)
        if len(self.img.shape) == 3:
            self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            self.img_rgb = self.img # Greyscale
            
        self.height, self.width = self.img.shape[:2]
        
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
        max_gray = "#FFFFFF" 
        custom_cmap = LinearSegmentedColormap.from_list('custom_gray', ['black', max_gray])
        if len(self.img.shape) ==2:
            self.ax.imshow(self.img_rgb, extent=[0, self.width, self.height, 0], 
                           aspect='auto', cmap = 'gray')
        else:
            self.ax.imshow(self.img_rgb, extent=[0, self.width, self.height, 0],
                           aspect='auto', interpolation='nearest')
        
        # Set limits strictly to edges
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. Save with zero padding
        self.fig.savefig(self.path, dpi=self.dpi, pad_inches=0, transparent=True)
        plt.close(self.fig)
        

def mark_pixels_alternating(img, batches: list[list[Any]], tup_idx=0, colorpairs = None):
    if colorpairs is None:
        colorpairs = itertools.cycle(PAIRED_COLORS)
    for splits_curr, colorpair in zip(batches, colorpairs):
        for i, split in enumerate(splits_curr):
            img[split[tup_idx][:, 0], split[tup_idx][:, 1]] = color_to_bgr(colorpair[i%2])

def adequate_font_size(ax, h):
    fig = ax.figure
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((0, h))

    height_px = abs(p1[1] - p0[1])
    fontsize_pt = height_px * 72 / fig.dpi
    return fontsize_pt

def plot_graph(ax, adjacency_lst, 
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
    # Plot nodes
    for _, (pos, _) in adjacency_lst.items():
        ax.scatter(pos[0], pos[1], s=point_size, color = color, zorder=2, edgecolors=None, linewidths=0,)

    if node_labels:
        sz = adequate_font_size(ax, 0.25)
        for k, txt in node_labels.items():
            pos = adjacency_lst[k][0]
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

    for node_id, (pos, neighbors) in adjacency_lst.items():
        x1, y1 = pos

        for neighbor_id in neighbors:
            if neighbor_id not in adjacency_lst:
                continue
            
            tup = tuple(sorted((node_id, neighbor_id)))
            if tup in drawn_edges and not both_directions:
                continue
            drawn_edges.add(tup)

            x2, y2 = adjacency_lst[neighbor_id][0]
            
            # if x2 < x1 and not both_directions:
            #     (x1, y1), (x2, y2) = (x2, y2), (x1, y1)

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    color=color,
                    shrinkA=0 if not both_directions else 0.02+point_size,
                    shrinkB=0 if not both_directions else 0.02+point_size,
                    arrowstyle="-" if not both_directions else "-|>, head_width=0.005, head_length=0.01",
                    linewidth=edge_width,
                    connectionstyle="arc3,rad=0.15",  # curvature
                ),
            )


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
            if M[u][v] >0:
                x1, y1 = positions[u]
                tup = tuple(sorted((u, v)))
                if tup in drawn_edges and not both_directions:
                    continue
                drawn_edges.add(tup)

                x2, y2 = positions[v]
                curr_color = "red" if M[u][v] == 1 else color
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

                      



def annotate_short_distances(ax: Axes, points: np.ndarray, d=1.0, color="white"):
    """
    Annotate pairwise distances smaller than d on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to place annotations.
    points : np.ndarray, shape (N, 2)
        Array of 2D point coordinates.
    d : float
        Distance threshold. Only distances < d are annotated.
    """
    if not len(points):
        return
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an array of shape (N, 2)")

    n = len(points)
    size_pt = adequate_font_size(ax, d/32)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]

            dist = np.linalg.norm(p2 - p1)
            if dist < d:
                print("small distance found")
                midpoint = 0.5 * (p1 + p2)

                # Reasonable rounding: 2–3 significant digits depending on scale
                label = f"{dist:.3g}"
                
                ax.text(
                    midpoint[0],
                    midpoint[1],
                    label,
                    ha="center",
                    va="center",
                    fontsize=size_pt,
                    color=color,
                )


def plot_graph_nx(ax, G, draw_edges = 1.0):
    """
    Plots the nodes and faint edges of the base graph on a specific Matplotlib axis.
    """
    pos = nx.get_node_attributes(G, 'pos')

    # Draw faint edges
    if draw_edges>0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.8, width=draw_edges)

    # Draw small nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color='black')

    # ax.set_xlim(0, w)
    # ax.set_ylim(0, h)
    ax.set_aspect('equal', adjustable='box')


def plot_paths(ax, paths):
    """
    Plots the paths on top of the graph with:
    1. Alternating solid/dashed lines.
    2. Offset start marker (square).
    3. Arrows in the middle of segments.
    """
    cmap = itertools.cycle(COLORS)

    print(f"Plotting {len(paths)} paths...")

    for i, path_coords in enumerate(paths):
        color = next(cmap)
        pts = np.array(path_coords)
        # --- 1. Lines & Arrows ---

        for j in range(len(pts) - 1):
            p_start = pts[j]
            p_end = pts[j+1]

            # Draw Line (Alternating style)
            style = '-' if j % 2 == 0 else '--'
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                    linestyle=style, linewidth=1.5, color=color, alpha=1)

        x, y = pts[:, 0], pts[:, 1]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        # Plot Arrows (Quiver)
        width = 0.0005
        hal = hl = 30

        ax.quiver(pos_x, pos_y, u/norm, v/norm,
                    color=color, pivot='mid',
                    angles='xy',
                    width=width,
                    headwidth=hl/1.5,
           headaxislength=hal,
           headlength=hl,
                     zorder=5)

        # --- 2. End Marker ---
        # Small circle at the very end
        ax.plot(pts[-1, 0], pts[-1, 1], marker='o', markersize=4,
                color=color, zorder=10)

        # --- 3. Offset Start Marker ---
        # Offset amount
        # dist_offset = 2.5 
        # # Rotate the offset angle slightly per path index to prevent stacking
        # angle_offset = (i * 45) * (np.pi / 180) 
        # dv = np.array([np.cos(angle_offset), np.sin(angle_offset)]) * dist_offset

        start_pt = pts[0]

        # Draw the marker
        ax.plot(start_pt[0], start_pt[1], marker='s', markersize=5,
                color=color, markeredgecolor='black', zorder=11,
                label=f"Path {i+1}")


def plot_tangent_vectors(ax, path_coords, vectors, color='red', scale=4.0, lw=1):
    """
    Plots tangent vectors as arrows rooted at the path nodes using ax.annotate.

    Args:
        ax: Matplotlib axis.
        path_coords: List of (x, y) tuples for the nodes in the path.
        vectors: List of (dx, dy) normalized direction tuples.
        color: Color of the arrows.
        scale: Length multiplier for the normalized vectors (visual size).
    """
    for (x, y), (dx, dy) in zip(path_coords, vectors):
        # Calculate the tip of the arrow
        end_x = x + dx * scale
        end_y = y + dy * scale

        # xy is the arrow tip, xytext is the arrow tail (root)
        ax.annotate("",
                    xy=(end_x, end_y),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->",
                                    color=color,
                                    linewidth=lw,
                                    shrinkA=0,  # Don't shrink at tail
                                    shrinkB=0)) # Don't shrink at head
 

def plot_save_points_lines(path: str, lines: list[np.ndarray] | None = None, 
    points: np.ndarray | None = None, color: str| list="white", pointcolor : str = "white", lw = 0.1, linealpha=0.25, psize=None):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_lines_and_points(ax, lines=lines,points = points,
                                    color=color, lw=1, 
                                    linealpha=1, pointcolor="black", psize=psize)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.axis("off")
    plt.savefig(path, bbox_inches='tight')

        
    

def plot_splitlines(dim, lines, sequences, points: np.ndarray, path: str,  d=1.0):
    img = np.zeros((dim, dim, 3), dtype=np.uint8) 
    mark_pixels_alternating(img, sequences, tup_idx=0)
    
    with ImageOverlay(path, img) as ax:
         plot_lines_and_points(ax, lines, points)
         annotate_short_distances(ax, points, d=d)  


def plot_half_edges(dim: int, M, half_edges, lines, all_ipoints, half_edge_pos, path:str, annotate = False):

    img = np.zeros((dim, dim, 3), dtype=np.uint8)
    mark_pixels_alternating(img, half_edges, tup_idx=0)
    
    node_labels = {i: str(i) for i in range(len(M))} if annotate else {}
    
    with ImageOverlay(path, img) as ax:
        plot_lines_and_points(ax, lines, all_ipoints)
        plot_graph_from_M(ax, M, half_edge_pos, point_size=0.2, edge_width=0.05, both_directions=False, node_labels=node_labels)
        

def plot_half_edges_lines(dim: int, M, lines: dict[int, np.ndarray], all_ipoints, path:str, annotate = False):
    
    node_labels = {i: str(i) for i in range(len(M))} if annotate else {}
    positions = {i: pl[len(pl)//2] for i, pl in lines.items()}
    with ImageOverlay(path, dims=(dim, dim)) as ax:
        plot_lines_and_points(ax, list(lines.values()), color=COLORS, lw = 1, linealpha=1)
        plot_graph_from_M(ax, M, positions, point_size=0.2, edge_width=0.05, both_directions=False, node_labels=node_labels, color="black")
        