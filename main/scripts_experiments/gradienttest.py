import cv2
import numpy as np
import torch
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



def sample_gradient(img, points):
    """Bilinearly-interpolated gradient at float (x, y) points.

    img: (H, W) grayscale tensor or array.
    points: (N, 2) array of (x, y) coordinates in image space.
    Returns: (N, 2) tensor of (gx, gy).
    """
    if not torch.is_tensor(img):
        img = torch.as_tensor(img, dtype=torch.float32)
    else:
        img = img.to(torch.float32)
    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)
    else:
        points = points.to(torch.float32)

    H, W = img.shape
    x = points[:, 0]
    y = points[:, 1]

    x0 = torch.floor(x).long().clamp(0, W - 2)
    y0 = torch.floor(y).long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0.to(x.dtype)).clamp(0.0, 1.0)
    fy = (y - y0.to(y.dtype)).clamp(0.0, 1.0)

    # Central differences along x using the 2x2 cell extended by one pixel.
    xL = (x0 - 1).clamp(0, W - 1)
    xR = (x1 + 1).clamp(0, W - 1)
    yU = (y0 - 1).clamp(0, H - 1)
    yD = (y1 + 1).clamp(0, H - 1)

    # gx at the 4 corners of the cell
    gx00 = 0.5 * (img[y0, x1] - img[y0, xL])
    gx10 = 0.5 * (img[y0, xR] - img[y0, x0])
    gx01 = 0.5 * (img[y1, x1] - img[y1, xL])
    gx11 = 0.5 * (img[y1, xR] - img[y1, x0])

    # gy at the 4 corners of the cell
    gy00 = 0.5 * (img[y1, x0] - img[yU, x0])
    gy10 = 0.5 * (img[y1, x1] - img[yU, x1])
    gy01 = 0.5 * (img[yD, x0] - img[y0, x0])
    gy11 = 0.5 * (img[yD, x1] - img[y0, x1])

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    gx = w00 * gx00 + w10 * gx10 + w01 * gx01 + w11 * gx11
    gy = w00 * gy00 + w10 * gy10 + w01 * gy01 + w11 * gy11
    return torch.stack([gx, gy], dim=-1)


def sample_bilinear(img, points):
    """Bilinear sampling of img at float (x, y) points via grid_sample.

    img: (H, W) grayscale tensor or array.
    points: (N, 2) tensor of (x, y) coordinates in image space.
    Returns: (N,) tensor of sampled values.
    """
    if not torch.is_tensor(img):
        img = torch.as_tensor(img, dtype=torch.float32)
    else:
        img = img.to(torch.float32)
    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)
    else:
        points = points.to(torch.float32)

    H, W = img.shape
    # align_corners=False: pixel centers at (j+0.5, i+0.5) image coords. Image spans [0, W] x [0, H].
    nx = 2.0 * points[:, 0] / W - 1.0
    ny = 2.0 * points[:, 1] / H - 1.0
    grid = torch.stack([nx, ny], dim=-1).view(1, -1, 1, 2)
    inp = img.view(1, 1, H, W)
    out = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return out.view(-1)


def sample_gradient_autograd(img, points):
    """Gradient of sample_bilinear w.r.t. point coords, via autograd."""
    if not torch.is_tensor(points):
        pts = torch.as_tensor(points, dtype=torch.float32)
    else:
        pts = points.to(torch.float32)
    pts = pts.detach().clone().requires_grad_(True)
    vals = sample_bilinear(img, pts)
    (grad,) = torch.autograd.grad(vals.sum(), pts)
    return grad


def spawn_points_at_nonzero(img):
    """Return an (N, 2) tensor of (x, y) at every nonzero pixel of img."""
    img = torch.as_tensor(img, dtype=torch.float32)
    ys, xs = torch.nonzero(img, as_tuple=True)
    return torch.stack([xs.to(torch.float32), ys.to(torch.float32)], dim=-1)


def neighborhood_climb(img, points, k, step=1.0):
    """Iteratively move each point toward the max value in its 8-neighborhood.

    The step magnitude is proportional to (v_max_neighbor - v_center).
    Direction is the unit vector toward the chosen neighbor.

    img: (H, W) grayscale image (numpy or torch).
    points: (N, 2) tensor/array of (x, y) starting positions.
    k: number of iterations.
    step: scalar multiplier on the value-difference magnitude.
    Returns: (N, 2) tensor of final (x, y) positions.
    """
    if not torch.is_tensor(img):
        img_t = torch.as_tensor(img, dtype=torch.float32)
    else:
        img_t = img.to(torch.float32)
    if not torch.is_tensor(points):
        pts = torch.as_tensor(points, dtype=torch.float32).clone()
    else:
        pts = points.to(torch.float32).clone()
    H, W = img_t.shape

    offsets = torch.tensor(
        [[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        dtype=torch.float32,
    )
    center_idx = 4
    norms = offsets.norm(dim=1, keepdim=True)
    norms[center_idx] = 1.0
    offsets_unit = offsets / norms

    for _ in range(k):
        N = pts.shape[0]
        sample_pts = (pts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 2)
        vals = sample_bilinear(img_t, sample_pts).reshape(N, 9)
        v_center = vals[:, center_idx]
        vals_neighbors = vals.clone()
        vals_neighbors[:, center_idx] = float("-inf")
        v_max, idx_max = vals_neighbors.max(dim=1)
        dv = (v_max - v_center).clamp(min=0)
        direction = offsets_unit[idx_max]
        pts = pts + step * dv.unsqueeze(1) * direction
        pts[:, 0].clamp_(0.0, W - 1.0)
        pts[:, 1].clamp_(0.0, H - 1.0)
    return pts


def gradient_descent(img, points, k, step=0.5, ascent=True):
    """Run k gradient steps on image values starting from given points.

    img: (H, W) grayscale image (numpy or torch).
    points: (N, 2) tensor/array of (x, y) starting positions.
    k: number of iterations.
    step: step size (in pixel units).
    ascent: if True, walks toward higher values; else toward lower values.
    Returns: (N, 2) tensor of final (x, y) positions.
    """
    if not torch.is_tensor(img):
        img_t = torch.as_tensor(img, dtype=torch.float32)
    else:
        img_t = img.to(torch.float32)
    if not torch.is_tensor(points):
        pts = torch.as_tensor(points, dtype=torch.float32).clone()
    else:
        pts = points.to(torch.float32).clone()
    H, W = img_t.shape

    sign = 1.0 if ascent else -1.0
    for _ in range(k):
        g = sample_gradient_autograd(img_t, pts)
        pts = pts + sign * step * g
        pts[:, 0].clamp_(0.0, W - 1.0)
        pts[:, 1].clamp_(0.0, H - 1.0)
    return pts



img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

initial_points = spawn_points_at_nonzero(img) + 0.5
moved_points = neighborhood_climb(img, initial_points, k=2, step=1.0 / 255.0)

initial_np = initial_points.cpu().numpy()
moved_np = moved_points.cpu().numpy()

# H_img, W_img = img.shape
# xs_grid = torch.arange(W_img, dtype=torch.float32) + 0.5
# ys_grid = torch.arange(H_img, dtype=torch.float32) + 0.5
# yy, xx = torch.meshgrid(ys_grid, xs_grid, indexing="ij")
# grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
# grid_grads = sample_gradient_autograd(img, grid_points).cpu().numpy()
# # grid_grads = sample_gradient(img, grid_points)
# grid_pts_np = grid_points.cpu().numpy()
# mag = np.hypot(grid_grads[:, 0], grid_grads[:, 1])
# mask = mag >1e-4
# arrow_len = 0.5  # pixels
# ux = grid_grads[mask, 0]/180 * arrow_len
# uy = grid_grads[mask, 1]/180 * arrow_len

with plu.ImageOverlay("gradienttest_overlay.svg", cv2_img=img) as ax:
    # ax.quiver(grid_pts_np[mask, 0], grid_pts_np[mask, 1], ux, uy,
    #           color="cornflowerblue", angles="xy", scale_units="xy", scale=1,
    #           width=0.06, headwidth=3, headlength=3, headaxislength=2.5,
    #           units="xy", zorder=1)
    ax.scatter(initial_np[:, 0], initial_np[:, 1],
               c="burlywood", s=0.5, zorder=2, edgecolors="none")
    ax.scatter(moved_np[:, 0], moved_np[:, 1],
               c="red", s=0.5, zorder=3, edgecolors="none")