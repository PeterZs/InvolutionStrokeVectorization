import numpy as np
import matplotlib.pyplot as plt
import itertools
import sampler.randompath as randompath
import plotutils
import sampler.line_sampler as line_sampler


#10, 30, 35, 66, 512, 121, 137, 138, 141, 142, 143, 144, 145, 148, 150, 151, 152, 180, 181, 191
# np.random.seed(191)
BASE_DIR = "samples"
N = 1000
MAX_SEED = 1000
offset = 30
scale = 10

count=0
for seed in range(1, MAX_SEED):
    np.random.seed(seed)
    pl = randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], offset, scale)
    pl_d = randompath.densify_polyline_maxdist(pl, max_dist=0.5)

    pl_is_valid, reason = line_sampler.validate(pl_d, min_dist=1.05)
    if not pl_is_valid:
        count+=1
        print("seed", seed, ", invalid count:", count)
        fig, ax = plt.subplots()
        plt.plot(*pl_d.T, color="grey", linewidth=0.1)
        if "small_loop" in reason:
            center, r = reason["small_loop"]

            circle = plt.Circle(tuple(center), r, fill=False, edgecolor='black', linewidth=0.1)
            ax.add_patch(circle)
            
        else:
            indexes = reason["invalid_endpoints"]
            red_points = pl_d[indexes]
            ax.scatter(*red_points.T, c="red", s=0.1, zorder=2, edgecolors=None, linewidths=0)
  
        plt.grid()
        plt.axis('equal')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        if "small_loop" in reason:
            path = f"{BASE_DIR}/invalid_loops/seed_{seed}.svg"
        else:
            path = f"{BASE_DIR}/invalid_endpoints/seed_{seed}.svg"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
