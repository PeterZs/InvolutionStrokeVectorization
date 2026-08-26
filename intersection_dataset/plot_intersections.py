import numpy as np
import matplotlib.pyplot as plt
import sampler.randompath as randompath
import sampler.intersections as intersections
import itertools
import plotutils

np.random.seed(210)




N = 1000
offset = 0
scale = 1


lines = [
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim = 500),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim = 500),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim = 500),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim = 500),
randompath.fractal_random_walk(N, [(80, 10.0), (20,1)], desired_dim = 500),
 np.array([(100, 100), (180, 180)]),
 np.array([(101, 180), (181, 100)]),
]

lines_d = [randompath.densify_polyline_maxdist(pl, max_dist=0.5) for pl in lines]

res = intersections.compute_intersections(lines_d)
all_ipoints = np.array([x.point for i in res for x in i])

propagated_indexes =intersections.propagate_intersections_multi(lines_d, res, d=1)

plt.figure(figsize=(14, 8.5))
colors = itertools.cycle(plotutils.PAIRED_COLORS)
for pl, ipoints_curr, indexes, colorpair, in zip(lines_d, res, propagated_indexes, colors):
    # splitted = intersections.split_up(pl, ipoints_curr)
    plt.plot(*pl.T, color=colorpair[0], alpha=1, lw=1)
    # for idx, p in enumerate(splitted):
    #     color = colorpair[idx%2]
    #     plt.plot(*p.T, color=color, alpha=0.7, lw=1.1)
    for i in intersections.split_unit_progressions(indexes):
        subp = pl[i]
        plt.plot(*subp.T, color="red", lw=2)


plt.plot(*all_ipoints.T, color='black', alpha=0.7, marker='o', markersize=2.5, linewidth=0)


# plt.title("2 random walks")


plt.axis('equal')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()