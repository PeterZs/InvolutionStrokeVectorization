import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sampler.randompath as randompath


N = 1000
plt.figure(figsize=(14, 8.5))



# 1. Correlated (Angle)
plt.subplot(2, 3, 1)
p= randompath.correlated_random_walk(N, smoothness=0.1)
plt.plot(*p.T, color='blue', alpha=0.7, lw=1.5)
plt.title("Correlated Random Walk\n(Angle Perturbation)")
plt.axis('equal')


# 2. Gaussian Smoothed
plt.subplot(2, 3, 2)
p= randompath.fractal_random_walk(N, [(80, 10.0), (8,1)])
plt.plot(*p.T, color='green', alpha=0.7, lw=1.5)
plt.title("Gaussian Smoothed Walk (2 octaves)")
plt.axis('equal')

# 3. Hobby Splines
plt.subplot(2, 3, 3)
p = randompath.curvy_walk(N, 100)
plt.plot(*p.T, color='black', alpha=0.7, lw=1.5)
plt.title("Hobby splines from random walk")
plt.axis('equal')


plt.tight_layout()
plt.show()