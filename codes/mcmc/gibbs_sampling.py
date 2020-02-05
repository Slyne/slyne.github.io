import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Transition():
    def __init__(self, mean, cov):
        self.mean = mean
        self.sigmas = []
        for i in range(K):
            self.sigmas.append(np.sqrt(cov[i][i]))
        self.rho = cov[0][1]/(self.sigmas[0] * self.sigmas[1])

    def sample(self, id1, id2_list, x2_list):
        id2 = id2_list[0]  # only consider two dimension
        x2 = x2_list[0]  # only consider two dimension
        cur_mean = self.mean[id1] + self.rho*self.sigmas[id1]/self.sigmas[id2] * (x2-self.mean[id2])
        cur_sigma = (1-self.rho**2) * self.sigmas[id1]**2
        return np.random.normal(cur_mean, scale=cur_sigma, size=1)[0]


def gibbs(p, m, n):
    # randomize a number
    x = np.random.rand(K)
    for t in range(0, m+n):
        for j in range(K):
            total_indexes = list(range(K))
            total_indexes.remove(j)
            left_x = x[total_indexes]
            x[j] = p.sample(j, total_indexes, left_x)

        if t >= m:
            yield x


mean = [5, 8]
cov = [[1, 0.5], [0.5, 1]]
K = len(mean)
q = Transition(mean, cov)
m = 100
n = 1000

gib = gibbs(q, m, n)

simulated_samples = []

x_samples = []
y_samples = []
for li in gib:
    x_samples.append(li[0])
    y_samples.append(li[1])


fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')

hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=100, range=[[0,10],[0,16]])
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax = fig.add_subplot(132)
ax.hist(x_samples, bins=50)
ax.set_title("Simulated on dim1")

ax = fig.add_subplot(133)
ax.hist(y_samples, bins=50)
ax.set_title("Simulated on dim2")
plt.show()