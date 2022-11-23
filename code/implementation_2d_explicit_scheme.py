import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib as mpl
import scipy.signal

mpl.use("Qt5Agg")

from RungeKutta import *

print("2D heat equation solver")

nz = 50  # nz
nr = 50  # nr
nt = 100  # nt

length = 1
radius = 1

eta = 4.45e-3
mu = eta

nu = 1.24e-7
xi = 4.24e-11
delta_r = radius/nr
delta_z = length/nz

delta_t = delta_z**2 / (16*eta)

# Initialize solution: the grid of u(k, i, j)
N = np.empty((nt, nr, nz))
R = np.zeros((nt, nr, nz))
NR = np.zeros((nt, nr, nz))


# Initial condition everywhere inside the grid
u_initial = 0

u_outer = 0
u_left = 1
u_inner = 0
u_right = 0

# Set the initial conditions
N.fill(u_initial)

N[0, -1, :] = u_outer
N[0, :, 0] = u_left * np.linspace(0, 3/2, nr)
N[0, 0, 1:] = u_inner
N[0, :, -1] = u_right

R[0, :, -1] = 1 * np.linspace(0, 3/2, nr)

_k = np.zeros_like(N)
_k[:, :, -1] = 1.0
nu = _k * nu
xi = _k * xi


def calculate(N):
    for k in tqdm(range(0, nt - 1)):
        R[k+1, :, -1], NR[k+1, :, -1] = RK(N[k, :, -1], R[k, :, -1], NR[k, :, -1], delta_t)

        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                # What a mess :/
                N[k + 1, i, j] = (1 + nu[k, i, j] * (R[k + 1, i, j] + R[k, i, j]) / 4) ** (-1) * (eta * delta_t * (
                        (N[k][i + 1][j] - (2 * i * delta_r) * N[k][i][j] + N[k][i - 1][j]) / (i * delta_r ** 3)) + delta_t * ((
                        N[k][i][j + 1] - 2 * N[k][i][j] + N[k][i][j - 1]) / delta_z ** 2) + (1 - nu[k, i, j] * (
                            R[k + 1, i, j] + R[k, i, j]) / 4) * N[k][i][j] + xi[k, i, j] * (NR[k + 1, i, j] + NR[k, i,
                                                                                                                 j]) / 2)

        # Enforce neumann BCs
        N[k+1, 0, :] = N[k+1, 1, :]
        N[k+1, -1, :] = N[k+1, -2, :]
        N[k+1, :, 0] = N[k+1, :, 1]
        N[k+1, :, -1] = N[k+1, :, -2]

    return N


def plotheatmap(N_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"[N] at t = {k * delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(N_k, cmap=plt.cm.jet)
    plt.colorbar()

    return plt


# Do the calculation here
N = calculate(N)


def animate(k):
    plotheatmap(N[k], k)


anim = animation.FuncAnimation(plt.figure(), animate, interval=1000, frames=nt, repeat=True)
plt.show()
#anim.save("heat_equation_solution.gif")

print("Done!")
