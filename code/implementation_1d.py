import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits  import mplot3d


from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

import seaborn


def implementation():
    """
    Full 1D implementation of the reaction diffusion
    equations. Discretize the length of the synaptic
    cleft with an initial concentration of neurotransmitters
    at the beginning, and an pseudo concentration of receptors
    at the end. The diffusion is modelled with a Crank-Nicolson
    implisit scheme, while the reaction part is computed with 
    a simple forward Euler iteration.
    """

    # Define scheme
    L = 15e-9  # Length of the synaptic cleft (z-axis)
    rad = 0.22e-6  # Radius of the synaptic cleft 
    nN = 360  # Initial concentration of neurotransmitters
    denR = 1000e12  # Initial receptor density at the membrane
    nR = 11  # Initial concentration of unbound receptors
    # nR = denR * 1e-6  # Receptor density in one dimension
    alphaN = 8e-7  # Diffusion coefficient for the neurotransmitters
    k_on = 4e6  # Forward reaction constant
    k_off = 5   # Backward reaction ocnstant

    # Define grid
    nz = 100       # Number of grid points
    nt = 7000       # Number of time iterations
    z = np.linspace(0, L, nz)  # z-axis
    dz = z[1] - z[0]    # Grid spacing
    dt = 1e-11     # Iteration interval in the time dimension
    N = np.zeros(shape = (nz, nt))  # Neurotransmitters
    R = np.zeros(shape = N.shape)   # Open receptors
    RN = np.zeros(shape = N.shape)  # Bound receptors
    r = alphaN*dt/dz**2
    b = 1/2 + np.sqrt(3)/3


    # Initial values
    N[1, 0] = nN
    R[-1, 0] = nR

    # Run simulation 
    for i in range(1, nt):
        # Update receptor concetrations
        R[:,i] = R[:,i-1] + dt * (-k_on*N[:,i-1]*R[:,i-1] +
                                     k_off*RN[:,i-1])
        RN[:,i] = RN[:,i-1] - (R[:,i] - R[:,i-1])
        
        # Update neurotransmitter concentrations using 
        # Crank-Nicolson
        C = dt*k_off/2*(RN[:,i] - RN[:,i-1])
        V = r/2
        V = r
        T = dt*k_on/4*(R[:,i] + R[:,i-1])
        W = 2*V*np.ones(nz) + T
        Z = -2*V*np.ones(nz) + T

        A = spdiags([np.full(nz, -V), np.full(nz, 1) + W, np.full(nz, -V)],
                    diags = [-1,0,1], format = "lil")
        A[0, 0] = A[-1, -1] = 1 + V/b + T[0]
        A[0, 1] = A[-1, -2] = -V/b
        A = A.tocsr()

        B = spdiags([np.full(nz, V), np.full(nz, 1) + Z, np.full(nz, V)],
                    diags = [-1,0,1], format = "lil")
        B[0, 0] = B[-1, -1] = 1 - V/b + T[-1]
        B[0, 1] = B[-1, -2] = V/b
        B = B.toarray()
        B = np.identity(nz)

        # Compute right hand side 
        rhs = np.matmul(B, N[:,i-1]) + C

        # Solve system
        N[:,i] = np.linalg.solve(A.toarray(), rhs)
    
    
    # Find the transition time
    trans_t = np.argmax(RN[-1,:] / nR > 0.5)

    plt.figure(figsize = (12, 5))
    plot_receptors(nt, dt, RN, nR, R, trans_t)
    # plt.savefig("./figures/receptors.pdf", bbox_inches = "tight")
    plt.show()

     
    lim = 50 
    plt.figure(figsize = (12, 5))
    plot_neurotransmitter_end(N[:,:lim], lim, dt)
    # plt.savefig("./figures/neurotransmitters.pdf", bbox_inches = "tight")
    plt.show()


def plot_receptors(nt, dt, RN, nR, R, trans_t):
    """
    Plot evolution of bound and unbound receptors
    at the membrane as a function of time.
    """
    t = np.arange(nt) * dt
    plt.plot(t*1e9, RN[-1,:] / nR,
             label = "Proportion of bound receptors")
    plt.plot(t*1e9, R[-1,:] / nR,
             label = "Proportion of unbound receptors")
    plt.vlines(t[trans_t]*1e9, 0, 1, colors = "red",
               linestyles = "dashed", alpha = 0.7,
               label = "Transmission time: " + 
               str(round(t[trans_t]*1e9, 5)) + "ns")
    plt.xlabel(r'$t$' + " [ns]")
    plt.legend()



def plot_neurotransmitter_end(N, nt, dt):
    """
    Plot the concentration of neurotransmitters
    at the membrane as a function of time.
    """
    t = np.arange(nt) * dt
    plt.plot(t*1e9, N[-1,:], label = r'$[N](L, t)$')
    plt.xlabel(r'$t$' + " [ns]")
    plt.legend()


def main():
    seaborn.set_theme()
    implementation()


if __name__ == "__main__":
    main()
