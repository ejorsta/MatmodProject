#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 12:57:08 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


'''
Numerical implementation for 2D+time problem.
The idea is buildt around A N[n+1] = B N[n] + C. 
N[n] is a (nr*nz, 1) vector. 
A and B are (nr*nz, nr*nz). 

So we need a map from an index somewhere between 0 to nr*nz to an index i for 
r-position and j for z-position. We have chosen the map idx = j + i*nz. 
'''

'''
Model coeffiicients and constants
'''

alpha = 8*10**(-7) # m^2 s^-1, Diffusion coefficient for the neurotransmitters
eta = 4.45*10**(-3)
nu = 1.24*10**(-7) 
xi = 4.24*10**(-11) 

N0 = 3.65 # mol m^(-3)
R0 = 1.10*10**(-1) # mol m^(-3)

Rr = 0.22*10**(-6) # m, Radius of the synaptic cleft 
Z = 15*10**(-9) # m, Length of the synaptic cleft (z-axis)
T = Z**2 / alpha  # s


nN = 5000  # Number of released neurotransmitters
denR = 1000e12  # Initial receptor density at the membrane

#k_on = 4e6  # Forward reaction constant
#k_off = 5   # Backward reaction ocnstant

nt = 5
nr = 3
nz = 3

dt = T / nt
dr = Rr / nr
dz = Z / nz



'''
As A and B contains values for R at time n+1, and C contains values of N-R at 
time n+1, the procedure is as follows:
    - Find R[n+1] and (N-R)[n+1] using Runge-Kutta
    - Create A, B and C in A @ N[n+1] = B @ N[n] + C
    - Solve N[n+1] = inv(A) @ B @ N[n] + inv(A) @ C
    
'''


'''
Runge Kutta to find R and (N-R) densities
t = 0, 1, ..., nt-1 
'''


def RK(N, R, NR, t, nu, xi, N0, R0):
    R[:,t+1] = R[:,t] - nu * (N0/R0) * N[:,t]*R[:,t] + xi * (N0/R0) * NR[:,t]
    
    NR[:,t+1] = NR[:,t] + nu * (N0/R0) * N[:,t]*R[:,t] - xi * (N0/R0) * NR[:,t]
    return R, NR


'''
Functions to create A, B and C with elements according to Crank Nicholson and Neumann boundary conditions.
i = 0, 1, ..., nr-1
j = 0, 1, ..., nz-1
'''


def getAandB(nr, nz, dr, dz, dt, R, t, eta, nu):
    #A = np.zeros((nr*nz, nr*nz))
    B = np.zeros((nr*nz, nr*nz))
    il = np.linspace(nz, nr*nr-1, nr*nz-nz)
    iu = np.linspace(0, nr*nr-nz, nr*nz-nz)
    
    dA = 6*np.ones(nr*nz) # np.ones(nr*nz) + eta * dt/(2*dr**2) + dt /(dz**2) - nu*(dt/2)*(R[:,t+1]-R[:,t]) # i,j
    djA = 7*np.ones(nr*nz-1) # - dt /(2*dz**2) * np.ones(nr*nz-1)   #i, j+-1
    DiAupper = 8*np.ones(nr*nz-nz) # - eta * (dt/(2*iu*dr**2)) * (iu+0.05)  # i+1,j
    DiAlower = 8*np.ones(nr*nz-nz) # - eta * (dt/(2*il*dr**2)) * (il-0.05)  # i-1,j
    
    A = np.diag(dA, 0) + np.diag(DiAupper, nz) + np.diag(DiAlower, -nz) + np.diag(djA, 1) + np.diag(djA, -1) 
    #A[0,nz] = 1
    
    dB = 1 - eta * dt/ (2*dr**2) - dt /(dz**2)  - nu*(dt/2)*(R[:,t+1]-R[:,t]) # i,j
    djB = dt /(2*dz**2) * np.ones(nr*nz-1)  #i, j+-1
    DiBupper = eta * (dt/(2*iu*dr**2)) * (iu+0.05) # i+1,j
    DiBlower = eta * (dt/(2*il*dr**2)) * (il-0.05) # i-1,j
    
    B = B + np.diag(dB) + np.diag(DiBupper, nz) + np.diag(DiBlower, -nz) + np.diag(djB, 1) + np.diag(djB, -1) 
    #B[0,nz] = 1
            
    # Boundary conditions in A
    for j in range(1,nr*nz-1, nz):
        # N(i = i, j = 1, t = t) - N(i = i, j = 0, t = t) = 0
        A[:,j-1] = A[:, j]

    for j in range(0, nz):
        # N(i = i, j = nz, t = t) - N(i = i, j = nz-1, t = t) = 0
        A[:,j*nz-1] = A[:, j*nz-2] 
        
    # Boundary conditions
    # N(i = nr, j = j, t = t) - N(i = nr-1, j = j, t = t) = 0
    A[:,(nr*nz-nz):] = A[:,(nz*nr-2*nz):(nr*nz-nz)] 
    
    # N(i = 0, j = j, t = t) - N(i = 1, j = j, t = t) = 0
    A[:,nz:2*nz] = A[:,0:nz] 

    
    return A,B



    
def getC(xi, NR,t):
    return xi*(NR[:,t+1]-NR[:,t])
 
   
'''
    - Find R[n+1] and (N-R)[n+1] using Runge-Kutta
    - Create A, B and C in A @ N[n+1] = B @ N[n] + C
    - Solve N[n+1] = inv(A) @ B @ N[n] + inv(A) @ C
'''
def implementation():

    N = np.zeros((nr*nz, nt))
    R = np.zeros((nr*nz, nt))
    NR = np.zeros((nr*nz, nt))
    
    i = np.linspace(0, nr-1, nr)
    
    # initial conditions
    # N(t=0, r=i, z=0) = nN/nr
    # R(t=0, r=i, z=Z) = rdens/nr
    for i in range(nr):
        N[i*nz,0] = nN # z = 0, r = 0, 1, 2, ...
        R[nr-1+i*nz,0] = denR * i / nr# z = Z, r = 0, 1, 2, ...
        

    for t in np.linspace(0, nt-2, nt-1,dtype=(int)):
        # One step 
        
        # Runge Kutta 
        R, NR = RK(N, R, NR, t, nu, xi, N0, R0)
        
        
        # Create A and B
        A, B = getAandB(nr, nz, dr, dz, dt, R, t, eta, nu)
        
        
        # Create C
        C = getC(xi, NR, t)
        
        i = 1
        j = 1
        idx = j+i*nz
        #print("A[idx,:] ",A[idx,:])
        #print("A",A)
        
        Ainv = la.inv(A)
        
        D = Ainv @ B
        
        E = Ainv @ C    
        
        N[:,t+1] =  D @ N[:,t] + E 
        
    return N, R, NR
    

N, R, NR = implementation()
#print("N[:,0]",N[:,0])


t = 0

for t in range(nt-1):
    # plot N
    Nmatrix = N[:,t].reshape(nr, nz)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    rside = np.linspace(0, Rr, nr)
    zside = np.linspace(0, Z, nz)
    z, r = np.meshgrid(zside, rside)
    plt.pcolormesh(z, r, Nmatrix, shading='auto')
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title("Density of N at t = " + str(t))
    plt.colorbar()
    plt.show()
    
    
    # plot R
    Rmatrix = R[:,t].reshape(nr, nz)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    rside = np.linspace(0, Rr, nr)
    zside = np.linspace(0, Z, nz)
    z, r = np.meshgrid(zside, rside)
    plt.pcolormesh(z, r, Rmatrix, shading='auto')
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title("Density of R at t = " + str(t))
    plt.colorbar()
    plt.show()
    
    
    # plot NR
    NRmatrix = NR[:,t].reshape(nr, nz)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    rside = np.linspace(0, Rr, nr)
    zside = np.linspace(0, Z, nz)
    z, r = np.meshgrid(zside, rside)
    plt.pcolormesh(z, r, NRmatrix, shading='auto')
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title("Density of N-R at t = " + str(t))
    plt.colorbar()
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
