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
r-position and j for z-position. We have chosen the map idx = j + i*nr. 
'''

'''
Model coeffiicients and constants
'''

alpha = 8*10**(-7) # m^2 s^-1, Diffusion coefficient for the neurotransmitters
Rr = 0.22*10**(-6) # m, Radius of the synaptic cleft 
Z = 15*10**(-9) # m, Length of the synaptic cleft (z-axis)
T =  alpha / Z**2 # s

nN = 5000  # Number of released neurotransmitters
denR = 1000e12  # Initial receptor density at the membrane

k_on = 4e6  # Forward reaction constant
k_off = 5   # Backward reaction ocnstant

nt = 3
nr = 5
nz = 5

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
def RK1_R(N, R, NR, t, dt):
    return dt * ( -2*k_on*N[:,t]*R[:,t] + k_off*NR[:,t])
    
def RK2_R(N, R, NR, t, dt, RK1):
    return dt * ( -2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] + RK1/2) + k_off*(NR[:,t] + RK1/2) )

def RK3_R(N, R, NR, t, dt, RK2)  :
    return dt * ( -2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] + RK2/2) + k_off*(NR[:,t] + RK2/2) )

def RK4_R(N, R, NR, t, dt, RK3):
    return dt * ( -2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] + RK3/2) + k_off*(NR[:,t] + RK3/2) )
    

def RK1_RN(N, R, NR, t, dt):
    return dt * ( 2*k_on*((N[:,t+1] + N[:,t])/2)*R[:,t] - k_off*NR[:,t])

# Disse tre er egentlig helt like
def RK2_RN(N, R, NR, t, dt, RK1):
    return dt * ( 2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] - RK1/2) - k_off*(NR[:,t] + RK1/2) )

def RK3_RN(N, R, NR, t, dt, RK2)  :
    return dt * ( 2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] - RK2/2) - k_off*(NR[:,t] + RK2/2) )

def RK4_RN(N, R, NR, t, dt, RK3):
    return dt * ( 2*k_on*((N[:,t+1] + N[:,t])/2)*(R[:,t] - RK3/2) - k_off*(NR[:,t] + RK3/2) )


def RK(N, R, NR, t, dt):
    # print("N", N)
    # print("R", R)
    # print("NR", NR)
    
    RK1 = RK1_R(N, R, NR, t, dt)
    RK2 = RK2_R(N, R, NR, t, dt, RK1)
    RK3 = RK3_R(N, R, NR, t, dt, RK2)
    RK4 = RK4_R(N, R, NR, t, dt, RK3)
    
    #print("RK1", RK1)
    #print("RK2", RK2)
    #print("RK3", RK3)
    #print("RK4", RK4)
    
    R[:,t+1] = R[:,t] + (1/6)*(RK1 + 2*RK2 + 2*RK3 + RK4)
    
    RK1 = RK1_RN(N, R, NR, t, dt)
    RK2 = RK2_RN(N, R, NR, t, dt, RK1)
    RK3 = RK3_RN(N, R, NR, t, dt, RK2)
    RK4 = RK4_RN(N, R, NR, t, dt, RK3)
    
    NR[:,t+1] = NR[:,t] + (1/6)*(RK1 + 2*RK2 + 2*RK3 + RK4)
    return R, NR


'''
Functions to create A, B and C with elements according to Crank Nicholson and Neumann boundary conditions.
i = 0, 1, ..., nr-1
j = 0, 1, ..., nz-1
'''
def getAandB(nr, nz, dr, dz, dt, alpha, k_on, R, t):
    #print("nr ", nr)
    #print("nz ", nz)
    print("nr*nz = ", nr*nz)
    A = np.zeros((nr*nz, nr*nz))*np.nan
    B = np.zeros((nr*nz, nr*nz))*np.nan

    for i in range(nr):
        for j in range(nz):
            idx = j + i*nz
            print("idx = ",idx)
            # tenker at ri = (i)*dr?

                       
            ### A t+1 ###   ### B t ###
            
            # i,j N(i,j)
            A[idx,idx] = 1/dt + alpha /(dz**2) + alpha*((i+0.5+i-0.5+2)*dr) / (2*((i+1)*dr)*dr**2) - (k_on/4)*(R[idx,t+1]-R[idx,t])     
            print("her = ", - (k_on/4)*(R[idx,t+1]-R[idx,t]))
            
            
            
            # i,j
            B[idx,idx] = 1/dt - alpha*((i+0.5+i+0.5+2)*dr) / (2*((i+1)*dr)*dr**2)  - alpha /(dz**2) - (k_on/4)*(R[idx,t+1]-R[idx,t])
           
            
            
            if idx+nz <= nr*nr-1:
                # i+1,j N(i+1,j)
                A[idx,idx+nz] = -alpha*((i+0.5)*dr) / (2*((i+1)*dr)*dr**2) 
            
                # i+1,j
                B[idx,idx+nz] = alpha*((i+1.5)*dr) / (2*((i+1)*dr)*dr**2)
                
                
            if idx-nz >= 0:
                # i-1,j N(i-1,j)
                A[idx,idx-nz] = alpha*((i-0.5)*dr) / (2*((i+1)*dr)*dr**2)
             
                # i-1,j
                B[idx,idx-nz] = alpha*((i+0.5)*dr) / (2*((i+1)*dr)*dr**2)
                   
            
            if  idx+1 <= nz*nr-1:
                # i,j+1 N(i,j+1)
                A[idx, idx+1] = -alpha /(2*dz**2) 
                
                # i,j+1
                B[idx, idx+1] = alpha /(2*dz**2) 
                
            if idx-1 >= 0:
                # i,j-1 N(i,j-1)
                A[idx, idx-1] =  -alpha /(2*dz**2) 
                
                # i,j-1
                B[idx, idx-1] =  alpha /(2*dz**2) 

            
            # Neumann boundary conditions in A
            # N(i = nr, j = nz, t = t) - N(i = R-1, j = z, t = t) = 0
            if idx == (nz-1):
                A[idx,idx] = -A[idx,idx-1]
                

    
    # Neumann boundary conditions in A
    # N(i = nr, j = nz, t = t) - N(i = R-1, j = z, t = t) = 0
    A[:,(nr*nz-nz):] = -A[:,(nz*nr-2*nz):(nr*nz-nz)]
            
    return  A, B

    
def getC(k_off, NR,t):
    return (k_off/2)*(NR[:,t+1]-NR[:,t])
 
   
'''
    - Find R[n+1] and (N-R)[n+1] using Runge-Kutta
    - Create A, B and C in A @ N[n+1] = B @ N[n] + C
    - Solve N[n+1] = inv(A) @ B @ N[n] + inv(A) @ C
'''
def implementation():

    N = np.zeros((nr*nz, nt))
    R = np.zeros((nr*nz, nt))
    NR = np.zeros((nr*nz, nt))
    
    # initial conditions
    # N(t=0, r=i, z=0) = nN/nr
    # R(t=0, r=i, z=Z) = rdens/nr
    for i in range(nr):
        N[i*nz,0] = nN/nr # z = 0, r = 0, 1, 2, ...
        R[nr-1+i*nz,0] = denR/nr # z = Z, r = 0, 1, 2, ...
        

    for t in np.linspace(0, nt-2, nt,dtype=(int)):
        print("N", N)
        print("R", R)
        print("NR", NR)
        # One step 
        
        # Runge Kutta 
        R, NR = RK(N, R, NR, t, dt)
        
        
        # Create A and B
        A, B = getAandB(nr, nz, dr, dz, dt, alpha, k_on, R, t)
        #print("A ", A)
        
        
        # Create C
        C = getC(k_off, NR, t)
        
        i = 0
        j = 0
        idx = j+i*nz
        print("A[idx,:] ",A[idx,:])
        #print("A",A)
        
        #print("determinant ", la.det(A))
        
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
    print("Nmatrix", Nmatrix)
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
    print("Rmatrix", Rmatrix)
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
    print("NRmatrix", NRmatrix)
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
