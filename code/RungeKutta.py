alpha = 8 * 10 ** (-7)  # m^2 s^-1, Diffusion coefficient for the neurotransmitters
Rr = 0.22 * 10 ** (-6)  # m, Radius of the synaptic cleft
Z = 15 * 10 ** (-9)  # m, Length of the synaptic cleft (z-axis)
T = alpha / Z ** 2  # s

nN = 5000  # Number of released neurotransmitters
denR = 1000e12  # Initial receptor density at the membrane

k_on = 4e6  # Forward reaction constant
k_off = 5  # Backward reaction ocnstant

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


def RK1_R(N, R, NR, dt):
    return dt * (-k_on * (N) * R + k_off * NR)


def RK2_R(N, R, NR, dt, RK1):
    return dt * (-k_on * N * (R + RK1 / 2) + k_off * (NR + RK1 / 2))


def RK3_R(N, R, NR, dt, RK2):
    return dt * (-k_on * N * (R + RK2 / 2) + k_off * (NR + RK2 / 2))


def RK4_R(N, R, NR, dt, RK3):
    return dt * (-k_on * N * (R + RK3 / 2) + k_off * (NR + RK3 / 2))


def RK1_RN(N, R, NR, dt):
    return dt * (k_on * N * R - k_off * NR)


# Disse tre er egentlig helt like
def RK2_RN(N, R, NR, dt, RK1):
    return dt * (k_on * N * (R - RK1 / 2) - k_off * (NR + RK1 / 2))


def RK3_RN(N, R, NR, dt, RK2):
    return dt * (k_on * N * (R - RK2 / 2) - k_off * (NR + RK2 / 2))


def RK4_RN(N, R, NR, dt, RK3):
    return dt * (k_on * N * (R - RK3 / 2) - k_off * (NR + RK3 / 2))


def RK(N, R, NR, dt):
    # print("N", N)
    # print("R", R)
    # print("NR", NR)

    RK1 = RK1_R(N, R, NR, dt)
    RK2 = RK2_R(N, R, NR, dt, RK1)
    RK3 = RK3_R(N, R, NR, dt, RK2)
    RK4 = RK4_R(N, R, NR, dt, RK3)

    # print("RK1", RK1)
    # print("RK2", RK2)
    # print("RK3", RK3)
    # print("RK4", RK4)

    RK1 = RK1_RN(N, R, NR, dt)
    RK2 = RK2_RN(N, R, NR, dt, RK1)
    RK3 = RK3_RN(N, R, NR, dt, RK2)
    RK4 = RK4_RN(N, R, NR, dt, RK3)

    return R + (1 / 6) * (RK1 + 2 * RK2 + 2 * RK3 + RK4), NR + (1 / 6) * (RK1 + 2 * RK2 + 2 * RK3 + RK4)