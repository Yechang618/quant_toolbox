##################################
# Financial Computing Assignment 2
# Group Members:
# Chen HU (EID: chu232, Student ID: 60078083)
# Chang YE (EID: changye3, Student ID: 60020360)
# Jiamin SHI (EID: jiaminshi7, Student ID: 60122001)
# Muchun CHENG (EID: muchcheng2, Student ID: 60025803)
# Rongxi ZHU (EID: rongxizhu6, Student ID: 59666664)
###### Instruction ################
# If you want to interact with Excel, make sure gui2.xlsx is in the same folder,
# and install the add-in thru command: xlwings addin install
# Otherwise, the code will run with default parameters.
###################################

import numpy as np
import matplotlib.pyplot as plt
from volSkewData import volSkewData
import os
import xlwings as xw

# (a)
def impliedVol(K_q,tau_q):
    ## calculate volatility surface
    assetPrice, r_f, volSkewCoeff = volSkewData()
    volSkewCoeff = np.array(volSkewCoeff)
    tau = volSkewCoeff[:,0]
    h1 = volSkewCoeff[:,1]
    h2 = volSkewCoeff[:,2]
    h3 = volSkewCoeff[:,3]
    x_inter = assetPrice*np.exp(r_f*tau)
    x = np.arctan(np.log(K_q/x_inter))
    v = h1 + h2*x + h3* x**2
    n = len(x)

    ## cubic polynomials parameters
    h = np.diff(tau)
    A = np.zeros((n-2,n-2))
    b = np.zeros(n-2)
    M = np.zeros(n)
    for k in range(n-2):
        if k > 0:
            A[k, k-1] = h[k]
        A[k, k] = 2*(h[k] + h[k+1])
        if k < n-3:
            A[k, k+1] = h[k +1]
        b[k] = 6 * ((v[k+2] - v[k+1]) / h[k+1] - (v[k+1] - v[k]) / h[k])
    M[1:n-1] = np.linalg.solve(A,b)
    M[0] = 0.0
    M[n-1] = 0.0    
    # left side 0 < tau_q < tau_1
    if tau_q <= tau[0]:
        phi_prime_l = (v[1] - v[0]) / h[0] - h[0]*(2*M[0]+ M[1]) / 6
        v_inter = phi_prime_l*(tau_q - tau[0]) + v[0]
        return v_inter
    # right side tau_n < tau_q
    if tau_q > tau[n-1]:
        phi_prime_r = (v[n-1] - v[n-2]) / h[n-2] + h[n-2]*(M[n-2] + 2*M[n-1])/6
        v_inter = phi_prime_r * (tau_q - tau[n-1]) + v[n-1]
        return v_inter
    # interpolation for tau_k < tau_q < tau_{k+1}
    k = np.searchsorted(tau,tau_q) - 1
    part1 = M[k]*(tau[k+1] - tau_q)**3 / (6*h[k])
    part2 = M[k+1]*(tau_q - tau[k])**3 / (6*h[k])
    part3 = (v[k] - M[k]*h[k]**2/6) * ((tau[k+1]-tau_q)/h[k])
    part4 = (v[k+1] - M[k+1]*h[k]**2/6) * ((tau_q - tau[k])/h[k])
    v_inter = part1 + part2 + part3 + part4
    return v_inter


print(impliedVol(230, 0.5))


#(b)
def localVol(S,t):
    delta = 1e-4
    assetPrice, r_f,_= volSkewData()
    v0 = impliedVol(S,t)
    v01 = impliedVol(S*(1+delta),t)
    v02 = impliedVol(S*(1-delta),t)
    v03 = impliedVol(S,t*(1+delta))
    v04 = impliedVol(S,t*(1-delta))
    KvK = (v01 - v02) / (2*delta)
    KvK2 = (v01 - 2*v0 + v02)/ (delta**2)
    tvt = (v03 - v04) / (2*delta)
    b = (np.log(assetPrice/S) + (r_f + 0.5*v0**2)*t) / v0
    numerator = v0**2 + 2*v0*(tvt + r_f*t*KvK)
    denominator = (1 + b*KvK)**2 + v0*t*(KvK2 - b*KvK**2)
    if denominator <= 0:
        return np.nan
    sigma = np.sqrt(max(numerator/denominator, 1e-12))
    return sigma

print(localVol(230, 0.5))



#(c)
def fdpricing(jmax, imax, K, f_r, T, L, q, DeltaS) :
    jmax = 100
    imax = 100
    T = 1.0
    K = 230
    L = 150
    q = 0.01
    f_r = 0.043
    DeltaS = 5.0

    S_grid, t_grid, ds, dt = construct_grid(jmax, imax, DeltaS, T)
    F = ini_price(jmax, K, ds, L, q)
    S = np.zeros(jmax + 1)
    for i in range(imax - 1, -1, -1):
        P, Q = PQ_matrix(jmax, S_grid, T, f_r, i, dt)
        # print('P matrix is',P)
        # print('Q matrix is',Q)
        F = np.linalg.solve(P, Q @ F)

        for j in range(jmax + 1):
            S[j] = j * ds
            if S[j] <= L:
                F[j] = q * max(K - S[j], 0.0)
            else:
                F[j] = max(F[j], max(K - S[j], 0.0))
        F[jmax] = 0.0
    return S,F


def construct_grid(jmax, imax, dS, T):
    S_max = jmax * dS
    dt = T / imax
    S_grid = np.array([j * dS for j in range(jmax + 1)], dtype=float)
    t_grid = np.array([i * dt for i in range(imax + 1)], dtype=float)
    return S_grid, t_grid, dS, dt


def ini_price(jmax, K, ds, L, q):
    F = np.zeros(jmax + 1)
    for j in range(jmax + 1):
        S = j * ds
        if S <= L:
            F[j] = q * max(K - S, 0.0)
        else:
            F[j] = max(K - S, 0.0)
    return F


def localvolatility(tau, s, f_r):
    # use local volatility from part (b)
    return localVol(s, tau)


def PQ_matrix(m, S_grid, T, f_r, i, dt):
    P = np.zeros((m + 1, m + 1))
    Q = np.zeros((m + 1, m + 1))

    P[0, 0] = 1.0
    Q[0, 0] = 1.0
    P[m, m] = 1.0
    Q[m, m] = 1.0

    tau = T - (i + 0.5) * dt

    for j in range(1, m):
        S_j = S_grid[j]
        sigma = localvolatility(tau, S_j, f_r)
        #print("sigma is:", sigma)
        if not np.isfinite(sigma) or sigma < 0:
            #print("bad sigma at", i, j, tau, S_j, sigma)
            sigma = 0.2   # temporary fallback

        a = 0.5 * (f_r * j * dt) - 0.5 * (sigma**2 * j**2 * dt)
        c = -0.5 * (f_r * j * dt) - 0.5 * (sigma**2 * j**2 * dt)
        d = (f_r * dt) + (sigma**2 * j**2 * dt)

        P[j, j - 1] = 0.5 * a
        P[j, j]     = 1.0 + 0.5 * d
        P[j, j + 1] = 0.5 * c

        Q[j, j - 1] = -0.5 * a
        Q[j, j]     = 1.0 - 0.5 * d
        Q[j, j + 1] = -0.5 * c

    return P, Q


def main():
    path = os.getcwd()
    print("Current working directory:", path)
    filename = os.path.join(path,'gui2.xlsx')
    try:
        with xw.App() as app:
            sht = xw.books.open(filename).sheets['SKO American put option']
            jmax = int(sht.range('B2').value)
            imax = int(sht.range('B3').value)
            T = sht.range('B4').value
            K = sht.range('B5').value
            L = sht.range('B6').value
            q = sht.range('B7').value
            f_r = sht.range('B8').value
            DeltaS = sht.range('B9').value
        print(f"Excel file ({filename}) read successfully.")
        write_to_excel = True
    except Exception as e:
        print("Error reading input from Excel:", e)
        print("Using default input parameters.")
        jmax, imax = 1000, 100
        T, K, f_r, L, q, DeltaS = 1.0, 1.0, 0.025, 0.5, 0.01, 0.005
        write_to_excel = False
    print(f"Input parameters: jmax={jmax}, imax={imax}, T={T}, K={K}, L={L}, q={q}, f_r={f_r}, DeltaS={DeltaS}")

    S, F = fdpricing(jmax, imax, K, f_r, T, L, q, DeltaS) 

    assetPrice, r_f,_= volSkewData()
    
    # Output option prices
    for i in range(0,imax+1):
        print('%6.2f%3s%8.6f'%(S[i],' , ',F[i]))

    # Write results back to Excel if read was successful
    if write_to_excel:
        app = xw.App()
        sht = xw.books.open(filename).sheets['SKO American put option']
        sht.range('E2').options(transpose=True).value = S
        sht.range('F2').options(transpose=True).value = F

    # Visualization
    plt.plot(S,F)
    plt.xlabel('Asset Price')
    plt.ylabel('Option Price')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()