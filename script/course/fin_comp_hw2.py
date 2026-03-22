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
import os
import xlwings as xw
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
def fdpricing(jmax, imax, K, f_r, T, L, q, DeltaS) :

    S_grid, ds, dt = construct_grid(jmax, imax, DeltaS, T)
    F = ini_price(jmax, K, ds, L, q)
    S = np.zeros(jmax + 1)
    for i in range(imax - 1, -1, -1):
        P, Q = PQ_matrix(jmax, S_grid, T, f_r, i, dt)
        F = np.linalg.solve(P, Q @ F)
        for j in range(jmax + 1):
            S[j] = j * ds
            if S[j] <= L:
                F[j] = q * max(K - S[j], 0.0)
            else:
                F[j] = max(F[j], max(K - S[j], 0.0))
        F[jmax] = 0.0
    return S,F
#
#-------------------------------------------------------------------
def construct_grid(jmax, imax, dS, T):
    # S_max = jmax * dS
    dt = T / imax
    S_grid = np.array([j * dS for j in range(jmax + 1)], dtype=float)
    # t_grid = np.array([i * dt for i in range(imax + 1)], dtype=float)
    return S_grid, dS, dt


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
    s_eff = max(s, 1e-12)
    z = np.log(s_eff * np.exp(-f_r * tau))

    v = 0.25 - 0.05 * tau + 0.01 * tau**2 + 0.001 * z**2
    v_t = -0.05 + 0.02 * tau - 0.002 * f_r * z
    x_vx = 0.002 * z

    sigma2 = v**2 + 2 * tau * v * v_t + 2 * f_r * tau * v * x_vx
    sigma2 = max(sigma2, 1e-12)

    return np.sqrt(sigma2)


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

        a = 0.5 * (f_r * j * dt) - 0.5 * (sigma**2 * j**2 * dt)
        c = -0.5 * (f_r * j * dt) - 0.5 * (sigma**2 * j**2 * dt)
        d = (f_r * dt) + (sigma**2 * j**2 * dt)

        P[j, j - 1] = 0.5 * a
        P[j, j] = 1.0 + 0.5 * d
        P[j, j + 1] = 0.5 * c

        Q[j, j - 1] = -0.5 * a
        Q[j, j] = 1.0 - 0.5 * d
        Q[j, j + 1] = -0.5 * c

    return P, Q
#-----------------------------------------------------------------

# === main ===
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
    