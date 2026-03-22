import volSkewData as data
import numpy as np
# import matplotlib.pyplot as plt
import math, bisect, os
import xlwings as xw
import matplotlib.pyplot as plt

# Part (a)
def cubicSpline(n,knots):
#
#  convert into x and y with labels [1 to n]
    x=[0 for i in range(n+1)]
    y=[0 for i in range(n+1)]
    for i in range(1,n+1):
        x[i]=knots[i-1,0]
        y[i]=knots[i-1,1]     
#
    L=[0 for i in range(4*(n-1)+1)]
    M=[[0 for j in range(4*(n-1)+1)] for i in range(4*(n-1)+1)]
#
#  define the column vector L
    for i in range(1,n):
        L[i]=y[i]
        L[n-1+i]=y[i+1]
#
#  define row pointers
    k=[0,n-1,2*(n-1),3*(n-1)-1,4*(n-1)-2,4*(n-1)-1]
#
#  define the entries of M in the first (n-1) rows
    for i in range(1,n):
        M[k[0]+i][4*(i-1)+1]=1
        M[k[0]+i][4*(i-1)+2]=x[i]        
        M[k[0]+i][4*(i-1)+3]=x[i]**2
        M[k[0]+i][4*(i-1)+4]=x[i]**3     
#
#  define the entries of M in the second (n-1) rows
    for i in range(1,n):
        M[k[1]+i][4*(i-1)+1]=1
        M[k[1]+i][4*(i-1)+2]=x[i+1]        
        M[k[1]+i][4*(i-1)+3]=x[i+1]**2
        M[k[1]+i][4*(i-1)+4]=x[i+1]**3       
#
#  define the entries of M in the following (n-2) rows
    for i in range(1,n-1):
        M[k[2]+i][4*(i-1)+2]=1
        M[k[2]+i][4*(i-1)+3]=2*x[i+1]        
        M[k[2]+i][4*(i-1)+4]=3*x[i+1]**2
        M[k[2]+i][4*(i-1)+6]=-1 
        M[k[2]+i][4*(i-1)+7]=-2*x[i+1]
        M[k[2]+i][4*(i-1)+8]=-3*x[i+1]**2
#
#  define the entries of M in the next (n-2) rows
    for i in range(1,n-1):
        M[k[3]+i][4*(i-1)+3]=2
        M[k[3]+i][4*(i-1)+4]=6*x[i+1]        
        M[k[3]+i][4*(i-1)+7]=-2
        M[k[3]+i][4*(i-1)+8]=-6*x[i+1] 
#
#  define the entries of M in the last 2 rows
    M[k[4]+1][3]=2
    M[k[4]+1][4]=6*x[1]
#
    M[k[5]+1][4*(n-2)+3]=2
    M[k[5]+1][4*(n-2)+4]=6*x[n]
#
#  solve the matrix equation for R
    # R=mop.solveAxb(M,L,4*(n-1),1,1,1)
    # print(f"Len of R: {len(R)}, len of M: {len(M)}, len of L: {len(L)}")
    # print(R)
    # print(L[0])
    M = np.array(M, dtype=float)
    M_eff = M[1:,1:]
    R = np.linalg.solve(M_eff, L[1:])
    R = np.insert(R, 0, 0)  # Insert the first element back to align with original indexing
#
    splineCoeff=[]
    for i in range(1,n):
        coeff=[R[4*(i-1)+1],R[4*(i-1)+2],R[4*(i-1)+3],R[4*(i-1)+4]]
        splineCoeff.append(coeff)
#
    return splineCoeff

def impliedVol(K, t):
    # Check whether S0, r and volSkewCoeff are already in globals, if not, load them
    if 'S0' not in globals() or 'r' not in globals() or 'volSkewCoeff' not in globals():
        globals()['S0'], globals()['r'], globals()['volSkewCoeff'] = data.volSkewData()
    #
    n = len(volSkewCoeff)
    H = np.array(volSkewCoeff)

    # Compute the local volatility using cubic spline interpolation 
    # on the implied volatility surface.
    # Skip the computation of local volatility 
    # if the parameters g has been computed before to save time.
    if 'g' not in globals() or g.shape != (n - 1, 4, 3):
        globals()['g'] = np.zeros((n - 1, 4  , 3))
        for j in range(3):
            knots = H[:,[0,j+1]]
            g[:,:,j] = cubicSpline(n, knots)
    # X is moneyness
    X = math.atan((math.log(K/(S0*math.exp(r*t)))))
    # Use bisect to find the right interval for t in H[:,0]
    idx = bisect.bisect_left(H[:,0], t)
    if idx == 0:
        t0 = H[0,0]
        h1 = (g[0, 1, 0] + 2*g[0, 2, 0]*t0 + 3*g[0, 3, 0]*t0**2)*(t - t0) + H[0,1]
        h2 = (g[0, 1, 1] + 2*g[0, 2, 1]*t0 + 3*g[0, 3, 1]*t0**2)*(t - t0) + H[0,2]
        h3 = (g[0, 1, 2] + 2*g[0, 2, 2]*t0 + 3*g[0, 3, 2]*t0**2)*(t - t0) + H[0,3]
        v = h1 + h2*X + h3*X**2
    elif idx == n: 
        tn = H[n-1,0]
        h1 = (g[n-2, 1, 0] + 2*g[n-2, 2, 0]*tn + 3*g[n-2, 3, 0]*tn**2)*(t - tn) + H[n-1,1]
        h2 = (g[n-2, 1, 1] + 2*g[n-2, 2, 1]*tn + 3*g[n-2, 3, 1]*tn**2)*(t - tn) + H[n-1,2]
        h3 = (g[n-2, 1, 2] + 2*g[n-2, 2, 2]*tn + 3*g[n-2, 3, 2]*tn**2)*(t - tn) + H[n-1,3]
        v = h1 + h2*X + h3*X**2
    else:
        h1 = g[idx-1, 0, 0] + g[idx-1, 1, 0]*t + g[idx-1, 2, 0]*t**2 + g[idx-1, 3, 0]*t**3
        h2 = g[idx-1, 0, 1] + g[idx-1, 1, 1]*t + g[idx-1, 2, 1]*t**2 + g[idx-1, 3, 1]*t**3
        h3 = g[idx-1, 0, 2] + g[idx-1, 1, 2]*t + g[idx-1, 2, 2]*t**2 + g[idx-1, 3, 2]*t**3
        v = h1 + h2*X + h3*X**2
    return v
    
def localVol(K, t: int):
    delta = 1e-4
    vkt = impliedVol(K, t)
    KvK = (impliedVol(K * (1 + delta), t) - impliedVol(K * (1 - delta), t))/2/delta
    K2vK2 = (impliedVol(K * (1 + delta), t) + impliedVol(K * (1 - delta), t) - 2 * vkt)/(delta**2)
    tKt = (impliedVol(K, t * (1 + delta)) - impliedVol(K, t * (1 - delta)))/2/delta
    b = (math.log(S0/K) + (r + .5*vkt**2) * t)/ vkt
    upper_part = vkt ** 2 + 2 * vkt * (tKt + r * t * KvK)
    lower_part = (1 + b * KvK) ** 2 + vkt * t * (K2vK2 - b * KvK ** 2)
    sigma2 = max(upper_part/(lower_part + 1e-10),1e-12)
    return np.sqrt(sigma2)

    

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
    dt = T / imax
    S_grid = np.array([j * dS for j in range(jmax + 1)], dtype=float)
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
        sigma = localVol(S_j, tau)

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
        jmax, imax = 100, 100
        T, K, f_r, L, q, DeltaS = 1.0, 230.0, 0.043, 150, 5, 0.005
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


    

