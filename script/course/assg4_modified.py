##################################
# Financial Computing Assignment 4
# Group Members:
# Chen HU (EID: chu232, Student ID: 60078083)
# Chang YE (EID: changye3, Student ID: 60020360)
# Jiamin SHI (EID: jiaminshi7, Student ID: 60122001)
# Muchun CHENG (EID: muchcheng2, Student ID: 60025803)
# Rongxi ZHU (EID: rongxizhu6, Student ID: 59666664)
###### Instruction ################
###################################

import numpy as np
import irTree, statistics
import genBondData as gbd
from math import exp, sqrt, log

def loadData():
    a = [[1/12,0.01932],[3/12,0.01974],
         [6/12,0.02012],[1,0.02077],
         [2,0.02156],[4,0.02202],
         [7,0.02259],[18,0.01839]]
    b = [[1/12,0.0008],[3/12,0.002],
         [6/12,0.004],[1,0.007],
         [2,0.011],[4,0.013],
         [7,0.014],[18,0.015]]
    return a,b 

# -----------------------------------------------------------------
def bondOptionTree(T,
                   K,
                   tau,
                   par,
                   upperBarrier,
                   lowerBarrier,
                   rzero,
                   bondVol,
                   treeType,
                   prec):
#
# Generate bond price and volatility term structures with term=[0,dt,2dt,...,nTerm*dt] where nTerm*dt=underlying bond maturity 
# 
    n = int(tau/rzero[0][0])
    dt = tau/n
#
    marketBondPrices = gbd.genBondData(rzero,bondVol,dt,n+1)
    treeRate=irTree.irTree(tau,n,treeType,marketBondPrices,prec)
    print('Tree rate:')
    for j in range(0,n+1):
        for i in range(0,n+1):
            if(i<j):
                print('%6s '%(''),end=' ')
            else:
                print('%6.4f '%(treeRate[i][j]),end=' ')
        print('') 
#
    nOption = int(T / dt)
    print(f"Option maturity steps: {nOption}")
#
    Bf=[[] for i in range(0,(n+1)+1)]  
    fTree=[[] for i in range(0,nOption+1)]  
#
# Rolling back underlying bond price
#
    for j in range(n+1):
        Bf[n].append(par)
#
    for i in range(n-1,-1,-1):
        for j in range(i+1):
          Bf[i].append(exp(-treeRate[i][j]*dt)*(1/2)*(Bf[i+1][j]+Bf[i+1][j+1]))
#
# Rolling back option price starting as i=Hf
#
    for j in range(0,nOption+1):
        if lowerBarrier < Bf[nOption][j] < upperBarrier:
            fTree[nOption].append(max(Bf[nOption][j]-K,0))
        else:
            fTree[nOption].append(0.0)

    for i in range(nOption-1,-1,-1):
        for j in range(0,i+1):
            fairValue = exp(-treeRate[i][j] * dt) * 0.5 * (fTree[i + 1][j] + fTree[i + 1][j + 1])
            if lowerBarrier < Bf[i][j] < upperBarrier:
                fTree[i].append(fairValue)  # 
            else:
                fTree[i].append(0.0)  # 
    return fTree
#

#----------------------------------------------------------------------------------
def mcHoLeeZCBCall(marketBondPrices,
                   K,
                   par,
                   upperBarrier,
                   lowerBarrier,
                   timeMaturity,
                   uBondTimeMaturity,
                   n,
                   nsample):
#
    dt = timeMaturity / n
    _, P0, vol0 = marketBondPrices[1]
    r0 = - log(P0) / dt
    sigma = vol0 / dt    
    m = int(n * (uBondTimeMaturity-timeMaturity) / timeMaturity)
#
    dcfsample=[]

    for _ in range(nsample):
        r = r0
        y1 = 0
        rnd = np.random.normal(0, 1, n + m)
        knockoutFlag = False
        y1, y2, rList = get_yield_path(r0, sigma, dt, n, m, marketBondPrices, rnd)
        P_tau = par * exp(-y2)  # P(tau)
        if not (lowerBarrier < P_tau < upperBarrier):
            knockoutFlag = True
        for t in range(n - 1, -1, -1):
            P_t = P_tau * exp(-(rList[t] + rList[t+1])/2)  # P(t)
            if not (lowerBarrier < P_t*par < upperBarrier):
                knockoutFlag = True
                break
        if not knockoutFlag:
            valid_paths += 1
            dcfsample.append(exp(-y1)*payoff(P_tau,K))
        else:
            dcfsample.append(0)
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)

#
    return sampleMean, stderr
#
#----------------------------------------------------------------------------------
def payoff(bondPrice,K):
    return max(bondPrice-K,0)

#----------------------------------------------------------------------------------
def get_theta(marketBondPrices, i, dt, sigma):
    if i == 0:
        P0 = marketBondPrices[0][1]
        P1 = marketBondPrices[1][1]
        P2 = marketBondPrices[2][1]
        # f(0,t) ≈ -ln(P(t+dt)/P(t))/dt
        f0 = -log(P1 / P0) / dt
        f1 = -log(P2 / P1) / dt
        dfdt = (f1 - f0) / dt
    else:
        Pt_1 = marketBondPrices[i-1][1]
        Pt = marketBondPrices[i][1]
        Pt1 = marketBondPrices[i+1][1]
        f_prev = -log(Pt / Pt_1) / dt
        f_curr = -log(Pt1 / Pt) / dt
        dfdt = (f_curr - f_prev) / dt
        
    t = i * dt
    return dfdt + sigma**2 * t

def get_yield_path(r0, sigma, dt, n, m, marketBondPrices, rnd):
    r = r0
    y1 = 0
    y2 = 0
    rList = []
    for i in range(n + m):
        t = i * dt
        theta = get_theta(marketBondPrices, i, dt, sigma)
        r += theta * dt + sigma * sqrt(dt) * rnd[i]
        rList.append(r)
        if i < n:
            if(i == n - 1) or (i == 0): 
                y1 += r * (dt/2)
            else: y1 += r * dt
        else: 
            if (i == n) or (i == n + m - 1):
                y2 += r * (dt/2)
            else: y2 += r * dt
    return y1, y2, rList

def main():
    rzero, bondVol = loadData()
    # T: maturity of option, K: K price, 
    # tau: maturity of underlying bond, par: par value of underlying bond, 
    # upperBarrier/lowerBarrier: knockout barriers

    T, K, tau, par, upperBarrier, lowerBarrier = 1, 0.95, 2, 1, 0.98, 0.90
    treeType, prec = 'normal', 1.e-8
    n = int(T/rzero[0][0])
    m = int(n*(tau-T)/T)
    dt = T/n
    #
    marketBondPrices = gbd.genBondData(rzero,bondVol,dt,n+m+1)
    print('Market bond prices:')
    for i in range(len(marketBondPrices)):
        print('Time: %f, Bond Price: %f, Volatility: %f' %(marketBondPrices[i][0], marketBondPrices[i][1], marketBondPrices[i][2]))

    fTree = bondOptionTree(T,
                         K,
                         tau,
                         par,
                         upperBarrier,
                         lowerBarrier,
                         rzero,
                         bondVol,
                         treeType,
                         prec)
    #
    print('Binomial tree pricing: ')
    print(fTree[0][0])

    #
    #----------------------------------------------------------------------------------
    np.random.seed(42)
    #
    nSamples = [100, 1000, 10000, 100000]
    # marketBondPrices = gbd.genBondData(rzero,bondVol,dt,n + m + 1)
    #
    print('MC pricing with different sample sizes:')
    for nSample in nSamples:
        optionprice, error = mcHoLeeZCBCall(marketBondPrices,
                                            K,
                                            par,
                                            upperBarrier,
                                            lowerBarrier,
                                            T,
                                            tau,
                                            n,
                                            nSample)
        print(f"{nSample} samples, Option Price: {optionprice:.6f}, Standard Error: {error:.6f}")

    #

    return 

if __name__ == '__main__':
    main()