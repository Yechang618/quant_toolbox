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
import irTree, rand, statistics
# import cubicSpline as cs
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
def bondOptionTree(optionMaturity,
                   strike,
                   uBondMaturity,
                   uBondPar,
                   upperBarrier,
                   lowerBarrier,
                   uBondCoupon,
                   paymentSchedule,
                   rzero,
                   bondVol,
                   treeType,
                   prec):
#
# Generate bond price and volatility term structures with term=[0,dt,2dt,...,nTerm*dt] where nTerm*dt=underlying bond maturity 
# 
    nCoupon=len(uBondCoupon)
    timeHorizon=uBondMaturity
    print(f"rzero: {rzero}")
    n=int(timeHorizon/rzero[0][0])
    timeInc=timeHorizon/n
    print(f"timeInc: {timeInc}, n: {n}")
#
    marketBondPrices = gbd.genBondData(rzero,bondVol,timeInc,n+1)
    treeRate=irTree.irTree(timeHorizon,n,treeType,marketBondPrices,prec)
    print('Tree rate:')
    for j in range(0,n+1):
        for i in range(0,n+1):
            if(i<j):
                print('%6s '%(''),end=' ')
            else:
                print('%6.4f '%(treeRate[i][j]),end=' ')
        print('') 
#
    H=int(optionMaturity/timeInc)
    print(f"Option maturity steps: {H}")
#
    Bf=[[] for i in range(0,(n+1)+1)]  
    fTree=[[] for i in range(0,H+1)]  
#
# Rolling back underlying bond price
#
    coupon=couponPayment((n-0.5)*timeInc,(n+0.5)*timeInc,uBondCoupon,nCoupon,paymentSchedule)
    for j in range(0,n+1):
        Bf[n].append(uBondPar+coupon)
#
    for i in range(n-1,-1,-1):
        coupon=couponPayment((i-0.5)*timeInc,(i+0.5)*timeInc,uBondCoupon,nCoupon,paymentSchedule)
        for j in range(0,i+1):
          Bf[i].append(exp(-treeRate[i][j]*timeInc)*(1/2)*(Bf[i+1][j]+Bf[i+1][j+1])+coupon)
#
# Rolling back option price starting as i=Hf
#
    for j in range(0,H+1):
        fTree[H].append(payoff(Bf[H][j],strike, upperBarrier, lowerBarrier))
#
    for i in range(H-1,-1,-1):
        for j in range(0,i+1):
            fairValue=exp(-treeRate[i][j]*timeInc)*(1/2)*(fTree[i+1][j]+fTree[i+1][j+1])
            # pricing_temp = max(fairValue,payoff_mc(Bf[i][j],strike))
            # if lowerBarrier < pricing_temp < upperBarrier:
            #     fTree[i].append(pricing_temp)
            # else:
            #     fTree[i].append(0.0)
            fTree[i].append(max(fairValue,payoff(Bf[i][j],strike, upperBarrier,lowerBarrier)))
#
    return fTree
#        
#---------------------------------------------------------------------------
def couponPayment(timeLow,timeUp,uBondCoupon,nCoupon,paymentSchedule):
    coupon=0   
    for i in range(nCoupon-1,-1,-1):
        if(timeLow<paymentSchedule[i] and timeUp>=paymentSchedule[i]):
            coupon=coupon+uBondCoupon[i]
#
    return coupon
#
#---------------------------------------------------------------------------------
def payoff(assetPrice,strike, upperBarrier, lowerBarrier):
    if lowerBarrier < assetPrice < upperBarrier:
        return max(assetPrice-strike,0)
    else:
        return 0
    
#

#----------------------------------------------------------------------------------
def mcHoLeeZCBCall(marketBondPrices,
                   strike,
                   par,
                   upperBarrier,
                   lowerBarrier,
                   timeMaturity,
                   uBondTimeMaturity,
                   nstep,
                   nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    m=int(nstep*(uBondTimeMaturity-timeMaturity)/timeMaturity)
#
    dcfsample=[]
    _, P0, vol0 = marketBondPrices[1]
    r0 = - log(P0) / dt
    sigma = vol0 / dt
    print(f"Initial short rate r0: {r0}, initial volatility sigma: {sigma}")
    for Ls in range(nsample):
        r=r0
        y=r*(dt/2)
        skipFlag=False
        rnd = np.random.normal(0, 1, nstep)
        rList = []
        for i in range(nstep):
            t = i*dt
            dFdt, sigma = get_dFdt(marketBondPrices, i, dt)
            r += (dFdt + sigma * sigma * t) * dt + sigma * sqrt(dt) * rnd[i]
            # if(r<0): 
            #     skipFlag=True             
            #     break
            rList.append(r)
            if(i==nstep-1): y+=r*(dt/2)
            else: y+=r*dt
#
        if(not skipFlag):
            P, _ = mcHoLeeZCBPrice(marketBondPrices,
                                             sigma,
                                             r,
                                             par,
                                             uBondTimeMaturity-timeMaturity,
                                             nstep,
                                             m, nsample)
            if not (lowerBarrier < P < upperBarrier):
                # print(f"Simulated bond price {P} breached barrier at maturity, skipping payoff calculation.")
                dcfsample.append(0)
                break
            knockoutFlag=False
            for i in range(nstep-1, -1, -1):
                P = P * exp(-rList[i]*dt)
                # print(f"Discounted bond price at time step {i}: {P}, with rate {rList[i]}")
                if not (lowerBarrier < P < upperBarrier):
                    # print(f"Simulated bond price {P} breached barrier at time step {i}, skipping payoff calculation.")
                    knockoutFlag=True
                    break
            if not knockoutFlag:
                dcfsample.append(exp(-y)*payoff_mc(P,strike))
            else:
                dcfsample.append(0)
#

    print(f"Simulated {len(dcfsample)} valid bond price paths out of {nsample} samples.")
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
def payoff_mc(bondPrice,strike):
    return max(bondPrice-strike,0)
#
#----------------------------------------------------------------------------------
def mcHoLeeZCBPrice(marketBondPrices, 
                    sigma, 
                    rinit, 
                    par, 
                    timeMaturity, 
                    m0,
                    m, nsample):
    snn=None
#
    dt=timeMaturity/m
#
    dcfsample=[]
    for Ls in range(nsample):
        r=rinit
        y=r*(dt/2)
        skipFlag=False
        rnd = np.random.normal(0, 1, m)
        for i in range(m):
            t = i*dt
            dFdt, sigma = get_dFdt(marketBondPrices, m0 + i, dt)
            r += (dFdt + sigma * sigma * t) * dt + sigma * sqrt(dt) * rnd[i]
            # if(r<0): 
            #     skipFlag=True             
            #     break
            if(i==m-1): y+=r*(dt/2)
            else: y+=r*dt
#
        if(not skipFlag):
            dcfsample.append(exp(-y)*par)
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)

#
    return sampleMean, stderr
#----------------------------------------------------------------------------------
def get_dFdt(marketBondPrices, i, dt):
    if i == 0:
        P0 = marketBondPrices[0][1]
        P1 = marketBondPrices[1][1]
        P2 = marketBondPrices[2][1]
        dFdt = log(P1**2/P0/P2)/dt/dt
        # sigma = marketBondPrices[1][2] / dt
    else:
        Pt = marketBondPrices[i][1] 
        Pt1 = marketBondPrices[i+1][1]
        Pt_1 = marketBondPrices[i-1][1]
        dFdt = log(Pt**2/Pt1/Pt_1)/dt/dt
        # sigma = marketBondPrices[i][2] / dt
    sigma = marketBondPrices[1][2] / dt
    return dFdt, sigma


def main():
    rzero, bondVol = loadData()
    T, K, tau, par, upperBarrier, lowerBarrier = 1, 0.95, 2, 1, 0.98, 0.90

    timeHorizon, treeType, prec = T, 'normal', 1.e-8
    n = int(timeHorizon/rzero[0][0])
    m = int(n*(tau-T)/T)
    timeInc=timeHorizon/n
    #
    marketBondPrices = gbd.genBondData(rzero,bondVol,timeInc,n+m+1)
    print('Market bond prices:')
    for i in range(len(marketBondPrices)):
        print('Time: %f, Bond Price: %f, Volatility: %f' %(marketBondPrices[i][0], marketBondPrices[i][1], marketBondPrices[i][2]))

    optionMaturity, strike, uBondMaturity,uBondPar= T, K, tau, par
    paymentSchedule, uBondCoupon=[], []
    treeType, prec='lognormal', 1.e-8
    #
    fTree = bondOptionTree(optionMaturity,
                         strike,
                         uBondMaturity,
                         uBondPar,
                         upperBarrier,
                         lowerBarrier,
                         uBondCoupon,
                         paymentSchedule,
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
    nstep, nsample= n, 100000
    dt = timeHorizon/n
    # m = int(nstep*(uBondMaturity-T)/T)
    print(f"n: {n}, dt: {dt}, m: {m}")
    marketBondPrices = gbd.genBondData(rzero,bondVol,dt,n + m + 1)
    #
    optionprice, error = mcHoLeeZCBCall(marketBondPrices,
                                        strike,
                                        uBondPar,
                                        upperBarrier,
                                        lowerBarrier,
                                        optionMaturity,
                                        uBondMaturity,
                                        nstep,
                                        nsample)
    print('MC pricing: ')
    print(optionprice, error)
    #

    return 

if __name__ == '__main__':
    main()