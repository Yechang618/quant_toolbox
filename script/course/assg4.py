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

# #----------------------------------------------------------------------------------
# def genBondData(rzero,bondVol,timeInc,m):
# #
#     splineCoeff=cubicSpline.cubicSpline(len(rzero),rzero)
#     splineCoeffv=cubicSpline.cubicSpline(len(bondVol),bondVol)
# #
#     marketBondPrices=[[0,1,0]]
#     for i in range(1,m+1):
#         timeMaturity=i*timeInc
#         rate=cubicSpline.interpolation(len(rzero),rzero,splineCoeff,timeMaturity)    
#         zeroBondPrice=exp(-rate*timeMaturity)    
#         vol=cubicSpline.interpolation(len(bondVol),bondVol,splineCoeffv,timeMaturity) 	
#         marketBondPrices.append([timeMaturity,zeroBondPrice,vol])
#     return marketBondPrices
#-----------------------------------------------------------------
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
    n=int(timeHorizon/rzero[0][0])
    timeInc=timeHorizon/n
#
    marketBondPrices = gbd.genBondData(rzero,bondVol,timeInc,n+1)
    treeRate=irTree.irTree(timeHorizon,n,treeType,marketBondPrices,prec)
#
    H=int(optionMaturity/timeInc)
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
def mcHoLeeZeroCouponBondCall(a,
                              b,
                              marketBondPrices,
                              r0,
                              strike,
                              par,
                              timeMaturity,
                              uBondTimeMaturity,
                              nstep,
                              nsample):
    snn=None
#
    dt = timeMaturity / nstep
#
    m=int(nstep*(uBondTimeMaturity-timeMaturity)/timeMaturity)
#
    dcfsample=[]
    for Ls in range(nsample):
        r = r0
        y = r * (dt / 2)
        skipFlag = False
        for i in range(nstep):
            snn = rand.stdnormnum(snn)
            r += a * (b - r) * dt + sigma * sqrt(r) * sqrt(dt) * snn[0]
            if (r < 0): 
                skipFlag = True             
                break
            if (i == nstep-1): y += r * (dt / 2)
            else: y += r * dt
#
        if(not skipFlag):
            uBondPrice, error = mcHoLeeZeroCouponBondPrice(a,
                                                         b,
                                                         marketBondPrices,
                                                         r,
                                                         par,
                                                         uBondTimeMaturity - timeMaturity,
                                                         m,
                                                         nsample)
            dcfsample.append(exp(-y) * payoff(uBondPrice, strike))
#
    sampleMean = statistics.mean(dcfsample)
    stderr = (statistics.stdev(dcfsample)) / sqrt(nsample)
#
    return sampleMean, stderr
#
#----------------------------------------------------------------------------------
def payoff(bondPrice,strike):
    return max(bondPrice-strike,0)
#
#----------------------------------------------------------------------------------
def mcHoLeeZeroCouponBondPrice(a,
                               b,
                               marketBondPrices,
                               r0,
                               par,
                               timeMaturity,
                               nstep,
                               nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    # timeMaturity,zeroBondPrice,vol = gbd.genBondData
    dcfsample=[]
    for Ls in range(nsample):
        r=r0
        y=r*(dt/2)
        skipFlag=False
        for i in range(nstep):
            snn=rand.stdnormnum(snn)
            #
            timeMaturity,zeroBondPrice,vol = marketBondPrices[i]
            #
            sigma = vol/dt
            if i == 0:
                P0 = marketBondPrices[0][1]
                P2 = 
                dFdt = log()
            r+=a*(b-r)*dt+sigma*sqrt(r)*sqrt(dt)*snn[0]
            if(r<0): 
                skipFlag=True             
                break
            if(i==nstep-1): y+=r*(dt/2)
            else: y+=r*dt
#
        if(not skipFlag):
            dcfsample.append(exp(-y)*par)
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr


def main():
    rzero, bondVol = loadData()
    T, K, tau, par, H, L = 1, 0.95, 2, 1, 0.98, 0.90

    n, timeHorizon, treeType, prec = 5, T, 'lognormal', 1.e-8
    timeInc=timeHorizon/n
    #
    marketBondPrices = gbd.genBondData(rzero,bondVol,timeInc,n+1)

    treeRate=irTree.irTree(timeHorizon,n,treeType,marketBondPrices,prec)
    #
    print('Tree rate:')
    for j in range(0,n+1):
        for i in range(0,n+1):
            if(i<j):
                print('%6s '%(''),end=' ')
            else:
                print('%6.4f '%(treeRate[i][j]),end=' ')
        print('') 

    optionMaturity, strike, uBondMaturity,uBondPar= T, K, tau, par
    # paymentSchedule,uBondCoupon=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
    paymentSchedule, uBondCoupon=[], []
    treeType, prec='lognormal', 1.e-8
    #
    fTree = bondOptionTree(optionMaturity,
                         strike,
                         uBondMaturity,
                         uBondPar,
                         H,
                         L,
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
    rand.seedinit(5678)
    #
    a,b,rinit,sigma,timeMaturity,uBondTimeMaturity,par,strike,nstep,nsample=0.1,0.1,0.03,0.015,0.5,1.0,100,70,100,100
    #
    optionprice,error = mcHoLeeZeroCouponBondCall(a,
                                                b,
                                                marketBondPrices,
                                                rinit,
                                                strike,
                                                par,
                                                timeMaturity,
                                                uBondTimeMaturity,
                                                nstep,
                                                nsample)
    print(optionprice, error)
    #

    return 

if __name__ == '__main__':
    main()