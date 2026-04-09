#
import genBondData
import irTree
from math import exp
#-----------------------------------------------------------------
def bondOptionTree(optionMaturity,strike,uBondMaturity,uBondPar,uBondCoupon,paymentSchedule,\
    rzero,bondVol,treeType,prec):
#
# Generate bond price and volatility term structures with term=[0,dt,2dt,...,nTerm*dt] where nTerm*dt=underlying bond maturity 
# 
    nCoupon=len(uBondCoupon)
    timeHorizon=uBondMaturity
    n=int(timeHorizon/rzero[0][0])
    timeInc=timeHorizon/n
#
    marketBondPrices=genBondData.genBondData(rzero,bondVol,timeInc,n+1)
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
# Rolling back option price starting at i=Hf
#
    for j in range(0,H+1):
        fTree[H].append(payoff(Bf[H][j],strike))
#
    for i in range(H-1,-1,-1):
        for j in range(0,i+1):
            fairValue=exp(-treeRate[i][j]*timeInc)*(1/2)*(fTree[i+1][j]+fTree[i+1][j+1])
            fTree[i].append(max(fairValue,payoff(Bf[i][j],strike)))
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
def payoff(assetPrice,strike):
    return max(assetPrice-strike,0)
#
#---------------------------------------------------------------------------------

