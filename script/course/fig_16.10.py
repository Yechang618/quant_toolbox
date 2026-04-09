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
# Rolling back option price starting as i=Hf
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
rzero=[[1/12,0.04286],[2/12,0.04313],[3/12,0.04338],[4/12,0.04339],[6/12,0.04315],\
[1,0.04109],[2,0.03898],[3,0.03858],[5,0.03959],[7,0.04170],[10,0.04397],[20,0.04928],[30,0.04923]]
#
bondVol=[[1/12,0.000168],[2/12,0.000169],[3/12,0.000912],[4/12,0.000853],\
[6/12,0.00125],[1,0.00486],[2,0.0186],[3,0.0312],[5,0.0531],[7,0.0743],[10,0.100],[20,0.184],[30,0.263]]
#
optionMaturity,strike,uBondMaturity,uBondPar=3.0,95,4.0,100
paymentSchedule,uBondCoupon=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
treeType,prec='lognormal',1.e-8
#
fTree=bondOptionTree(optionMaturity,strike,uBondMaturity,uBondPar,uBondCoupon,paymentSchedule,\
rzero,bondVol,treeType,prec)
#
print(fTree[0][0])


