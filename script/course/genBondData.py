#
import cubicSpline
from math import exp
#----------------------------------------------------------------------------------
def genBondData(rzero,bondVol,timeInc,m):
#
    splineCoeff=cubicSpline.cubicSpline(len(rzero),rzero)
    splineCoeffv=cubicSpline.cubicSpline(len(bondVol),bondVol)
#
    marketBondPrices=[[0,1,0]]
    for i in range(1,m+1):
        timeMaturity=i*timeInc
        rate=cubicSpline.interpolation(len(rzero),rzero,splineCoeff,timeMaturity)    
        zeroBondPrice=exp(-rate*timeMaturity)    
        vol=cubicSpline.interpolation(len(bondVol),bondVol,splineCoeffv,timeMaturity) 	
        marketBondPrices.append([timeMaturity,zeroBondPrice,vol])
    return marketBondPrices
#
#----------------------------------------------------------------------------------