#
import irTree
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
rzero=[[1/12,0.04286],[2/12,0.04313],[3/12,0.04338],[4/12,0.04339],[6/12,0.04315],\
[1,0.04109],[2,0.03898],[3,0.03858],[5,0.03959],[7,0.04170],[10,0.04397],[20,0.04928],[30,0.04923]]
#
bondVol=[[1/12,0.000168],[2/12,0.000169],[3/12,0.000912],[4/12,0.000853],\
[6/12,0.00125],[1,0.00486],[2,0.0186],[3,0.0312],[5,0.0531],[7,0.0743],[10,0.100],[20,0.184],[30,0.263]]
#
n,timeHorizon,treeType,prec=5,2.0,'lognormal',1.e-8
timeInc=timeHorizon/n
#
marketBondPrices=genBondData(rzero,bondVol,timeInc,n+1)
#
treeRate=irTree.irTree(timeHorizon,n,treeType,marketBondPrices,prec)
#
for j in range(0,n+1):
    for i in range(0,n+1):
        if(i<j):
            print('%6s '%(''),end=' ')
        else:
            print('%6.4f '%(treeRate[i][j]),end=' ')
    print('') 
#