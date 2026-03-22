#
import cubicSpline
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------
#
marketZeroRates=[[1/12,0.04286],[2/12,0.04313],[3/12,0.04338],[4/12,0.04339],[6/12,0.04315],\
[1,0.04109],[2,0.03898],[3,0.03858],[5,0.03959],[7,0.04170],[10,0.04397],[20,0.04928],[30,0.04923]]
#
n=len(marketZeroRates)
maturities,zeroRates=[],[]
for i in range(n):    
    maturities.append(marketZeroRates[i][0])
    zeroRates.append(marketZeroRates[i][1])
plt.plot(maturities,zeroRates,'r.')	
#
splineCoeff=cubicSpline.cubicSpline(n,marketZeroRates)
#
m=100
x,y=[],[]
for i in range(0,m):
    xint=maturities[0]+i*(maturities[n-1]-maturities[0])/(m-1)
    y.append(cubicSpline.interpolation(n,marketZeroRates,splineCoeff,xint))    
    x.append(xint)
plt.plot(x,y)	
plt.show()
#



