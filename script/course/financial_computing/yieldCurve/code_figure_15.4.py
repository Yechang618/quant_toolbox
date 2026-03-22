#
import cubicSpline
import matplotlib.pyplot as plt
import yieldCurve
#---------------------------------------------------------------------------------
n=4
bondMaturity=[0.5,1,2,3]
bondPrice=[98.94,97.65,114.26,120.75]
par=[100,100,100,100]
coupon=[0,0,5,5]
m=[0,0,4,6]
tc=[[],[],[0.5,1,1.5,2],[0.5,1,1.5,2,2.5,3]]
prec=1e-14 
#
rzero=yieldCurve.calZeroRates(n,bondMaturity,bondPrice,par,coupon,m,tc,prec)
plt.plot(bondMaturity,rzero,'r.')	
#
marketZeroRates=[]
for i in range(n):marketZeroRates.append([bondMaturity[i],rzero[i]])
splineCoeff=cubicSpline.cubicSpline(n,marketZeroRates)
#
m=100
x,y=[],[]
for i in range(0,m):
    xint=bondMaturity[0]+i*(bondMaturity[n-1]-bondMaturity[0])/(m-1)
    y.append(cubicSpline.interpolation(n,marketZeroRates,splineCoeff,xint))    
    x.append(xint)
plt.plot(x,y)	
plt.show()
