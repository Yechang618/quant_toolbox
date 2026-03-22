#
import volSkew 
#----------------------------------------------------------------------------------------
def fitVolSkewCoeffTerm(yteList,skewCoeff):
    from scipy.optimize import minimize
    skewCoeffParameters=[0]
    L=len(yteList)-1		
    for j in [1,2,3]:
        pa=yteList,skewCoeff,j    
        xctrial=[0.0,skewCoeff[L][j],skewCoeff[0][j]-skewCoeff[L][j]]
        skewCoeffParameters.append(minimize(mse,xctrial,args=(pa)).x)		
    return skewCoeffParameters
#
def mse(xc,yteList,skewCoeff,j):
    from math import exp,sqrt
    L=len(yteList)-1	
    mse=0
    for i in range(L+1):
        yte=yteList[i]
        hfit=xc[1]+exp(-xc[0]*sqrt(yte))*xc[2]		
        mse+=(skewCoeff[i][j]-hfit)**2
    mse=mse/(L+1)		
    return mse
#
#----------------------------------------------------------------------------------------
def volSkewCoeffTerm(skewCoeffParameters,yte,j):
    from math import exp,sqrt		
    return skewCoeffParameters[j][1]+exp(-skewCoeffParameters[j][0]*sqrt(yte))*skewCoeffParameters[j][2]
#
def fittedVolTermCurve(assetPrice,riskFreeRate,skewCoeffParameters,yte,strike):
    x=volSkew.moneyness(assetPrice,riskFreeRate,strike,yte)
    h=[None]
    for j in [1,2,3]:
        h.append(volSkewCoeffTerm(skewCoeffParameters,yte,j))	
    return h[1]+h[2]*x+h[3]*x**2
#
#----------------------------------------------------------------------------------------
def tdVolSkewCoeffTerm(skewCoeffParameters,yte,j):
    from math import exp,sqrt
    return (1/2)*(-skewCoeffParameters[j][0]*sqrt(yte))*exp(-skewCoeffParameters[j][0]*sqrt(yte))*skewCoeffParameters[j][2]
#
#----------------------------------------------------------------------------------------