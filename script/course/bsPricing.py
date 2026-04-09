#
from math import erf,sqrt,log,exp
#-------------------------------------------------------------------------------------------
def normsdist(x):
    normsdist=(1+erf(x/sqrt(2)))/2
    return normsdist
#
def blackScholeEuropeanCall(assetPrice,strike,riskFree,sigma,expiration):
    d1=(log(assetPrice/strike)+(riskFree+(1/2)*sigma**2)*expiration)/(sigma*sqrt(expiration))
    d2=d1-sigma*sqrt(expiration)
    callPrice=assetPrice*normsdist(d1)-strike*exp(-riskFree*expiration)*normsdist(d2)
    return callPrice
#
def blackScholeEuropeanPut(assetPrice,strike,riskFree,sigma,expiration):
    d1=(log(assetPrice/strike)+(riskFree+(1/2)*sigma**2)*expiration)/(sigma*sqrt(expiration))
    d2=d1-sigma*sqrt(expiration)
    putPrice=strike*exp(-riskFree*expiration)*normsdist(-d2)-assetPrice*normsdist(-d1)
    return putPrice
#
#-------------------------------------------------------------------------------------------
def blackScholeAsianCallGa(assetPrice,strike,riskFree,sigma,t,nstep):
    sumt,wsumt=0.0,0.0
    for i in range(1,nstep+1):
        sumt=sumt+t[i]
        wsumt=wsumt+(2*(nstep-i)+1)*t[i]
    E=(riskFree-(1/2)*sigma**2)*(1/nstep)*sumt
    S=sigma*(1/nstep)*sqrt(wsumt)
    X=assetPrice*exp(E+(1/2)*S**2)
    d1=(log(X/strike)+(1/2)*S**2)/S
    d2=d1-S
    callPrice=exp(-riskFree*t[nstep])*X*normsdist(d1)-exp(-riskFree*t[nstep])*strike*normsdist(d2)
    return callPrice
#
#-------------------------------------------------------------------------------------------	