from math import erf, sqrt,log,exp
import newtonRaphson
#
def normsdist(x):
    normsdist=(1+erf(x/sqrt(2)))/2
    return normsdist
#
def matchBSEuropeanCallPrice(sigma,pa):
    assetPrice,strike,riskFreeRate,timeExpiration,marketCallPrice=pa
    d1=(log(assetPrice/strike)+(riskFreeRate+(1/2)*sigma**2)*timeExpiration)/(sigma*sqrt(timeExpiration))
    d2=d1-sigma*sqrt(timeExpiration)
    BSEuropeanCallPrice=assetPrice*normsdist(d1)-strike*exp(-riskFreeRate*timeExpiration)*normsdist(d2)
    return BSEuropeanCallPrice-marketCallPrice
#
#------------------------------------------------------
#
assetPrice,strike,riskFreeRate,timeExpiration,marketCallPrice=100.0,95.0,0.05,1.0,13.0
pa=assetPrice,strike,riskFreeRate,timeExpiration,marketCallPrice
ivtrial=0.17
prec=1.e-8
iv,precMet,dev=newtonRaphson.newtonRaphson_scalar(matchBSEuropeanCallPrice,ivtrial,prec,pa)
print(iv,precMet,dev)