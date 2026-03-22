#
from math import erf,sqrt,log,exp,fabs
#----------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------
def impVol(assetPrice,riskFreeRate,hisVol,marketCallData,numberOfCall,prec,tol):
    import newtonRaphson
#
    yteData,strikeData,ivData=[],[],[]
    for i in range(numberOfCall):
        yte,strike,marketCallPrice=marketCallData[i]		
        pa=assetPrice,strike,riskFreeRate,yte,marketCallPrice
        ivtrial=hisVol
        if(marketCallPrice>=assetPrice-strike*exp(-riskFreeRate*yte)):
            iv,precMet,dev=newtonRaphson.newtonRaphson_scalar(matchBSEuropeanCallPrice,ivtrial,prec,pa) 
            if(precMet==True and fabs(dev)<=tol):	
                yteData.append(yte)
                strikeData.append(strike)
                ivData.append(iv)			
#
# data are already sorted in order of increasing yte.
# breaking point for changing value of yte, it includes also end of data.
    ytebreak=[0]
    for i in range(len(yteData)-1):
        if(yteData[i]!=yteData[i+1]):ytebreak.append(i+1) 
    ytebreak.append(len(yteData))
#
    return yteData,strikeData,ivData,ytebreak
# 
#------------------------------------------------------------------------------------