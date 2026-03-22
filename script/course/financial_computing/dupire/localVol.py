#
import volSkew
import volTerm
#----------------------------------------------------------------------------------
def localVol(assetPrice,riskFreeRate,skewCoeffParameters,forwardPrice,forwardTime): 
    from math import log,tan,sqrt
#
    strike,yte=forwardPrice,forwardTime
#
    h1=volTerm.volSkewCoeffTerm(skewCoeffParameters,yte,1)
    h2=volTerm.volSkewCoeffTerm(skewCoeffParameters,yte,2)    
    h3=volTerm.volSkewCoeffTerm(skewCoeffParameters,yte,3)
#
    tdh1=volTerm.tdVolSkewCoeffTerm(skewCoeffParameters,yte,1)
    tdh2=volTerm.tdVolSkewCoeffTerm(skewCoeffParameters,yte,2)
    tdh3=volTerm.tdVolSkewCoeffTerm(skewCoeffParameters,yte,3)
#
    X=volSkew.moneyness(assetPrice,riskFreeRate,strike,yte)
    y=tan(X)
#
    term1=(h2+2*h3*X)/(1+y**2)
    term2=2*h3/(1+y**2)**2-(h2+2*h3*X)*(1+y)**2/(1+y**2)
    term3=tdh1+tdh2*X+tdh3*X**2	
#
    iv=volTerm.fittedVolTermCurve(assetPrice,riskFreeRate,skewCoeffParameters,yte,strike)	
    b=(log(assetPrice/strike)+(riskFreeRate+(1/2)*iv**2)*yte)/iv
    localVolSq=(iv**2+2*iv*term3)/((1+b*term1)**2+iv*yte*(term2-b*term1**2))
#   
# precaution: for extrapolation far away from at-the-money strike price
# 
    if(localVolSq<0):localVolSq=0
    return sqrt(localVolSq)
#
#----------------------------------------------------------------------------------------
