# 
#------------------------------------------------------------------------------------
def moneyness(assetPrice,riskFreeRate,strike,yte):
    from math import log,exp,atan
    return atan(log(strike/(assetPrice*exp(riskFreeRate*yte))))
#
def volSkewCoeff(assetPrice,riskFreeRate,yteData,strikeData,ivData,ytebreak):
    from linearLeastSquaresFit import bfuncFit
#
    skewCoeff=[]
    yteList=[]
    nCurve=len(ytebreak)-1	
    for i in range(nCurve):
        yte=yteData[ytebreak[i]]			
        strikes=strikeData[ytebreak[i]:ytebreak[i+1]-1]
        ivs=ivData[ytebreak[i]:ytebreak[i+1]-1]
#
        phi=[]
        nStrike=len(strikes)-1		
        for j in range(nStrike+1):
            x=moneyness(assetPrice,riskFreeRate,strikes[j],yte)
            phi.append([0,1.0,x,x**2])           
        skewCoeff.append(bfuncFit(ivs,phi,nStrike,3))
        yteList.append(yte)		
#
    return yteList,skewCoeff
#
def fittedVolSkewCurve(assetPrice,riskFreeRate,yteList,skewCoeff,strike,i):
    x=moneyness(assetPrice,riskFreeRate,strike,yteList[i])
    return skewCoeff[i][1]+skewCoeff[i][2]*x+skewCoeff[i][3]*x**2
#
#----------------------------------------------------------------------------------------