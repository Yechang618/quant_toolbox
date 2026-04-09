import mop,midPoint
from basisfuncs import legendre7
import linearLeastSquaresFit
#-------------------------------------------------------------------------------------------
from math import sqrt,exp
import rand,statistics
rand.seedinit(5678)
#
def narbAmericanPut(assetPriceStart,strike,riskFree,sigma,expiration,Bc,iptr,nstep,nsample):
    snn=None
    dcfSample=[]
    t=[i*expiration/nstep for i in range(0,nstep+1)]	
    for Ls in range(0,nsample):
        Qt=assetPriceStart
        for Lt in range(iptr,nstep):
            snn=rand.stdnormnum(snn)
            Qt=Qt*exp((riskFree-(1/2)*sigma**2)*(t[Lt+1]-t[Lt])+sigma*sqrt(t[Lt+1]-t[Lt])*snn[0])
            stopTime=t[Lt+1]
            exerciseFlag=exerciseBoundary(Qt,Bc[Lt+1])            
            if(exerciseFlag):break
        dcf=exp(-riskFree*(stopTime-t[iptr]))*payoff(Qt,strike)
        dcfSample.append(dcf)
    sampleMean=statistics.mean(dcfSample)
    if(nsample==1):stdError=0.0
    else:stdError=(statistics.stdev(dcfSample))/sqrt(nsample)
    return sampleMean,stdError
#-------------------------------------------------------------------------------------------
def payoff(assetPrice,strike):
    return max(strike-assetPrice,0)
#-------------------------------------------------------------------------------------------
def exerciseBoundary(assetPrice,criticalPrice):
    if(assetPrice<=criticalPrice):exerciseFlag=True
    else:exerciseFlag=False
    return exerciseFlag
#-------------------------------------------------------------------------------------------		
#-------------------------------------------------------------------------------------------
def criticalPriceBasisFunc(assetPrice,strike,riskFree,sigma,expiration,nstep,h,nsample,m):
    Bc=[0 for i in range(nstep+1)]
    Bc[nstep]=strike	
#backward iterate critical prices Bc[0 to nstep]
    for i in range(nstep-1,-1,-1):
        yFit,bfuncs=[],[] 
        xmax,xmin=strike,0.8*Bc[i+1]
        for j in range(h+1):
            x=xmax-(xmax-xmin)*(j/h)
            bfuncs.append(legendre7(x))  # Evaluate the basis functions at x defined as bfuncs[j][1 to 7]
            sampleMean,stdError=narbAmericanPut(x,strike,riskFree,sigma,expiration,Bc,i,nstep,nsample)
            yFit.append(sampleMean-payoff(x,strike))	
        cvec=linearLeastSquaresFit.bfuncFit(yFit,bfuncs,h,m)
        pa=strike,cvec,m 
        Bc[i]=midPoint.midPoint(y,xmax,xmin,pa)	
#
    return Bc
#-------------------------------------------------------------------------------------------
def y(x,pa):
    strike,cvec,m=pa
    y=0	
    for ir in range(1,m+1):y+=cvec[ir]*legendre7(x)[ir]
    return y
#-------------------------------------------------------------------------------------------
assetPrice,strike,riskFree,sigma,expiration,nstep,h,nsample,m=100,100,0.05,0.2,1.0,100,1000000,1,7
#
Bc=criticalPriceBasisFunc(assetPrice,strike,riskFree,sigma,expiration,nstep,h,nsample,m)
#
if(assetPrice>Bc[0]):
    putPrice,stdError=narbAmericanPut(assetPrice,strike,riskFree,sigma,expiration,Bc,0,nstep,h)
else:
    putPrice,stdError=payoff(assetPrice,strike),0
#
print(nstep,putPrice,stdError)