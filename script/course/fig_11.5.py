import linearLeastSquaresFit
from math import sqrt,exp,log,fabs
import rand,statistics
#-------------------------------------------------------------------------------------------
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
def criticalPrice(assetPrice,strike,riskFree,sigma,expiration,nstep,nsample,h,eta):
    Bc=[0 for i in range(nstep+1)]
    Bc[nstep]=strike
    for i in range(nstep-1,-1,-1):
        xceiling=Bc[i+1]
        flagFit=True
        while(flagFit):
            xFit,yFit,sd=[],[],[]	
            xlowest=xceiling
            for j in range(h+1):
                x=(1-eta*j/h)*xceiling				
                narbPrice,stdError=narbAmericanPut(x,strike,riskFree,sigma,expiration,Bc,i,nstep,nsample)
                xFit.append(x)
                yFit.append(narbPrice-payoff(x,strike))
                sd.append(stdError)
                if(narbPrice-payoff(x,strike)>0): xlowest=x					
            slope,intercept=linearLeastSquaresFit.straightLineFitSd(xFit,yFit,sd,h)
            if(slope<0):
                xroot=xlowest
            else:
                xroot=-intercept/slope
            narbPrice,stdError=narbAmericanPut(xroot,strike,riskFree,sigma,expiration,Bc,i,nstep,nsample)
            flagFit=fabs(narbPrice-payoff(xroot,strike))>stdError
            xceiling=xroot
        Bc[i]=xroot
    return Bc
#-------------------------------------------------------------------------------------------
rand.seedinit(5678)
#
assetPrice,strike,riskFree,sigma,expiration,nsample,h=100,100,0.05,0.2,1.0,1000000,30
#
x,y=[],[]
for nstep in [100]:
    eta=0.1*(1/nstep)
    Bc=criticalPrice(assetPrice,strike,riskFree,sigma,expiration,nstep,nsample,h,eta)
    dt=expiration/nstep	
    if(assetPrice<=Bc[0]):
        optionPrice=strike-assetPrice
        stdError=0
    else:
        optionPrice,stdError=narbAmericanPut(assetPrice,strike,riskFree,sigma,expiration,Bc,0,nstep,nsample)
    print(optionPrice,stdError)
#