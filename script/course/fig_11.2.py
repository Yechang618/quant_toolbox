from math import sqrt,log,exp
import rand,numpy,statistics
#-------------------------------------------------------------------------------------------
def mcDKOCall(assetPrice,strike,upBarrier,lowBarrier,rebate,riskFree,sigma,expiration,nstep,nsample):
    snn=None   
    dcfSample=[]	
    if(DKOBoundary(assetPrice,upBarrier,lowBarrier)):return 0.0,0.0
    t=[i*expiration/nstep for i in range(0,nstep+1)]
    for Ls in range(0,nsample):
        Qt=assetPrice
        for i in range(0,nstep):
            snn=rand.stdnormnum(snn)
            Qt=Qt*exp((riskFree-(1/2)*sigma**2)*(t[i+1]-t[i])+sigma*sqrt(t[i+1]-t[i])*snn[0])
            tagTime=t[i+1]
            crossFlag=DKOBoundary(Qt,upBarrier,lowBarrier)
            if(crossFlag):break
        if(crossFlag):
            dcf=exp(-riskFree*tagTime)*rebate
        else:
            dcf=exp(-riskFree*tagTime)*payoff(Qt,strike)		
        dcfSample.append(dcf)
    sampleMean=statistics.mean(dcfSample)
    stdError=(statistics.stdev(dcfSample))/sqrt(nsample)
    return sampleMean,stdError
#-------------------------------------------------------------------------------------------
def DKOBoundary(assetPrice,upBarrier,lowBarrier):
    if(assetPrice>=upBarrier or assetPrice<=lowBarrier):crossFlag=True
    else:crossFlag=False
    return crossFlag
#-------------------------------------------------------------------------------------------
def payoff(assetPrice,strike):
    return max(assetPrice-strike,0)
#-------------------------------------------------------------------------------------------
rand.seedinit(5678)
assetPrice,strike,upBarrier,lowBarrier,rebate,riskFree,sigma,expiration,nsample=100.0,100.0,120.0,80.0,0.05,0.05,0.2,1.0,100000
for nstep in [10,100,1000,10000]:
    optionPrice,error=mcDKOCall(assetPrice,strike,upBarrier,lowBarrier,rebate,riskFree,sigma,expiration,nstep,nsample)
    print(nstep,optionPrice,error)		