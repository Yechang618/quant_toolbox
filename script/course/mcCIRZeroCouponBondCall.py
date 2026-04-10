import rand
import statistics
from math import exp,sqrt
#
#----------------------------------------------------------------------------------
def mcCIRZeroCouponBondCall(a,b,sigma,r0,strike,par,timeMaturity,uBondTimeMaturity,nstep,nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    m=int(nstep*(uBondTimeMaturity-timeMaturity)/timeMaturity)
#
    dcfsample=[]
    for Ls in range(nsample):
        r=rinit
        y=r*(dt/2)
        skipFlag=False
        for i in range(nstep):
            snn=rand.stdnormnum(snn)
            r+=a*(b-r)*dt+sigma*sqrt(r)*sqrt(dt)*snn[0]
            if(r<0): 
                skipFlag=True             
                break
            if(i==nstep-1): y+=r*(dt/2)
            else: y+=r*dt
#
        if(not skipFlag):
            uBondPrice,error=mcCIRZeroCouponBondPrice(a,b,sigma,r,par,uBondTimeMaturity-timeMaturity,m,nsample)
            dcfsample.append(exp(-y)*payoff(uBondPrice,strike))
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
def payoff(bondPrice,strike):
    return max(bondPrice-strike,0)
#
#----------------------------------------------------------------------------------
def mcCIRZeroCouponBondPrice(a,b,sigma,r0,par,timeMaturity,nstep,nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    dcfsample=[]
    for Ls in range(nsample):
        r=r0
        y=r*(dt/2)
        skipFlag=False
        for i in range(nstep):
            snn=rand.stdnormnum(snn)
            r+=a*(b-r)*dt+sigma*sqrt(r)*sqrt(dt)*snn[0]
            if(r<0): 
                skipFlag=True             
                break
            if(i==nstep-1): y+=r*(dt/2)
            else: y+=r*dt
#
        if(not skipFlag):
            dcfsample.append(exp(-y)*par)
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
rand.seedinit(5678)
#
a,b,rinit,sigma,timeMaturity,uBondTimeMaturity,par,strike,nstep,nsample=0.1,0.1,0.03,0.015,0.5,1.0,100,70,100,100
#
optionprice,error=mcCIRZeroCouponBondCall(a,b,sigma,rinit,strike,par,timeMaturity,uBondTimeMaturity,nstep,nsample)
print(optionprice,error)
#