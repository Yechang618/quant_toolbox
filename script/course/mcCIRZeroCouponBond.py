import rand
import statistics
from math import exp,sqrt
import numpy
#
#----------------------------------------------------------------------------------
def mcCIRZeroCouponBondPrice(a,b,sigma,rinit,par,timeMaturity,nstep,nsample):
    snn=None
#
    dt=timeMaturity/nstep
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
            dcfsample.append(exp(-y)*par)
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
def cmcCIRZeroCouponBondPrice(a,b,sigma,rinit,par,timeMaturity,nstep,nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    g,h=[],[]
    dcfsample=[]
    for Ls in range(nsample):
        r,rc=rinit,rinit
        y=r*(dt/2)
        yc=rc*(dt/2)
        skipFlag=False
        for i in range(nstep):
            snn=rand.stdnormnum(snn)
            r+=a*(b-r)*dt+sigma*sqrt(r)*sqrt(dt)*snn[0]
            rc+=a*(b-rc)*dt+sigma*sqrt(rinit)*sqrt(dt)*snn[0]
            if(r<0 or rc<0): 
                skipFlag=True             
                break
            if(i==nstep-1): 
                y+=r*(dt/2)
                yc+=rc*(dt/2)
            else: 
                y+=r*dt
                yc+=rc*dt
#
        if(not skipFlag):
            g.append(exp(-y)*par)
            h.append(exp(-yc)*par)
#
    alpha=numpy.cov(g,h)[0][1]/numpy.cov(h,h)[0][1]
    for Ls in range(len(g)):
        dcfsample.append(g[Ls]-alpha*h[Ls])
#
    sampleMean=statistics.mean(dcfsample)+alpha*vasicekZeroCouponBondPrice(a,b,sigma*sqrt(rinit),r0,par,timeMaturity)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
def vasicekZeroCouponBondPrice(a,b,sigma,rinit,par,timeMaturity):
    Bfunc=(1/a)*(1-exp(-a*timeMaturity))
    Afunc=exp((1/a**2)*(Bfunc-timeMaturity)*(a**2*b-(1/2)*sigma**2)-sigma**2*Bfunc**2/(4*a))
    bondPrice=par*Afunc*exp(-Bfunc*rinit)
    return bondPrice
#
#----------------------------------------------------------------------------------
def amcCIRZeroCouponBondPrice(a,b,sigma,rinit,par,timeMaturity,nstep,nsample):
    snn=None
#
    dt=timeMaturity/nstep
#
    dcfsample=[]
    for Ls in range(nsample):
        r,ra=rinit,rinit
        y=r*(dt/2)
        ya=ra*(dt/2)
        skipFlag=False
        for i in range(nstep):
            snn=rand.stdnormnum(snn)
            r+=a*(b-r)*dt+sigma*sqrt(r)*sqrt(dt)*snn[0]
            ra+=a*(b-ra)*dt+sigma*sqrt(ra)*sqrt(dt)*(-snn[0])
            if(r<0 or ra<0): 
                skipFlag=True             
                break
#
            if(i==nstep-1): 
                y+=r*(dt/2)
                ya+=ra*(dt/2)
            else: 
                y+=r*dt
                ya+=ra*dt
#
        if(not skipFlag):
            dcfsample.append((1/2)*(exp(-y)*par+exp(-ya)*par))
#
    sampleMean=statistics.mean(dcfsample)
    stderr=(statistics.stdev(dcfsample))/sqrt(nsample)
#
    return sampleMean,stderr
#
#----------------------------------------------------------------------------------
rand.seedinit(5678)
#
a,b,r0,sigma,timeMaturity,par,nstep,nsample=0.1,0.1,0.03,0.015,1.0,100,100,10000
#
optionprice,stderr=mcCIRZeroCouponBondPrice(a,b,sigma,r0,par,timeMaturity,nstep,nsample)
print(optionprice,stderr)
#
optionprice,stderr=cmcCIRZeroCouponBondPrice(a,b,sigma*sqrt(r0),r0,par,timeMaturity,nstep,nsample)
print(optionprice,stderr)
#
optionprice,stderr=amcCIRZeroCouponBondPrice(a,b,sigma,r0,par,timeMaturity,nstep,int(nsample/2))
print(optionprice,stderr)