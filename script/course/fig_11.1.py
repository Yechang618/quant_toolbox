from math import sqrt,log,exp
import rand,numpy,statistics,bsPricing
#-------------------------------------------------------------------------------------------
def mcAsianCall(assetPrice,strike,riskFree,sigma,t,nstep,nsample):
    snn=None   
    dcfSample=[]
    for Ls in range(0,nsample):
        Qt=assetPrice
        AT=0.0
        for i in range(0,nstep):
            snn=rand.stdnormnum(snn)
            Qt=Qt*exp((riskFree-(1/2)*sigma**2)*(t[i+1]-t[i])+sigma*sqrt(t[i+1]-t[i])*snn[0])
            AT=AT+Qt
        AT=AT/nstep
        dcf=exp(-riskFree*t[nstep])*payoff(AT,strike)
        dcfSample.append(dcf)
    sampleMean=statistics.mean(dcfSample)
    stdError=(statistics.stdev(dcfSample))/sqrt(nsample)
    return sampleMean,stdError
#-------------------------------------------------------------------------------------------
def cmcAsianCall(assetPrice,strike,riskFree,sigma,t,nstep,nsample):
    snn=None
    g,h,dcfsample=[],[],[]
    for Ls in range(0,nsample):
        Qt=assetPrice
        AT,GT=0.0,0.0
        for i in range(0,nstep):
            snn=rand.stdnormnum(snn)
            Qt=Qt*exp((riskFree-(1/2)*sigma**2)*(t[i+1]-t[i])+sigma*sqrt(t[i+1]-t[i])*snn[0])
            AT=AT+Qt
            GT=GT+log(Qt)
        AT=AT/nstep
        GT=exp(GT/nstep)
        g.append(exp(-riskFree*t[nstep])*payoff(AT,strike))
        h.append(exp(-riskFree*t[nstep])*payoff(GT,strike))
    alpha=numpy.cov(g,h)[0][1]/numpy.cov(h,h)[0][1]
    for Ls in range(0,nsample):
        dcfsample.append(g[Ls]-alpha*h[Ls])
    meanh=bsPricing.blackScholeAsianCallGa(assetPrice,strike,riskFree,sigma,t,nstep)
    sampleMean=statistics.mean(dcfsample)+alpha*meanh
    stdError=(statistics.stdev(dcfsample))/sqrt(nsample)
    return sampleMean,stdError
#-------------------------------------------------------------------------------------------
def payoff(assetPrice,strike):
    return max(assetPrice-strike,0)
#-------------------------------------------------------------------------------------------
rand.seedinit(5678)
assetPrice,strike,riskFree,sigma,nsample=100.0,100.0,0.05,0.2,100000
nstep,t=4,[0,0.25,0.5,0.75,1.0]
optionPrice,error=mcAsianCall(assetPrice,strike,riskFree,sigma,t,nstep,nsample)
print('crude     :',optionPrice,error)	
optionPrice,error=cmcAsianCall(assetPrice,strike,riskFree,sigma,t,nstep,nsample)
print('control   :',optionPrice,error)	