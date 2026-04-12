#
import newtonRaphson
from math import exp,log,sqrt
#
#-----------------------------------------------------------------
def modelRate(a,b,j,treeType):
    if(treeType=='normal'):
        rate=a-b*j
    elif(treeType=='lognormal'):
        rate=a*b**j
    return rate
#
#-------------------------------------------------------------------
def irTree(timeHorizon,n,treeType,marketBondPrices,prec):
#
    dtime=timeHorizon/n
#
    topRate=[-(1/dtime)*log(marketBondPrices[1][1])]
    downFactor=[0.0]
#
    x=[topRate[0],downFactor[0]]
    for k in range(1,n+1):
        pa=k,dtime,topRate,downFactor,marketBondPrices,treeType
        x,precMet,maxDev=newtonRaphson.newtonRaphson_vector(bondCalibrationErr,x,2,prec,pa)
        topRate.append(x[0])
        downFactor.append(x[1])
#
    treeRate=[]
    for i in range(0,n+1):
        vec=[]
        for j in range(0,i+1):	
            vec.append(modelRate(topRate[i],downFactor[i],j,treeType))
        treeRate.append(vec)
#
    return treeRate
#  
#---------------------------------------------------------------------------------
def bondCalibrationErr(x,pa):
#
    k,dtime,topRate,downFactor,marketBondPrices,treeType=pa
#
    Bf=[[] for i in range(0,(k+1)+1)]
#
# define face values of discount bond with maturity at k+1
#
    for j in range(0,(k+1)+1):
        Bf[k+1].append(1)
#
# Forward bond prices at k. Trial values of topmost rate x[0] and down factor x[1] 
#
    for j in range(0,k+1):
        rate=modelRate(x[0],x[1],j,treeType) 
        Bf[k].append(exp(-rate*dtime)*((1/2)*Bf[k+1][j]+(1/2)*Bf[k+1][j+1]))
#
# Backward iterate all forward bond prices
#
    for i in range(k-1,-1,-1):
        for j in range(0,i+1):
            rate=modelRate(topRate[i],downFactor[i],j,treeType)
            Bf[i].append(exp(-rate*dtime)*((1/2)*Bf[i+1][j]+(1/2)*Bf[i+1][j+1]))
#
# Bond price and volatility errors
#
    g=[marketBondPrices[k+1][1]-Bf[0][0],marketBondPrices[k+1][2]-(1/(2*sqrt(dtime)))*log(Bf[1][1]/Bf[1][0])]
#
    return g
#
#---------------------------------------------------------------------------------
#