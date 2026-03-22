import xlwings
#----------------------------------------------------------------------------------
def readfile(filename,sheetname1,sheetname2):
    wb=xlwings.Book(filename)
    ws1=wb.sheets[sheetname1]
    ws2=wb.sheets[sheetname2]
#
    totalstep=int(ws1["B2"].value)
    t=int(ws1["B4"].value)	
    tstar=int(ws1["B5"].value)
    m=int(ws1["B6"].value)	
    z=ws1["B13:D13"].value
   	
    priceReturn=[None]
    for i in range(1,totalstep+1):
        priceReturn.append(float(ws2.range(i,2).value))
#	
    return priceReturn,totalstep,t,tstar,m,z
#
#----------------------------------------------------------------------------------
def writefile(filename,sheetname):
    wb=xlwings.Book(filename)
    ws=wb.sheets['model']
#
    ws["B8"].value=mu
    ws["B9"].value=longVar
    ws["B10"].value=alpha
    ws["B11"].value=beta
#
    ws["B15:D15"].value=conf
    ws["B17"].value=predVar
#
    return
#
#-----------------------------------------------------------------------------
# Given {priceReturn[1],...,priceReturn[tlast]}, 
# evaluate the backtesting confidence level from tbegin to tlast given list of z
#
def backtest(mu,longVar,alpha,beta,priceReturn,tbegin,tlast,z):
    from math import sqrt
    listofVar=[None]
    garchVar=longVar
    listofVar.append(garchVar)
    for i in range(2,tlast+2):
        garchVar=(1-alpha-beta)*longVar+alpha*(priceReturn[i-1]-mu)**2+beta*garchVar
        listofVar.append(garchVar)
#
    conf=[]
    for k in range(len(z)):
        n=0
        for i in range(tbegin,tlast+1):
            if(priceReturn[i]>(mu-z[k]*sqrt(listofVar[i])) and priceReturn[i]<=(mu+z[k]*sqrt(listofVar[i]))):
                n=n+1
        conf.append(n/(tlast-tbegin+1))
#
    return conf,listofVar[tlast+1]
#
#-----------------------------------------------------------------------------
# Given {priceReturn[1],...,priceReturn[tlast]}, 
# evaluate the best alpha and beta for m increments  
#
def simpleSearch(m,mu,longVar,priceReturn,tstar,tlast):    
    minlogL=1.e10
    for i in range(0,m):
        for j in range(0,m-i):
            y=[(i/m),(j/m)]
            logL=logLikelihood(y,mu,longVar,priceReturn,tstar,tlast,-1)
            if logL<=minlogL :
                alpha,beta=y
                minlogL=logL
    return alpha,beta
#
#---------------------------------------------------------------------------------
def lowerLimitAlpha(y):
    return y[0]
#
def lowerLimitBeta(y):
    return y[1]
#
def limitAlphaBeta(y):
    tol=1.e-6
    return -y[0]-y[1]+1-tol
#
#---------------------------------------------------------------------------------
# Given {priceReturn[1],...,priceReturn[tlast]}, 
# evaluate log of likelihood 
#
def logLikelihood(y,mu,longVar,priceReturn,tstar,tlast,sign):
    from math import log
    alpha,beta=y
    garchVar=longVar
    sum=0
    for i in range(2,tlast+1):
        garchVar=(1-alpha-beta)*longVar+alpha*(priceReturn[i-1]-mu)**2+beta*garchVar
        if(i>=tstar):
            sum=sum+(1/2)*log(garchVar)+(1/2)*(priceReturn[i]-mu)**2/garchVar
    logL=(-1)*sum 
#   
    return sign*logL
#
#-----------------------------------------------------------------------------
import numpy,statistics
from scipy.optimize import minimize
#
priceReturn,totalstep,t,tstar,m,z=readfile('DJI.xlsx','model','returns')
mu=statistics.mean(priceReturn[1:t+1])
longVar=statistics.variance(priceReturn[1:t+1])
ytrial=numpy.array(simpleSearch(m,mu,longVar,priceReturn,tstar,t))
#
sign=-1
cons=({'type':'ineq','fun':lowerLimitAlpha},{'type':'ineq','fun':lowerLimitBeta},{'type':'ineq','fun':limitAlphaBeta})
alpha,beta=minimize(logLikelihood,ytrial,args=(mu,longVar,priceReturn,tstar,t,sign),constraints=cons).x
#
conf,predVar=backtest(mu,longVar,alpha,beta,priceReturn,t+1,totalstep,z)
writefile('DJI.xlsx','model')
#
#----------------------------------------------------------------------------------
#