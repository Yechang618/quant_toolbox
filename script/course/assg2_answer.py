#
import matplotlib.pyplot as plt
import xlwings
import mop
#-------------------------------------------------------------------
def readfile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['SKO American put option']
#  
    jmax=int(ws["B2"].value)
    imax=int(ws["B3"].value)  
    maturity=ws["B4"].value
    strike=ws["B5"].value
    lowerBarrier=ws["B6"].value	
    rebate=ws["B7"].value
    riskFree=ws["B8"].value
    ds=ws["B9"].value
#
    return jmax,imax,maturity,strike,lowerBarrier,rebate,riskFree,ds
#
#-----------------------------------------------------------------
def writefile(filename,S,F):
    wb=xlwings.Book(filename)
    ws=wb.sheets['SKO American put option']
#
    # jmax=len(S)-1
    # for i in range(jmax+1):
    #     ws.range(i+2,5).value=S[i]
    #     ws.range(i+2,6).value=F[i]
        
#
    plt.plot(S,F)
    plt.xlabel('Asset Price')
    plt.ylabel('Option Price')
    plt.show()
#
    return
#
#-------------------------------------------------------------------
def ifdCN(n,m,ds,strike,lowerBarrier,rebate,riskFree,maturity) :
#
    F=[0 for j in range(0,m+1)]
    S=[0 for j in range(0,m+1)]
#
    dt=maturity/n
#
#  Define volatility structure sigma[j][i]
    sigma=localvol(n,m,dt,ds,riskFree)
#
#  Initialize the price array according to the payoff condition and apply boundary
    for j in range(0,m+1):
        F[j]=payoff(strike,j*ds)
        S[j]=j*ds
    F=boundary(F,m,ds,strike,lowerBarrier,rebate)
#
#  Perform backward iteration
    for i in range(n-1,0-1,-1):
        P,Q=constructPQ(i,m,dt,ds,riskFree,sigma)
        F=iterateF(F,P,Q,m)      		
        F=boundary(F,m,ds,strike,lowerBarrier,rebate)
#
    return S,F
#
#-------------------------------------------------------------------
def constructPQ(i,m,dt,ds,riskFree,sigma):
    from math import exp
#
    P=[[0.0 for k in range(0,m+1)] for j in range(0,m+1)]
    Q=[[0.0 for k in range(0,m+1)] for j in range(0,m+1)]	
#
# insert corner entries
#    P[0][0]=exp(riskFree*dt)
    P[0][0]=1.0
    P[m][m]=1.0
    Q[0][0]=1.0
    Q[m][m]=1.0
#
# insert tridiagonal entries
    for j in range(1,m-1+1):
        vol=sigma[j][i]
        a=0.5*(riskFree*j*dt)-0.5*(vol**2*j**2*dt)
        d=(riskFree*dt)+(vol**2*j**2*dt)
        c=-0.5*(riskFree*j*dt)-0.5*(vol**2*j**2*dt)
        P[j][j-1]=0.5*a
        P[j][j]=1.0+0.5*d
        P[j][j+1]=0.5*c
        vol=sigma[j][i+1]
        a=0.5*(riskFree*j*dt)-0.5*(vol**2*j**2*dt)
        d=(riskFree*dt)+(vol**2*j**2*dt)
        c=-0.5*(riskFree*j*dt)-0.5*(vol**2*j**2*dt)
        Q[j][j-1]=-0.5*a
        Q[j][j]=1.0-0.5*d
        Q[j][j+1]=-0.5*c
#
    return P,Q
#
#------------------------------------------------------------------
def iterateF(F,P,Q,m) :
    import mop
#  calculate P^(-1)*Q*F
    F=mop.multAbc(Q,F,m+1,0,0,0)
    F=mop.solveAxb(P,F,m+1,0,0,0)        
    return F
#
#-------------------------------------------------------------------
def localvol(n,m,dt,ds,riskFree):
    from math import sqrt
    sigma=[[0.0 for i in range(0,n+1)] for j in range(0,m+1)]
    for i in range(1,n+1):
        for j in range(0,m+1):
            v,dvdt,xdvdx=impVolTerms(j*ds,i*dt,riskFree)		
            sigma[j][i]=sqrt(v**2+2*(i*dt)*v*dvdt+2*riskFree*(i*dt)*v*xdvdx)
    return sigma
#
def impVolTerms(x,t,riskFree):
    from math import log
    h0,h1,h2,s1=0.25,-0.05,0.01,0.001
    v=h0+h1*t+h2*t**2+s1*(log(x+1.e-14)-riskFree*t)**2
    dvdt=h1+2*h2*t-2*s1*riskFree*(log(x+1.e-14)-riskFree*t)
    xdvdx=2*s1*(log(x+1.e-14)-riskFree*t)
    return v,dvdt,xdvdx
#
#-------------------------------------------------------------------
def payoff(strike,price):
    return max(strike-price,0.0)
#
#------------------------------------------------------------------
def boundary(F,m,ds,strike,lowerBarrier,rebate):
    for j in range(0,m+1):
        if((j*ds)<=lowerBarrier ):
            F[j]=rebate*payoff(strike,j*ds)
        else:			
            intrinsicValue=payoff(strike,j*ds)
            F[j]=max(F[j],intrinsicValue) 
    return F
#
#-----------------------------------------------------------------
#
jmax,imax,maturity,strike,lowerBarrier,rebate,riskFree,ds=readfile('gui2.xlsx')
S,F=ifdCN(imax,jmax,ds,strike,lowerBarrier,rebate,riskFree,maturity)
writefile('gui2.xlsx',S,F)
#
