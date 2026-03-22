#
#-------------------------------------------------------------------
def fdpricing(n,m,strike,riskfree,expiration) :
#
    F=[0 for j in range(0,m+1)]
    S=[0 for j in range(0,m+1)]
#
    dt=expiration/n
    ds=2*strike/m
#
#  Define volatility structure sigma[j][i]
    sigma=localvol(n,m,dt,ds)
#
#  Initialize the price array according to the payoff condition
    for j in range(0,m+1):
        F[j]=payoff(strike,j*ds)
        S[j]=j*ds
#
#  Perform backward iteration
    for i in range(n-1,0-1,-1):
        P,Q=constructPQ(i,m,dt,ds,riskfree,sigma)
        F=iterateF(F,P,Q,m)      		
        F=boundary(F,m,ds,strike)
#
    return S,F
#
#-------------------------------------------------------------------
def constructPQ(i,m,dt,ds,riskfree,sigma):
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
        a=0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
        d=(riskfree*dt)+(vol**2*j**2*dt)
        c=-0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
        P[j][j-1]=0.5*a
        P[j][j]=1.0+0.5*d
        P[j][j+1]=0.5*c
        vol=sigma[j][i+1]
        a=0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
        d=(riskfree*dt)+(vol**2*j**2*dt)
        c=-0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
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
def localvol(n,m,dt,ds):
    from math import sqrt
    sigma=[[0.0 for i in range(0,n+1)] for j in range(0,m+1)]
    for i in range(0,n+1):
        for j in range(0,m+1):
            v,dvt=impvol(j*ds,i*dt)		
            sigma[j][i]=sqrt(v**2+2*(i*dt)*v*dvt)
    return sigma
#
def impvol(strike,term):
    h0,h1,h2=0.2,0.05,0.01
    v=h0+h1*term+h2*term**2
    dvt=h1+2*h2*term
    return v,dvt
#
#-------------------------------------------------------------------
def payoff(strike,price):
    return max(strike-price,0.0)
#
#------------------------------------------------------------------
def boundary(F,m,ds,strike):
    for j in range(0,m+1):
        intrinsicValue=payoff(strike,j*ds)
        F[j]=max(F[j],intrinsicValue) 
    return F
#
#-----------------------------------------------------------------
#
import matplotlib.pyplot as plt
expiration,strike,riskfree=1.0,10,0.05
n,m=100,50
S,F=fdpricing(n,m,strike,riskfree,expiration)
# Output option prices
for i in range(0,m+1):
    print('%6.2f%3s%8.6f'%(S[i],' , ',F[i]))
plt.plot(S,F)
plt.show()