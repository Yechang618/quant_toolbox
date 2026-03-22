#
#-------------------------------------------------------------------
def fdpricing(n,m,strike,knockOut,riskfree,contractSize,tau,ntau) :
    F=[0 for j in range(0,m+1)]
    S=[0 for j in range(0,m+1)]
#
    dt=tau[ntau]/n
    ds=2*knockOut/m
    P,Q=constructPQ(m,dt,ds,riskfree)
#
    for j in range(0,m+1):
        F[j]=payoff(contractSize,strike,j*ds)
        S[j]=j*ds
#
    for i in range(n-1,0-1,-1):
        F=iterateF(F,P,Q,m)      		
        F=boundary(F,m,ds,dt,i,strike,knockOut,contractSize,tau,ntau)
#
    return S,F
#
#-------------------------------------------------------------------
def constructPQ(m,dt,ds,riskfree):
    from math import exp
    P=[[0.0 for k in range(0,m+1)] for j in range(0,m+1)]
    Q=[[0.0 for k in range(0,m+1)] for j in range(0,m+1)]	
#
# insert corner entries
    P[0][0]=exp(riskfree*dt)
    P[m][m]=1.0
    Q[0][0]=1.0
    Q[m][m]=1.0
#
# insert tridiagonal entries
    for j in range(1,m-1+1):
        vol=CIRvol(j*ds)
        a=0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
        d=(riskfree*dt)+(vol**2*j**2*dt)
        c=-0.5*(riskfree*j*dt)-0.5*(vol**2*j**2*dt)
        P[j][j-1]=0.5*a
        P[j][j]=1.0+0.5*d
        P[j][j+1]=0.5*c
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
#------------------------------------------------------------------
def CIRvol(price):
    from math import sqrt
    volfactor=0.1
    return volfactor*sqrt(price)	
#
#-------------------------------------------------------------------
def payoff(contractSize,strike,price):
    return contractSize*(price-strike)
#
#------------------------------------------------------------------
def boundary(F,m,ds,dt,i,strike,knockOut,contractSize,tau,ntau):	
    eps=1.e-14
    Settlement=False    
    for k in range(ntau,1-1,-1):
        if(tau[k]>(i*dt-0.5*dt+eps) and tau[k]<=(i*dt+0.5*dt+eps)):
            Settlement= True   
            break
#
    if(Settlement):
        for j in range(0,m+1):
            if(j*ds>=knockOut):
                F[j]=payoff(contractSize,strike,j*ds)
            else:
                F[j]=F[j]+payoff(contractSize,strike,j*ds)
    else:		
        for j in range(0,m+1):
            if(j*ds>=knockOut and i!=0):
                F[j]=0		
    return F
#
#-------------------------------------------------------------------
#
import matplotlib.pyplot as plt
n,m=100,50
strike,knockOut,riskfree,contractSize=5,10,0.05,1
ntau,duration=12,1./12
tau=[k*duration for k in range(0,ntau+1)]
S,F=fdpricing(n,m,strike,knockOut,riskfree,contractSize,tau,ntau)
# Output option prices
for i in range(0,m+1):
    print('%6.2f%3s%8.6f'%(S[i],' , ',F[i]))
plt.plot(S,F)
plt.show()

