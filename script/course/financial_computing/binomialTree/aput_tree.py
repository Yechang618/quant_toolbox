#
from math import exp,sqrt
#-------------------------
def omega(tf,ti):
    vi,dvti=impvol(0.0,ti)
    vf,dvtf=impvol(0.0,tf)
    omega=tf*vf*vf-ti*vi*vi
    return omega
#
def impvol(strike,term):
    h0,h1,h2=0.2,0.05,0.01
    v=h0+h1*term+h2*term*term
    dvt=h1+2*h2*term
    return v,dvt
#
def treefactor(i,type,r,dt):
    s=omega((i+1)*dt,i*dt)
    if(type=='crr'):
        w=exp(-r*dt)+exp(r*dt+s)
        u=0.5*w+0.5*sqrt(w*w-4.0)  		
        d=1.0/u
        p=(exp(r*dt)-1.0/u)/(u-1.0/u)
    elif(type=='jr'):
        w=exp(s)-1.0
        u=exp(r*dt)*(1.0+sqrt(w))
        d=exp(r*dt)*(1.0-sqrt(w))
        p=0.5
    return p,u,d
#
#-------------------------
def decbinArray(i,j):
    A=['0' for k in range(0,i)]
    B=bin(j)[2:]
    for k in range(0,len(B)):
        A[i-len(B)+k]=B[k]
    return A
#
def branching(i,j,type,r,dt):
    bseq=1.0
    if(i==0): return bseq	
    A=decbinArray(i,j)
    for k in range(0,i):
        p,u,d=treefactor(k,type,r,dt)
        if(A[k]=='0'):
            bseq=bseq*u
        else:
            bseq=bseq*d
    return bseq
#
#------------------------------------------
def payoff(strike,assetprice):
    return max(strike-assetprice,0.0)

def boundary(price,strike,assetprice):
    intrinsicValue=payoff(strike,assetprice) 
    return max(price,intrinsicValue)
#	
#------------------------------------------
def treepricing(n,assetprice,strike,expiration,r,type):
    dt=expiration/n
    F=[0.0 for j in range(0,2**n)]
# 
    for j in range(0,2**n):   
        bseq=branching(n,j,type,r,dt)
        F[j]=payoff(strike,bseq*assetprice)
#
    for i in reversed(range(0,n)): 
        p,u,d=treefactor(i,type,r,dt)
        for j in range(0,2**i):  	
            price=exp(-r*dt)*(p*F[2*j]+(1.0-p)*F[2*j+1])		
            bseq=branching(i,j,type,r,dt)		
            F[j]=boundary(price,strike,bseq*assetprice)
#
    return F[0]
#
#------------------------------------------
n,assetprice,strike,expiration,r,type=10,10.0,10.0,1.0,0.05,'crr'
contractprice=treepricing(n,assetprice,strike,expiration,r,type)
print(contractprice)
#

