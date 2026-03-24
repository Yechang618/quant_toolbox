#
import newtonRaphson
#
#---------------------------------------------------------------------------
def funcTest(x):
    from math import sqrt
    return (x**2+x-1)**3
#
#---------------------------------------------------------------------------
def integration(userfunc,a,b,n):
#
    inc=(b-a)/n
    intg=(inc/2)*(userfunc(a)+userfunc(b))
    for i in range(1,n):
        intg+=inc*userfunc(a+i*inc)
#
    return intg
#
#---------------------------------------------------------------------------
def improve(userfunc,a,b,h):
#    
    P=[[0.0 for m in range(0,k+1)] for k in range(0,h+1)]
#   
    for k in range(h+1):
        P[k][0]=integration(userfunc,a,b,2**k)
        for m in range(1,k+1):
            P[k][m]=(4**m*P[k][m-1]-P[k-1][m-1])/(4**m-1)
#
    return P[h][h]
#
#---------------------------------------------------------------------------
def improveZero(b,pa):
#
    userfunc,a,h=pa
    return improve(userfunc,a,b,h)
#---------------------------------------------------------------------------    
a=-1
b=1
n=10000
print(integration(funcTest,a,b,n))


h=10
print(improve(funcTest,a,b,h))
#
pa=funcTest,a,h
prec=1.e-8
x=1
x,precMet,dev=newtonRaphson.newtonRaphson_scalar(improveZero,x,prec,pa)
print(x,precMet,dev)