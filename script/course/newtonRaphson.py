#
import mop
#---------------------------------------------------------------------------
def maxIteration():
    return 1000
#
#---------------------------------------------------------------------------
# Newton-Raphson procedure to determine the root of a single-variable function 
#
#   userfunc(x) - user-defined callable function 
#
#   Input:
#   x - As input, it is an initial guess of the root
#   prec - Required precision in the estimation
#   pa - Other passing arguments to the function
#
#   Output:
#   x - As output, it refers to the estimated root
#   precMet - Flag equals True if the procedure sucessfully determines
#             the root under the required precision
#             Flag equals False if it is unsuccessful
#   dev - Value of the function evaluated using the exit x
#
def newtonRaphson_scalar(userfunc,x,prec,pa):
    from math import fabs
#
    for nItr in range(maxIteration()):
        xtrial=x
        g=userfunc(xtrial,pa)
        gshift=userfunc(xtrial+prec,pa)
        dgdx=(gshift-g)/prec
        x=xtrial-g/dgdx
        precMet=(fabs(x-xtrial)<=prec)
        if(precMet): break
    dev=userfunc(x,pa)
#
    return x,precMet,dev
#
#---------------------------------------------------------------------------
def kroneckerDelta(i,j):
    if(i==j):delta=1
    else:delta=0
#
    return delta
#
#---------------------------------------------------------------------------
# Newton-Raphson procedure to determine the root vector of n functions
# with n variables
#
#   userfuncs(x) - user-defined callable functions
#                   Input variables x(0:n-1)
#                   Output function values g(0:n-1)
#   Input:
#   x(0:n-1) - As input, it is the initial guess of the root vector
#   prec - Required precision in the estimation
#   pa - Other passing arguments to the function
#
#   Output:
#   x(0:n-1) - As output, it refers to the estimated root vector
#   precMet - Flag equals True if the procedure sucessfully determines
#             the root vector under the required precision
#             Flag equals False if it is unsuccessful
#   maxDev - Maximum value among the functions evaluated using the exit x.
#
def newtonRaphson_vector(userfuncs,x,n,prec,pa):
    from math import fabs
    xtrial=[0 for i in range(n)]
    xshift=[0 for i in range(n)]
    omega=[[0 for j in range(n)] for i in range(n)]
#
    for nItr in range(maxIteration()):
        for i in range(n):xtrial[i]=x[i]
        g=userfuncs(xtrial,pa)
        for j in range(n):
            for i in range(n):xshift[i]=xtrial[i]+kroneckerDelta(i,j)*prec
            gshift=userfuncs(xshift,pa)
            for i in range(n):omega[i][j]=(gshift[i]-g[i])/prec
        Dx=mop.solveAxb(omega,g,n,0,0,0)
        for i in range(n):x[i]=xtrial[i]-Dx[i]
#
        precMet=True
        for i in range(n):
            precMet=precMet and (fabs(x[i]-xtrial[i])<=prec)
        if(precMet):break
#
    g=userfuncs(x,pa)
    maxDev=0
    for i in range(n):
        if(fabs(g[i])>maxDev):maxDev=fabs(g[i])
#
    return x,precMet,maxDev
#
#---------------------------------------------------------------------------






