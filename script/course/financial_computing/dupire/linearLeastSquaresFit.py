#
#------------------------------------------------------------------------
def straightLineFitSd(x,y,sd,h):
# Given x[0 to h],y[0 to h], and sd[0 to h]
    sum1,sumx,sumy,sumx2,sumxy=0,0,0,0,0
    for k in range(h+1):
        wf=1/sd[k]**2
        sum1+=wf
        sumx+=wf*x[k]
        sumy+=wf*y[k]
        sumx2+=wf*x[k]*x[k]
        sumxy+=wf*x[k]*y[k]
    slope=(sum1*sumxy-sumx*sumy)/(sum1*sumx2-sumx**2)
    intercept=(sumx2*sumy-sumx*sumxy)/(sum1*sumx2-sumx**2)
    return slope,intercept
#
#--------------------------------------------------------------------------
def straightLineFit(x,y,h):
# Given x[0 to h],y[0 to h], and constant sd
    sum1,sumx,sumy,sumx2,sumxy=0,0,0,0,0
    for k in range(h+1):
        sum1+=1
        sumx+=x[k]
        sumy+=y[k]
        sumx2+=x[k]*x[k]
        sumxy+=x[k]*y[k]
    slope=(sum1*sumxy-sumx*sumy)/(sum1*sumx2-sumx**2)
    intercept=(sumx2*sumy-sumx*sumxy)/(sum1*sumx2-sumx**2)
    return slope,intercept
#
#---------------------------------------------------------------------------------------------
# Linear regression: y[i]=coeff[1]*phi[i][1]+...+coeff[m]*phi[i][m]
# Given points y[0:n] and {phi[0:n][1],...,phi[0:n][m]}
# Least-square fitting of coefficients coeff[1:m]
#
def bfuncFit(y,phi,n,m):
    import mop
#  
    bvec=[0 for ir in range(m+1)]
    Amatrix=[[0 for ir in range(m+1)] for ic in range(m+1)]
# 
    for ir in range(1,m+1):
        for ic in range(1,m+1):
            for k in range(n+1):Amatrix[ir][ic]+=phi[k][ir]*phi[k][ic]
    for ir in range(1,m+1):
        for k in range(n+1):bvec[ir]+=y[k]*phi[k][ir]
    coeff=mop.solveAxb(Amatrix,bvec,m,1,1,1)
#
    return coeff
#
#---------------------------------------------------------------------------------------------