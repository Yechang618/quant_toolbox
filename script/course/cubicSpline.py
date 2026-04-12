#
import mop
#
#-------------------------------------------------------------------
# Given n knots [[x1,y1]...,[xn,xn]]
# Return spline coefficients
#    [[a1,b1,c1,d1],...,[a(n-1),b(n-1),c(n-1),d(n-1)]]
#   
def cubicSpline(n,knots):
#
#  convert into x and y with labels [1 to n]
    x=[0 for i in range(n+1)]
    y=[0 for i in range(n+1)]
    # print('Test for cubic')
    # print(knots, n)
    for i in range(1,n+1):
        # print(i, knots[i-1])
        x[i]=knots[i-1][0]
        y[i]=knots[i-1][1]     
#
    L=[0 for i in range(4*(n-1)+1)]
    M=[[0 for j in range(4*(n-1)+1)] for i in range(4*(n-1)+1)]
#
#  define the column vector L
    for i in range(1,n):
        L[i]=y[i]
        L[n-1+i]=y[i+1]
#
#  define row pointers
    k=[0,n-1,2*(n-1),3*(n-1)-1,4*(n-1)-2,4*(n-1)-1]
#
#  define the entries of M in the first (n-1) rows
    for i in range(1,n):
        M[k[0]+i][4*(i-1)+1]=1
        M[k[0]+i][4*(i-1)+2]=x[i]        
        M[k[0]+i][4*(i-1)+3]=x[i]**2
        M[k[0]+i][4*(i-1)+4]=x[i]**3     
#
#  define the entries of M in the second (n-1) rows
    for i in range(1,n):
        M[k[1]+i][4*(i-1)+1]=1
        M[k[1]+i][4*(i-1)+2]=x[i+1]        
        M[k[1]+i][4*(i-1)+3]=x[i+1]**2
        M[k[1]+i][4*(i-1)+4]=x[i+1]**3       
#
#  define the entries of M in the following (n-2) rows
    for i in range(1,n-1):
        M[k[2]+i][4*(i-1)+2]=1
        M[k[2]+i][4*(i-1)+3]=2*x[i+1]        
        M[k[2]+i][4*(i-1)+4]=3*x[i+1]**2
        M[k[2]+i][4*(i-1)+6]=-1 
        M[k[2]+i][4*(i-1)+7]=-2*x[i+1]
        M[k[2]+i][4*(i-1)+8]=-3*x[i+1]**2
#
#  define the entries of M in the next (n-2) rows
    for i in range(1,n-1):
        M[k[3]+i][4*(i-1)+3]=2
        M[k[3]+i][4*(i-1)+4]=6*x[i+1]        
        M[k[3]+i][4*(i-1)+7]=-2
        M[k[3]+i][4*(i-1)+8]=-6*x[i+1] 
#
#  define the entries of M in the last 2 rows
    M[k[4]+1][3]=2
    M[k[4]+1][4]=6*x[1]
#
    M[k[5]+1][4*(n-2)+3]=2
    M[k[5]+1][4*(n-2)+4]=6*x[n]
#
#  solve the matrix equation for R
    R=mop.solveAxb(M,L,4*(n-1),1,1,1)
#
    splineCoeff=[]
    for i in range(1,n):
        coeff=[R[4*(i-1)+1],R[4*(i-1)+2],R[4*(i-1)+3],R[4*(i-1)+4]]
        splineCoeff.append(coeff)
#
    return splineCoeff
#
#---------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------
# Given n knots [([x1,y1]...,[xn,yn]]
# and spline coefficient 
#   [[a1,b1,c1,d1],...,[a(n-1),b(n-1),c(n-1),d(n-1)] 
# Interpolate yint for xint inside x1 and xn
#
def interpolation(n,knots,splineCoeff,xint):
#
#  extract all x with labels [1 to n]
    x=[0 for i in range(n+1)]
    for i in range(1,n+1):x[i]=knots[i-1][0]   
#
#  interpolate yint for xint
    if xint < x[1]:
        a,b,c,d=splineCoeff[0][0:4]
        yint=a+b*xint+c*xint**2+d*xint**3
        return yint
    elif xint > x[n]:
        a,b,c,d=splineCoeff[n-2][0:4]
        yint=a+b*xint+c*xint**2+d*xint**3
        return yint
    for k in range(1,n):
        if(xint>=x[k] and xint<=x[k+1]):
            a,b,c,d=splineCoeff[k-1][0:4]
            yint=a+b*xint+c*xint**2+d*xint**3
            break

    return yint
#
#---------------------------------------------------------------------------------



