#
#--------------------------------------------------------------------------
#  Subroutine to solve vector xvec from
#    Amatrix(iptr:iptr+n-1,jptr:jptr+n-1)*xvec(kptr:kptr+n-1)
#                                      = bvec(kptr:kptr+n-1)
#
def solveAxb(Amatrix,bvec,n,iptr,jptr,kptr):
    wAmatrix=[[0 for j in range(n+1)] for i in range(n+1)]
    wbvec=[0 for i in range(n+1)]
    wxvec=[0 for i in range(n+1)]
    xvec=[0 for i in range(n+1)]
#
    for i in range(1,n+1):
        wbvec[i]=bvec[kptr+i-1]
        for j in range(1,n+1):
            wAmatrix[i][j]=Amatrix[iptr+i-1][jptr+j-1]
#
    Umatrix,Wvec,Vmatrix=svdcmp(wAmatrix,n,n)
    wxvec=svbksb(Umatrix,Wvec,Vmatrix,n,n,wbvec)
#
    for i in range(kptr,kptr+n-1+1):
        xvec[i]=wxvec[i-kptr+1]
#
    return xvec
#
#------------------------------------------------------------------------
#  Given a matrix A, this subroutine computes its singular value
#  decomposition : A(mxn) = U(mxn) W(nxn) V^T(nxn).
#  
#   - The matrix U is output as U
#   - The diagonal elements of W is output as a vector W.
#   - The matrix V is output as V
#
#   refer to 'Numerical Recipes' Page 60 SVD algorithm
#   Version of March 20, 1998
#
def svdcmp(A,m,n):
    from math import fabs
    from math import sqrt
    from math import copysign
    prec=1.e-14
#
    if(m<n):
        print('Error: row size(m) must be greater than or equal column size(n) in svdcmp')
        input()  
#
    U=[[0 for j in range(n+1)] for i in range(m+1)]
    V=[[0 for j in range(n+1)] for i in range(n+1)]
    W=[0 for i in range(n+1)]
    RV1=[0 for i in range(n+1)]
#
    for i in range(1,m+1):
        for j in range(1,n+1):
            U[i][j]=A[i][j]
#
# Householder reduction to bidiagonal form
# 
    g=0.0
    scale=0.0
    anorm=0.0
    for i in range(1,n+1):
        L=i+1
        RV1[i]=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i<=m):
            for k in range(i,m+1):
                scale=scale+fabs(U[k][i])
            if(scale!=0.0):
                for k in range(i,m+1):
                    U[k][i]=U[k][i]/scale
                    s=s+U[k][i]*U[k][i]
                f=U[i][i]
                g=-copysign(sqrt(s),f)
                h=f*g-s
                U[i][i]=f-g
                if(i!=n):
                    for j in range(L,n+1):
                        s=0.0
                        for k in range(i,m+1):
                            s=s+U[k][i]*U[k][j]
                        f=s/h
                        for k in range(i,m+1):
                            U[k][j]=U[k][j]+f*U[k][i]
                for k in range(i,m+1):
                    U[k][i]=scale*U[k][i]
        W[i]=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i<=m and i!=n):
            for k in range(L,n+1):
                scale=scale+fabs(U[i][k])
            if(scale!=0.0):
                for k in range(L,n+1):
                    U[i][k]=U[i][k]/scale
                    s=s+U[i][k]*U[i][k]
                f=U[i][L]
                g=-copysign(sqrt(s),f)
                h=f*g-s
                U[i][L]=f-g
                for k in range(L,n+1):
                    RV1[k]=U[i][k]/h
                if(i!=m):
                    for j in range(L,m+1):
                        s=0.0
                        for k in range(L,n+1):
                            s=s+U[j][k]*U[i][k]
                        for k in range(L,n+1):
                            U[j][k]=U[j][k]+s*RV1[k]
                for k in range(L,n+1):
                    U[i][k]=scale*U[i][k]
        anorm=max(anorm,fabs(W[i])+fabs(RV1[i]))        
#
# Accumulation of right-hand transformation
#
    for i in range(n,1-1,-1):
        if(i<n):
            if(g!=0.0):
                for j in range(L,n+1):
                    V[j][i]=(U[i][j]/U[i][L])/g
                for j in range(L,n+1):
                    s=0.0
                    for k in range(L,n+1):
                        s=s+U[i][k]*V[k][j]
                    for k in range(L,n+1):
                        V[k][j]=V[k][j]+s*V[k][i]
            for j in range(L,n+1):
                V[i][j]=0.0
                V[j][i]=0.0
        V[i][i]=1.0
        g=RV1[i]
        L=i
#
# Accumulation of the left-hand transformation
#
    for i in range(n,1-1,-1):
        L=i+1
        g=W[i]
        if(i<n):
            for j in range(L,n+1):
                U[i][j]=0.0
        if(g!=0.0):
            g=1.0/g
            if(i!=n):
                for j in range(L,n+1):
                    s=0.0
                    for k in range(L,m+1):
                        s=s+U[k][i]*U[k][j]
                    f=(s/U[i][i])*g
                    for k in range(i,m+1):
                        U[k][j]=U[k][j]+f*U[k][i]
            for j in range(i,m+1):
                U[j][i]=U[j][i]*g
        else:
            for j in range(i,m+1):
                U[j][i]=0.0
        U[i][i]=U[i][i]+1.0
#
# Diagonalization of the bidiagonal form
#
    for k in range(n,1-1,-1):
        for its in range(1,300+1):
            flag= True
            for L in range(k,1-1,-1):
                nm=L-1
                if(fabs(RV1[L])<prec):
                    flag= False
                    break
                if(fabs(W[nm])<prec):
                    break
#
            if(flag==True):
                c=0.0
                s=1.0
                for i in range(L,k+1):
                    f=s*RV1[i]
                    if((fabs(f)+anorm)!=anorm):
                        g=W[i]
                        h=sqrt(f*f+g*g)
                        W[i]=h
                        h=1.0/h
                        c=g*h
                        s=-(f*h)
                        for j in range(1,m+1):
                            y=U[j][nm]
                            z=U[j][i]
                            U[j][nm]=(y*c)+(z*s)
                            A[j][i]=-(y*s)+(z*c)
#
            z=W[k]            
            if(L==k):
                if(z<0.0):
                    W[k]=-z
                    for j in range(1,n+1):
                        V[j][k]=-V[j][k]
                break
            if(its==300):
                print ('Error: No convergence in svdcmp')
            x=W[L]
            nm=k-1
            y=W[nm]
            g=RV1[nm]
            h=RV1[k]
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g=sqrt(f*f+1.0)
            f=((x-z)*(x+z)+h*((y/(f+copysign(g,f)))-h))/x
#           
# Next QR transformation
#
            c=1.0
            s=1.0
            for j in range(L,nm+1):
                i=j+1
                g=RV1[i]
                y=W[i]
                h=s*g
                g=c*g
                z=sqrt(f*f+h*h)
                RV1[j]=z
                c=f/z
                s=h/z
                f=(x*c)+(g*s)
                g=-(x*s)+(g*c)
                h=y*s
                y=y*c
                for jj in range(1,n+1):
                    x=V[jj][j]
                    z=V[jj][i]
                    V[jj][j]=(x*c)+(z*s)
                    V[jj][i]=-(x*s)+(z*c)
                z=sqrt(f*f+h*h)
                W[j]=z
                if(z!=0.0):
                    z=1.0/z
                    c=f*z
                    s=h*z
                f=(c*g)+(s*y)
                x=-(s*g)+(c*y)
                for jj in range(1,m+1):
                    y=U[jj][j]
                    z=U[jj][i]
                    U[jj][j]=(y*c)+(z*s)
                    U[jj][i]=-(y*s)+(z*c)
            RV1[L]=0.0
            RV1[k]=f
            W[k]=x
#
    return U,W,V
#
#--------------------------------------------------------------------------
#  Subroutine to solve A(mxn) x(nx1) = b(mx1) for vector X using
#  singular value decomposition of A(mxn) = U(mxn) W(nxn) V^T(nxn).
#  
#  - The decomposition U, V, and W are inputed from the routine svdcmp
#
#  NOTE : If Ax=b is not solvable, the routine returns the specific x that
#         minimize the norm of Ax-b
#
#   refer to 'Numerical Recipes' Page 57
#
def svbksb(U,W,V,m,n,b):
    TMP=[0 for i in range(n+1)]
    x=[0 for i in range(n+1)]
#
    for j in range(1,n+1):
        s=0.0
        if(W[j]!=0.0):
            for i in range(1,m+1):
                s=s+U[i][j]*b[i]
            s=s/W[j]
        TMP[j]=s
    for j in range(1,n+1):
        s=0.0
        for jj in range(1,n+1):
            s=s+V[j][jj]*TMP[jj]
        x[j]=s
#
    return x
#
#------------------------------------------------------------------------
#



