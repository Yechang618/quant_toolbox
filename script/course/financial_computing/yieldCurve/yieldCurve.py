#
import newtonRaphson
import cubicSpline
#---------------------------------------------------------------------------
def BondPriceError(rzero,pa):
    from math import exp
    n,bondMaturity,bondPrice,par,coupon,m,tc=pa
#
    knots=[]
    for i in range(n):knots.append([bondMaturity[i],rzero[i]])
#
#  evaluate the cubic spline functions
    splineCoeff=cubicSpline.cubicSpline(n,knots)
#
#  interpolate coupon discount rates at all payment dates tc
    rc=[]
    for i in range(n):
        rate=[]
        for j in range(m[i]):
            tau=tc[i][j]
            for L in range(i):
                if(tau>=bondMaturity[L] and tau<=bondMaturity[L+1]):
                    a,b,c,d=splineCoeff[L][0:4]
                    rate.append(a+b*tau+c*tau**2+d*tau**3)
                    break
        rc.append(rate)
#
#   calculate the bond price errors
    gvec=[]
    for i in range(n):
        npv=par[i]*exp(-rzero[i]*bondMaturity[i])
        for j in range(m[i]):
            npv+=coupon[i]*exp(-rc[i][j]*tc[i][j])
        gvec.append(bondPrice[i]-npv)
#
    return gvec
#
#---------------------------------------------------------------------------------
def calZeroRates(n,bondMaturity,bondPrice,par,coupon,m,tc,prec):
    from math import log
    pa=n,bondMaturity,bondPrice,par,coupon,m,tc
#
    rinit=-log(bondPrice[0]/par[0])/bondMaturity[0]
    rzero=[rinit for i in range(n)]
    rzero,precFlag,maxDev=newtonRaphson.newtonRaphson_vector(BondPriceError,rzero,n,prec,pa)
#
    return rzero
#
#---------------------------------------------------------------------------------