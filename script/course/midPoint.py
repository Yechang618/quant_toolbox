#
#------------------------------------------------------------- 
# Mid-point procedure that evaluates the root of a function
# func(x,pa) - user defined function of variable x with other
#              passing arguments pa.
# pa - passing arguments to func() defined as tuple pa=(arg1,arg2,...). 
# xplus, xminus - positive and negative of func(x) that initiates
#                 the search for which func(xplus)>0 and func(xminus)<0.
# xroot - root of func(x) as output.
#
def prec():
# precision of the procedure, i.e. |func(xroot)|<=prec
  return 1.e-14  
#
def midPoint(func,xplus,xminus,pa):
    from math import fabs
    xmid=(xplus+xminus)/2.0
    fmid=func(xmid,pa)
    nitr=1
    while fabs(fmid)>prec() and nitr <1000:
        if fmid>prec():
            xplus=xmid
        elif fmid<-prec():
            xminus=xmid
        xmid=(xplus+xminus)/2.0
        fmid=func(xmid,pa)
        nitr+=1
    xroot=xmid
#
    return xroot
#
#-------------------------------------------------------------