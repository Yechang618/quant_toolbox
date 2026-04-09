#
def seedinit(seed):
    global idum
    idum=seed
    return
#
#--------------------------------------------------------------------------
# Uniform number generator between 0.0 and 1.0 
#
def ran0():
    global idum
    IA=16807
    IM=2147483647
    IQ=127773
    IR=2836    
    MASK=123459876
    AM=1.0/IM
    idum=idum^MASK
    k=int(round(idum/IQ))
    idum=IA*(idum-k*IQ)-IR*k
    if(idum < 0):
        idum=idum+IM
    un=AM*idum
    idum=idum^MASK
    return un
#
#--------------------------------------------------------------------------
# Generation of single standard normal random number 
# by polar rejection method
# snn[0] is the usable number and snn[1] is a reserve
#
def stdnormnum(lastsnn):
    from math import sqrt,log
#
    if(lastsnn==None or len(lastsnn)==1):
        w=None
        while(w==None or w>=1):
            u1=2*ran0()-1
            u2=2*ran0()-1
            w=u1*u1+u2*u2
        fac=sqrt(-2*log(w)/w)
        snnsave=fac*u1
        snnuse=fac*u2        
        snn=[snnuse,snnsave]
    else:
        snnuse=lastsnn[1]
        snn=[snnuse]
#
    return snn
#
#--------------------------------------------------------------------------
#