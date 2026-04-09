#
#------------------------------------------------------------------------
def legendre7(x):
    legendrefuncs=[None]
    legendrefuncs.append(1)
    legendrefuncs.append(x)	
    legendrefuncs.append((1/2)*(3*x**2-1))
    legendrefuncs.append((1/2)*(5*x**3-3*x))	
    legendrefuncs.append((1/8)*(35*x**4-30*x**2+3))	
    legendrefuncs.append((1/8)*(63*x**5-70*x**3+15*x))
    legendrefuncs.append((1/16)*(231*x**6-315*x**4+105*x**2-5))	
#
    return legendrefuncs
#
#--------------------------------------------------------------------------
def hermite7(x):
    hermitefuncs=[None]
    hermitefuncs.append(1)
    hermitefuncs.append(2*x)	
    hermitefuncs.append(4*x**2-2)
    hermitefuncs.append(8*x**3-12*x)	
    hermitefuncs.append(16*x**4-48*x**2+12)	
    hermitefuncs.append(32*x**5-160*x**3+120*x)
    hermitefuncs.append(64*x**6-480*x**4+720*x**2-120)	
#
    return hermitefuncs
#
#--------------------------------------------------------------------------

