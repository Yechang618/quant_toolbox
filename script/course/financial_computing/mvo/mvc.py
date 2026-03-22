#
#-------------------------------------------------------------------
def mvc(rs):
    from statistics import mean
#
    nasset=len(rs)
    ndata=len(rs[1])
#
    mu,vc=[],[]
    for i in range(nasset):
        mu.append(mean(rs[i]))
        vcrow=[]		
        for j in range(nasset):
            vcrow.append(covariance(rs[i],rs[j]))
        vc.append(vcrow)
#
    return mu,vc
#
#------------------------------------------------------------------
def covariance(x,y):
    from statistics import mean
    n=len(x)
    mx=mean(x)
    my=mean(y)
    sum=0
    for i in range(n):
        sum=sum+(x[i]-mx)*(y[i]-my)
#
    return sum/n
#
#----------------------------------------------------------------------------------
#