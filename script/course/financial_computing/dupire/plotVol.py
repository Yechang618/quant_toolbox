#
import matplotlib.pyplot as plt
import xlwings
import impVol
import volSkew
import volTerm
import localVol
#----------------------------------------------------------------------------------
def readfile(filename,sheetname):
    wb=xlwings.Book(filename)
    ws=wb.sheets[sheetname]
#
    assetPrice=float(ws["C2"].value)
    hisVol=float(ws["D2"].value)
    numberOfCall=int(ws["E2"].value)	
    riskFreeRate=float(ws["B5"].value)/100.0
#
    marketCallData=[]
    for i in range(numberOfCall):
        thisCall=[]
        for j in [8,9,10]:thisCall.append(float(ws.range(2+i,j).value))
        marketCallData.append(thisCall)
#	
    return assetPrice,riskFreeRate,hisVol,marketCallData,numberOfCall
#
#----------------------------------------------------------------------------------------
assetPrice,riskFreeRate,hisVol,marketCallData,numberOfCall=readfile('AAPL_CALL.xlsx','Call prices')
#
prec=1.e-8
tol=1.e-12
yteData,strikeData,ivData,ytebreak=impVol.impVol(assetPrice,riskFreeRate,hisVol,marketCallData,numberOfCall,prec,tol)
#
fig = plt.figure()
ax=plt.axes(projection ='3d')
ax.scatter(yteData,strikeData,ivData,marker='.',color='red')
ax.set_xlabel('yte')
ax.set_ylabel('strike')
ax.set_zlabel('iv')
plt.show()
#plt.savefig('temp.jpeg')
#
yteList,skewCoeff=volSkew.volSkewCoeff(assetPrice,riskFreeRate,yteData,strikeData,ivData,ytebreak)
#
for i in range(len(yteList)):
    print(yteList[i],skewCoeff[i][1],skewCoeff[i][2],skewCoeff[i][3])
#
for i in [0,2,8]:
    plt.plot(strikeData[ytebreak[i]:ytebreak[i+1]-1],ivData[ytebreak[i]:ytebreak[i+1]-1],'r.')
    k,y=[],[]
    for j in range(170,300):        
        k.append(j*1.0)
        y.append(volSkew.fittedVolSkewCurve(assetPrice,riskFreeRate,yteList,skewCoeff,(j*1.0),i))     
    plt.plot(k,y,color='r')
plt.show()
#
#
skewCoeffParameters=volTerm.fitVolSkewCoeffTerm(yteList,skewCoeff)
print(skewCoeffParameters[1])
print(skewCoeffParameters[2])
print(skewCoeffParameters[3])
#
for j in [1,2,3]:
    y=[]
    for i in range(len(yteList)):
        y.append(skewCoeff[i][j])
    plt.plot(yteList,y,'r.')
#
    tau,y=[],[]
    for i in range(1,400):
        tau.append(i*0.005)
        y.append(volTerm.volSkewCoeffTerm(skewCoeffParameters,i*0.005,j))
    plt.plot(tau,y,color='r')
    plt.show()
#
#
fig = plt.figure()
ax=plt.axes(projection ='3d')
ax.set_xlabel('yte')
ax.set_ylabel('strike')
ax.set_zlabel('iv')
#
for j in range(170,281,2):
    t,k,y=[],[],[]
    for i in range(3,400):
        t.append(i*0.005)
        k.append(j*1.0)		
        y.append(volTerm.fittedVolTermCurve(assetPrice,riskFreeRate,skewCoeffParameters,i*0.005,j*1.0))				
    ax.plot3D(t,k,y,color='red')	
plt.ylim(160,280)
plt.show()
#
#
fig = plt.figure()
ax=plt.axes(projection ='3d')
ax.set_xlabel('time')
ax.set_ylabel('price')
ax.set_zlabel('volatility')

for j in range(170,281,2):
    t,k,y=[],[],[]
    for i in range(3,100):
        t.append(i*0.005)
        k.append(j)		
        vol=localVol.localVol(assetPrice,riskFreeRate,skewCoeffParameters,j,i*0.005)
        y.append(vol)
    ax.plot3D(t,k,y,color='red')	
plt.ylim(160,280)	
plt.show()
