#
import matplotlib.pyplot as plt
import xlwings
import impVol
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