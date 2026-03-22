#
import xlwings
import mvc
import mvo_strategies
#
#-------------------------------------------------------------------
def readfile(filename):
    wb=xlwings.Book(filename)
    ws1=wb.sheets['portfolio mvo']
    ws2=wb.sheets['dayclose']
# 
    nasset=int(ws1["B2"].value)
    ndata=int(ws1["B3"].value)
    horizon=int(ws1["B4"].value)
    portmean=ws1["B5"].value
    riskFree=ws1["B6"].value
#
    aptr=[]
    for i in range(nasset):
        aptr.append(int(ws1.range(i+8,3).value))
#
    rs=[]
    for i in range(0,nasset):
        r=[]
        openPrice=ws2.range(3,aptr[i]).value
        for j in range(horizon,ndata+1,horizon):
            closePrice=ws2.range(j+3,aptr[i]).value
            r.append((closePrice-openPrice)/openPrice)
            openPrice=closePrice
        rs.append(r)
#
    return rs,portmean,riskFree
#-----------------------------------------------------------------
def writefile(filename):
    wb=xlwings.Book(filename)
    ws=wb.sheets['portfolio mvo']
#
    ws.range(7,4).value=w0
    for i in range(0,len(w)):
        ws.range(i+8,4).value=w[i]
#
    return
#
#-----------------------------------------------------------------
datafile='mvo.xlsx'
rs,portmean,riskFree=readfile(datafile)
# 
mu,vc=mvc.mvc(rs)
nasset=len(mu)
u=[1 for i in range(0,nasset)]
#
w0,w=mvo_strategies.mvoLC(nasset,mu,vc,u,portmean,riskFree)
#
writefile(datafile)
#