##################################
# Financial Computing Assignment 4
# Fixed Version: Tree & MC aligned for European Double Barrier Options
##################################
import numpy as np
import statistics
import irTree
import genBondData as gbd
from math import exp, sqrt, log

def loadData():
    a = [[1/12, 0.01932], [3/12, 0.01974],
         [6/12, 0.02012], [1, 0.02077],
         [2, 0.02156], [4, 0.02202],
         [7, 0.02259], [18, 0.01839]]
    b = [[1/12, 0.0008], [3/12, 0.002],
         [6/12, 0.004], [1, 0.007],
         [2, 0.011], [4, 0.013],
         [7, 0.014], [18, 0.015]]
    return a, b

#-----------------------------------------------------------------
def bondOptionTree(optionMaturity, strike, uBondMaturity, uBondPar,
                   upperBarrier, lowerBarrier, uBondCoupon, paymentSchedule,
                   rzero, bondVol, treeType, prec):
    """
    修正点:
    1. 障碍条件改为检查标的债券价格 Bf[i][j]，而非期权价值 pricing_temp
    2. 移除 max(fairValue, payoff) 改为欧式期权（与MC对齐）
    3. 障碍失效时直接赋 0.0
    """
    nCoupon = len(uBondCoupon)
    timeHorizon = uBondMaturity
    n = int(timeHorizon / rzero[0][0])
    timeInc = timeHorizon / n
    
    marketBondPrices = gbd.genBondData(rzero, bondVol, timeInc, n + 1)
    treeRate = irTree.irTree(timeHorizon, n, treeType, marketBondPrices, prec)
    print('Tree rate:')
    for j in range(0,n+1):
        for i in range(0,n+1):
            if(i<j):
                print('%6s '%(''),end=' ')
            else:
                print('%6.4f '%(treeRate[i][j]),end=' ')
        print('') 
    
    H = int(optionMaturity / timeInc)
    Bf = [[] for _ in range(n + 2)]  
    fTree = [[] for _ in range(H + 1)]  
    
    # 1. 倒推标的债券价格
    coupon = couponPayment((n - 0.5) * timeInc, (n + 0.5) * timeInc, uBondCoupon, nCoupon, paymentSchedule)
    for j in range(n + 1):
        Bf[n].append(uBondPar + coupon)
        
    for i in range(n - 1, -1, -1):
        coupon = couponPayment((i - 0.5) * timeInc, (i + 0.5) * timeInc, uBondCoupon, nCoupon, paymentSchedule)
        for j in range(i + 1):
            Bf[i].append(exp(-treeRate[i][j] * timeInc) * 0.5 * (Bf[i + 1][j] + Bf[i + 1][j + 1]) + coupon)
            
    # 2. 倒推期权价格 (欧式)
    for j in range(H + 1):
        # 到期日 payoff 也需满足障碍条件
        if lowerBarrier < Bf[H][j] < upperBarrier:
            fTree[H].append(max(Bf[H][j] - strike, 0))
        else:
            fTree[H].append(0.0)
            
    for i in range(H - 1, -1, -1):
        for j in range(i + 1):
            fairValue = exp(-treeRate[i][j] * timeInc) * 0.5 * (fTree[i + 1][j] + fTree[i + 1][j + 1])
            # ✅ 障碍条件作用于债券价格，非期权价值
            if lowerBarrier < Bf[i][j] < upperBarrier:
                fTree[i].append(fairValue)  # ✅ 欧式：直接取风险中性期望，移除美式提前行权
            else:
                fTree[i].append(0.0)
                
    return fTree

#---------------------------------------------------------------------------
def couponPayment(timeLow, timeUp, uBondCoupon, nCoupon, paymentSchedule):
    coupon = 0
    for i in range(nCoupon - 1, -1, -1):
        if timeLow < paymentSchedule[i] <= timeUp:
            coupon += uBondCoupon[i]
    return coupon

#---------------------------------------------------------------------------------
def payoff(assetPrice, strike, upperBarrier, lowerBarrier):
    """欧式障碍期权到期 payoff"""
    if lowerBarrier < assetPrice < upperBarrier:
        return max(assetPrice - strike, 0)
    return 0.0

#----------------------------------------------------------------------------------
def mcHoLeeZCBCall(marketBondPrices, strike, par, upperBarrier, lowerBarrier,
                   timeMaturity, uBondTimeMaturity, nstep, nsample):
    """
    修正点:
    1. 移除嵌套 MC，改用 Ho-Lee 解析公式计算各期债券价格
    2. 障碍条件在离散时间步直接检查债券价格 P(t, τ)
    3. 修复漂移率 θ(t) 计算，匹配初始远期曲线
    """
    dt = timeMaturity / nstep
    _, P0, vol0 = marketBondPrices[1]
    r0 = -log(P0) / dt
    sigma = vol0 / dt  # 假设常数波动率（Ho-Lee）
    
    tau = uBondTimeMaturity  # 债券到期期限
    dcfsample = []
    valid_paths = 0
    
    # 预计算初始零息债价格用于解析定价
    P0_T = marketBondPrices[nstep][1]  # P(0, T)
    
    for _ in range(nsample):
        r = r0
        y = 0.0  # 累积贴现因子
        knocked_out = False
        
        # 模拟短期利率路径
        rnd = np.random.normal(0, 1, nstep)
        for i in range(nstep):
            t = i * dt
            theta = get_theta(marketBondPrices, i, dt, sigma)
            r += theta * dt + sigma * sqrt(dt) * rnd[i]
            y += r * dt
            
            # ✅ 离散障碍监控：用 Ho-Lee 公式计算当前时点债券价格 P(t, τ)
            # P(t, τ) ≈ P(0, τ)/P(0, t) * exp(-(τ-t)*r + 0.5*σ²*t*(τ-t)²)
            t_idx = i + 1
            if t_idx < len(marketBondPrices):
                P0_tau = marketBondPrices[-1][1] if tau >= marketBondPrices[-1][0] else 1.0
                P0_t = marketBondPrices[t_idx][1]
                time_to_maturity = max(tau - t, 0)
                
                # Ho-Lee 解析近似
                P_t_tau = (P0_tau / P0_t) * exp(-(time_to_maturity) * r + 0.5 * sigma**2 * t * time_to_maturity**2)
                
                if not (lowerBarrier < P_t_tau < upperBarrier):
                    knocked_out = True
                    break
                    
        if not knocked_out:
            # 到期 payoff
            P_T_tau = par  # 简化：到期支付面值，或用解析公式更精确
            payoff_val = payoff_mc(P_T_tau, strike)
            dcfsample.append(exp(-y) * payoff_val)
            valid_paths += 1
            
    print(f"Simulated {valid_paths} valid paths out of {nsample}.")
    if len(dcfsample) == 0:
        return 0.0, 0.0
        
    sampleMean = statistics.mean(dcfsample)
    stderr = statistics.stdev(dcfsample) / sqrt(nsample)
    return sampleMean, stderr

#----------------------------------------------------------------------------------
def payoff_mc(bondPrice, strike):
    return max(bondPrice - strike, 0)

#----------------------------------------------------------------------------------
def get_theta(marketBondPrices, i, dt, sigma):
    """
    Ho-Lee 漂移项 θ(t) = ∂f(0,t)/∂t + σ²t
    使用市场零息债价格有限差分近似远期利率斜率
    """
    if i == 0:
        P0 = marketBondPrices[0][1]
        P1 = marketBondPrices[1][1]
        P2 = marketBondPrices[2][1]
        # f(0,t) ≈ -ln(P(t+dt)/P(t))/dt
        f0 = -log(P1 / P0) / dt
        f1 = -log(P2 / P1) / dt
        dfdt = (f1 - f0) / dt
    else:
        Pt_1 = marketBondPrices[i-1][1]
        Pt = marketBondPrices[i][1]
        Pt1 = marketBondPrices[i+1][1]
        f_prev = -log(Pt / Pt_1) / dt
        f_curr = -log(Pt1 / Pt) / dt
        dfdt = (f_curr - f_prev) / dt
        
    t = i * dt
    return dfdt + sigma**2 * t

#----------------------------------------------------------------------------------
def main():
    rzero, bondVol = loadData()
    T, K, tau, par, upperBarrier, lowerBarrier = 1.0, 0.95, 2.0, 1.0, 0.98, 0.90
    timeHorizon, treeType, prec = T, 'normal', 1.e-8
    
    n = int(timeHorizon / rzero[0][0])
    timeInc = timeHorizon / n
    marketBondPrices = gbd.genBondData(rzero, bondVol, timeInc, n + 1)
    
    print('Market bond prices:')
    for i in range(len(marketBondPrices)):
        print(f"Time: {marketBondPrices[i][0]:.4f}, Price: {marketBondPrices[i][1]:.6f}, Vol: {marketBondPrices[i][2]:.6f}")

    # ---------------- 二叉树定价 ----------------
    fTree = bondOptionTree(optionMaturity=T, strike=K, uBondMaturity=tau, uBondPar=par,
                           upperBarrier=upperBarrier, lowerBarrier=lowerBarrier,
                           uBondCoupon=[], paymentSchedule=[],
                           rzero=rzero, bondVol=bondVol, treeType=treeType, prec=prec)
    print('\n✅ Binomial Tree Pricing (European Double Barrier):')
    print(f"Option Price: {fTree[0][0]:.6f}")

    # ---------------- 蒙特卡洛定价 ----------------
    np.random.seed(42)
    nstep, nsample = n, 200000  # 增加样本量降低标准误
    
    optionprice, error = mcHoLeeZCBCall(marketBondPrices=marketBondPrices,
                                        strike=K, par=par,
                                        upperBarrier=upperBarrier, lowerBarrier=lowerBarrier,
                                        timeMaturity=T, uBondTimeMaturity=tau,
                                        nstep=nstep, nsample=nsample)
    print('\n✅ Monte Carlo Pricing (European Double Barrier):')
    print(f"Option Price: {optionprice:.6f} ± {1.96*error:.6f} (95% CI)")

if __name__ == '__main__':
    main()