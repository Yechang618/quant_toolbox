import math
import numpy as np
from scipy.interpolate import CubicSpline
import irTree


# =========================================================
# Input data from assignment
# =========================================================
yields_data = np.array([
    [1/12, 0.01932],
    [3/12, 0.01974],
    [6/12, 0.02012],
    [1.0,  0.02077],
    [2.0,  0.02156],
    [4.0,  0.02202],
    [7.0,  0.02259],
    [18.0, 0.01839]
], dtype=float)

vol_data = np.array([
    [1/12, 0.0008],
    [3/12, 0.0020],
    [6/12, 0.0040],
    [1.0,  0.0070],
    [2.0,  0.0110],
    [4.0,  0.0130],
    [7.0,  0.0140],
    [18.0, 0.0150]
], dtype=float)

# option parameters from assignment example
T = 1.0      # option maturity
K = 0.95     # strike
tau = 2.0    # bond maturity
par = 1.0    # face value
H = 0.98     # upper barrier
L = 0.90     # lower barrier

dt = 1/12    # use 1 month step


# =========================================================
# Common helper: build marketBondPrices for irTree and MC
# marketBondPrices[i] = [t_i, P(0,t_i), nu(0,t_i)]
# =========================================================
def build_market_bond_prices(max_steps):
    y_spline = CubicSpline(yields_data[:, 0], yields_data[:, 1], extrapolate=True)
    v_spline = CubicSpline(vol_data[:, 0], vol_data[:, 1], extrapolate=True)

    out = [[0.0, 1.0, 0.0]]
    for i in range(1, max_steps + 1):
        t = i * dt
        y = float(y_spline(t))
        v = float(v_spline(t))
        p = math.exp(-y * t)
        out.append([t, p, v])
    return out


# =========================================================
# Problem 1
# =========================================================
def build_bond_tree(rate_tree, maturity_steps, par_value=1.0):
    B = [[] for _ in range(maturity_steps + 1)]
    B[maturity_steps] = [par_value] * (maturity_steps + 1)

    for i in range(maturity_steps - 1, -1, -1):
        row = []
        for j in range(i + 1):
            val = math.exp(-rate_tree[i][j] * dt) * 0.5 * (B[i+1][j] + B[i+1][j+1])
            row.append(val)
        B[i] = row
    return B


def price_problem1():
    option_steps = int(round(T / dt))
    bond_steps = int(round(tau / dt))

    # irTree calibrates using marketBondPrices[k+1], so prepare one extra step
    market_bond_prices = build_market_bond_prices(bond_steps + 1)

    # directly use uploaded irTree.py
    rate_tree = irTree.irTree(
        timeHorizon=tau,
        n=bond_steps,
        treeType='normal',
        marketBondPrices=market_bond_prices,
        prec=1e-10
    )

    # build underlying zero-coupon bond tree
    bond_tree = build_bond_tree(rate_tree, bond_steps, par)

    # option tree
    V = [[] for _ in range(option_steps + 1)]

    # terminal payoff at t = T
    terminal = []
    for j in range(option_steps + 1):
        b = bond_tree[option_steps][j]
        if L < b < H:
            terminal.append(max(b - K, 0.0))
        else:
            terminal.append(0.0)
    V[option_steps] = terminal

    # backward induction with knock-out condition
    for i in range(option_steps - 1, -1, -1):
        row = []
        for j in range(i + 1):
            b = bond_tree[i][j]
            if L < b < H:
                val = math.exp(-rate_tree[i][j] * dt) * 0.5 * (V[i+1][j] + V[i+1][j+1])
                row.append(val)
            else:
                row.append(0.0)
        V[i] = row

    return V[0][0]


# =========================================================
# Problem 2
# Ho-Lee model + Monte Carlo
# =========================================================
def estimate_dFdt(discounts):
    # assignment formula
    n = len(discounts) - 1
    out = np.zeros(n)

    out[0] = (1 / dt**2) * math.log((discounts[1]**2) / (discounts[0] * discounts[2]))
    for i in range(1, n):
        out[i] = (1 / dt**2) * math.log((discounts[i]**2) / (discounts[i-1] * discounts[i+1]))
    return out


def estimate_F0(discounts):
    n = len(discounts) - 1
    F0 = np.zeros(n)
    for i in range(n):
        F0[i] = -(1 / dt) * math.log(discounts[i+1] / discounts[i])
    return F0


def ho_lee_bond_price(t, T_bond, r_t, P0_t, P0_T, F0_t, sigma):
    s = T_bond - t
    return (P0_T / P0_t) * math.exp(s * F0_t - 0.5 * sigma**2 * t * s**2 - s * r_t)


def price_problem2(nsim=50000, seed=12345):
    option_steps = int(round(T / dt))
    bond_steps = int(round(tau / dt))

    market_bond_prices = build_market_bond_prices(bond_steps + 1)
    discounts = [x[1] for x in market_bond_prices]

    # assignment calibration formulas
    sigma = market_bond_prices[1][2] / dt
    r0 = -(1 / dt) * math.log(market_bond_prices[1][1])
    dFdt = estimate_dFdt(discounts[:option_steps + 2])
    F0 = estimate_F0(discounts[:option_steps + 2])

    rng = np.random.default_rng(seed)
    payoffs = []

    for _ in range(nsim):
        r = r0
        disc = 1.0

        # check barrier at t=0
        bond_t = par * discounts[bond_steps]
        alive = (L < bond_t < H)

        for i in range(option_steps):
            if not alive:
                break

            t_i = i * dt

            # discount one step
            disc *= math.exp(-r * dt)

            # evolve short rate
            z = rng.normal()
            r = r + (dFdt[i] + sigma**2 * t_i) * dt + sigma * math.sqrt(dt) * z

            # bond price at next time
            t_next = (i + 1) * dt
            if i + 1 < option_steps:
                bond_t = par * ho_lee_bond_price(
                    t_next, tau, r,
                    discounts[i+1], discounts[bond_steps], F0[i+1], sigma
                )
            else:
                bond_t = par * ho_lee_bond_price(
                    t_next, tau, r,
                    discounts[i+1], discounts[bond_steps], F0[i], sigma
                )

            # barrier check
            if not (L < bond_t < H):
                alive = False

        if alive:
            payoff = max(bond_t - K, 0.0)
            payoffs.append(disc * payoff)
        else:
            payoffs.append(0.0)

    price = np.mean(payoffs)
    std_error = np.std(payoffs, ddof=1) / math.sqrt(len(payoffs))
    return price, std_error


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    p1 = price_problem1()
    p2, se2 = price_problem2(nsim=50000)

    print("Problem 1 price =", round(p1, 10))
    print("Problem 2 price =", round(p2, 10))
    print("Problem 2 std error =", round(se2, 10))