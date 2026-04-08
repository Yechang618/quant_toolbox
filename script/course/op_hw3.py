import numpy as np

tolerance = 1e-12

def Qt(Qu, r, sigma, u, t , e):
    assert t >= u
    return Qu * np.exp((r - 0.5 * sigma ** 2) * (t - u) + sigma * np.sqrt(t - u) * e)

def generate_Q(S0, N, tau, u, sigma, r):
    np.random.seed(42)
    e1 = np.random.normal(0, 1, (N, 1))
    e2 = np.random.normal(0, 1, (N, 1))
    Qu_p = Qt(S0, r, sigma, 0, u, e1)
    Qu_n = Qt(S0, r, sigma, 0, u, -e1)
    Qtau_1 = Qt(Qu_p, r, sigma, u, tau, e2)
    Qtau_2 = Qt(Qu_n, r, sigma, u, tau, e2)
    Qtau_3 = Qt(Qu_p, r, sigma, u, tau, -e2)
    Qtau_4 = Qt(Qu_n, r, sigma, u, tau, -e2)
    return (Qu_p, Qtau_1), (Qu_n, Qtau_2), (Qu_p, Qtau_3), (Qu_n, Qtau_4)

def payoff(Qtau, Qu, S0, K):
    p_f = (Qu/(S0 + tolerance) * (1 + Qtau/(S0 + tolerance)))/2
    return np.array([1 if p <= K else 0 for p in p_f])

def pricing(S0, K, N, tau, u, sigma, r):
    Qtau_1, Qtau_2, Qtau_3, Qtau_4 = generate_Q(S0, N, tau, u, sigma, r)
    payoff_1 = payoff(Qtau_1[1], Qtau_1[0], S0, K)
    payoff_2 = payoff(Qtau_2[1], Qtau_2[0], S0, K)
    payoff_3 = payoff(Qtau_3[1], Qtau_3[0], S0, K)
    payoff_4 = payoff(Qtau_4[1], Qtau_4[0], S0, K)
    p = np.exp(-r * tau) * (payoff_1 + payoff_2 + payoff_3 + payoff_4)/4
    return np.mean(p), np.var(p) / (N-1)

if __name__ == "__main__":
    S0, K, tau, u = 1, 1.1, 1, 0.5
    r, sigma = 0.03, 0.2
    for n in [10, 100, 1000, 10000, 100000]:
        print(f"N={n}")
        price, var = pricing(S0, K, n, tau, u, sigma, r)
        print(f"Option Price: {price:.4f}, Variance: {var:.6f}")