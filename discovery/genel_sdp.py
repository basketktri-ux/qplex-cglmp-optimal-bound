# Genel agirliklar icin SDP gevsetmesi (Bloch bicimi):
#   p(a,b|x,y) = mu^x_a gam^y_b (1 + 2 A^x_a . B^y_b),  |A|,|B| <= 1,  tum ikili ic carpimlar >= -1/2,
#   sum_a mu^0_a A^0_a = sum_a mu^1_a A^1_a (= S),   sum_b gam^y_b B^y_b = 0.
# Sabit (mu, gam) icin 12 vektorun Gram matrisi uzerinde I3'u maksimize eden SDP (SIC pozitifligi ve boyut dusurulur).
import sys
import numpy as np
import cvxpy as cp

PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
iA = lambda x, a: 3*x + a
iB = lambda y, b: 6 + 3*y + b

def sdp_bound(mu, gam, verbose=False):
    X = cp.Variable((12, 12), symmetric=True)
    e = np.zeros(12); e[0:3] = mu[0]; e[3:6] = -mu[1]
    f0 = np.zeros(12); f0[6:9] = gam[0]
    f1 = np.zeros(12); f1[9:12] = gam[1]
    cons = [X >> 0, cp.diag(X) <= 1, X @ e == 0, X @ f0 == 0, X @ f1 == 0]
    cons += [X[i, j] >= -0.5 for i in range(12) for j in range(i+1, 12)]
    obj = 0
    for x, y, g, l in PAIRS:
        for a in range(3):
            for k, sgn in ((g, 1), (l, -1)):
                b = (a - k) % 3
                obj += sgn * mu[x][a] * gam[y][b] * (1 + 2*X[iA(x, a), iB(y, b)])
    pr = cp.Problem(cp.Maximize(obj), cons)
    pr.solve(solver=cp.CLARABEL)
    return pr.value

def weights(t):  # 8 parametre -> 4 olasilik uclusu (softmax)
    t = np.asarray(t).reshape(4, 3)
    w = np.exp(t - t.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return w[:2], w[2:]

if __name__ == "__main__":
    flat = np.full((2, 3), 1/3)
    print("duz agirliklar:", sdp_bound(flat, flat), "  2+2/sqrt3 =", 2 + 2/np.sqrt(3))
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    vals = []
    for t in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
        th = rng.standard_normal(12) * rng.choice([0.1, 0.3, 1.0])
        mu, gam = weights(th)
        vals.append((sdp_bound(mu, gam), th))
    vals.sort(key=lambda v: -v[0])
    print("rastgele agirliklarda en buyuk SDP degerleri:", [round(v[0], 6) for v in vals[:8]])
    # en iyi noktadan yerel arama (Nelder-Mead) ile agirliklar uzerinde maksimize et
    from scipy.optimize import minimize
    best = vals[0][1]
    r = minimize(lambda th: -sdp_bound(*weights(th)), best, method='Nelder-Mead',
                 options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-9})
    mu, gam = weights(r.x)
    print("agirliklar uzerinde yerel maksimum:", -r.fun)
    print("  mu =", mu.round(4).tolist(), " gam =", gam.round(4).tolist())
