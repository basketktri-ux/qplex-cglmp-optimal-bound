# Genel agirlikli qplex modeli (d=3): Bob'un onseli s duz degil, Alice agirliklari mu^x_a, Bob olcum agirliklari gamma^y_b serbest.
#   u^x_a : Alice x/a sonrasi Bob'un posterioru (9-simplex),  sum_a mu^x_a u^x_a = s  (her x icin ayni s)
#   v^y_b : Bob efekt vektoru (normalize),                     sum_b gamma^y_b v^y_b = c
#   p(a,b|x,y) = mu^x_a * gamma^y_b * (36 u^x_a . v^y_b - 3)          (Urgleichung, d=3)
#   qplex (germ): 12 vektorun tum ciftleri 1/12 <= w.w' <= 1/6
import sys
import numpy as np
from scipy.optimize import minimize

c = np.full(9, 1/9)
PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
NV = 2*3*9

def unpack(z):
    U = z[:NV].reshape(2, 3, 9); V = z[NV:2*NV].reshape(2, 3, 9)
    mu = z[2*NV:2*NV+6].reshape(2, 3); gam = z[2*NV+6:2*NV+12].reshape(2, 3)
    return U, V, mu, gam

def box(z):
    U, V, mu, gam = unpack(z)
    dots = np.einsum('xai,ybi->xyab', U, V)
    return mu[:, None, :, None] * gam[None, :, None, :] * (36*dots - 3)   # P[x,y,a,b]

def I3(P):
    tot = 0.0
    for x, y, g, l in PAIRS:
        for a in range(3):
            tot += P[x, y, a, (a-g) % 3] - P[x, y, a, (a-l) % 3]
    return tot

def eq_cons(z):
    U, V, mu, gam = unpack(z)
    s0 = mu[0] @ U[0]; s1 = mu[1] @ U[1]
    return np.concatenate([s0 - s1, gam[0] @ V[0] - c, gam[1] @ V[1] - c,
                           U.sum(-1).ravel() - 1, V.sum(-1).ravel() - 1, mu.sum(-1) - 1])

def ineq_cons(z):
    U, V, mu, gam = unpack(z)
    W = np.concatenate([U.reshape(6, 9), V.reshape(6, 9)])
    G = W @ W.T
    iu = np.triu_indices(12, 1)
    return np.concatenate([G[iu] - 1/12, 1/6 - G[iu], 1/6 - np.diag(G)])

def random_start(rng, scale):
    U = c + scale*rng.standard_normal((2, 3, 9)); V = c + scale*rng.standard_normal((2, 3, 9))
    mu = rng.dirichlet(np.ones(3), size=2); gam = rng.dirichlet(np.ones(3), size=2)
    return np.concatenate([U.ravel(), V.ravel(), mu.ravel(), gam.ravel()])

def solve(z0):
    bounds = [(0, 1)]*(2*NV) + [(0, 1)]*12
    r = minimize(lambda z: -I3(box(z)), z0, method='SLSQP', bounds=bounds,
                 constraints=[{'type': 'eq', 'fun': eq_cons}, {'type': 'ineq', 'fun': ineq_cons}],
                 options={'maxiter': 2000, 'ftol': 1e-12})
    z = r.x
    feas = max(np.abs(eq_cons(z)).max(), max(0, -ineq_cons(z).min()), max(0, -z.min()))
    return z, I3(box(z)), feas

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    best, results = None, []
    for t in range(n):
        z, val, feas = solve(random_start(rng, rng.choice([0.02, 0.05, 0.1, 0.2])))
        if feas < 1e-7:
            results.append(val)
            if best is None or val > best[1]:
                best = (z, val)
    results = np.array(results)
    print(f"uygulanabilir cozum: {len(results)}/{n}")
    print(f"en iyi I3 = {best[1]:.10f}   (2+2/sqrt3 = {2+2/np.sqrt(3):.10f})")
    print("en iyi 10:", np.round(np.sort(results)[-10:], 6))
    U, V, mu, gam = unpack(best[0])
    print("en iyi cozumde mu:", np.round(mu, 4).tolist(), " gamma:", np.round(gam, 4).tolist())
    print("Bob onseli s - c normu:", np.linalg.norm(mu[0] @ U[0] - c))
    np.save("genel_en_iyi.npy", best[0])
