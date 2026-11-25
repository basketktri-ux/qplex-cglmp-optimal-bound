# Genel agirlikli qplex kutulari Q1 (NPA seviye 1, pozitiflikle) icinde mi?
# 1) Rastgele Bell fonksiyonelleri icin qplex uzerinde (yerel) maksimize et -> uc noktalara yakin kutular
# 2) Her kutu icin Q1 gorunurlugu v*: v*>=1 ise kutu Q1 icinde.
import sys
from itertools import product
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import genel_qplex as G
import npa_cglmp as N

def q1_visibility(P):
    Gm, cons, p, _ = N.npa_problem("1", positivity=True)
    v = cp.Variable()
    cons += [p(a, b, x, y) == v*P[x, y, a, b] + (1 - v)/9
             for a, b, x, y in product(range(3), range(3), range(2), range(2))]
    pr = cp.Problem(cp.Maximize(v), cons + [v <= 1.5])
    pr.solve(solver=cp.CLARABEL)
    return float(v.value)

def solve_obj(z0, Wt):
    bounds = [(0, 1)]*(2*G.NV) + [(0, 1)]*12
    r = minimize(lambda z: -np.sum(Wt*G.box(z)), z0, method='SLSQP', bounds=bounds,
                 constraints=[{'type': 'eq', 'fun': G.eq_cons}, {'type': 'ineq', 'fun': G.ineq_cons}],
                 options={'maxiter': 2000, 'ftol': 1e-12})
    z = r.x
    feas = max(np.abs(G.eq_cons(z)).max(), max(0, -G.ineq_cons(z).min()), max(0, -z.min()))
    return z, feas

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    worst = 9.0
    for t in range(n):
        Wt = rng.standard_normal((2, 2, 3, 3))          # rastgele Bell fonksiyoneli
        z, feas = solve_obj(G.random_start(rng, 0.1), Wt)
        if feas > 1e-7:
            continue
        P = G.box(z)
        U, V, mu, gam = G.unpack(z)
        vis = q1_visibility(P)
        worst = min(worst, vis)
        print(f"{t:3d}  v*={vis:.6f}  I3={G.I3(P):.4f}  min(mu)={mu.min():.3f}  |s-c|={np.linalg.norm(mu[0]@U[0]-G.c):.4f}", flush=True)
    print("EN KUCUK v* =", worst, "(>= 1 ise hepsi Q1 icinde)")
