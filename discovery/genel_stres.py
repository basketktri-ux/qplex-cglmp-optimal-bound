# Stres testi: Alice agirliklari duz OLMAYACAK sekilde sabitlenir (mu^x verilir), geri kalan her sey serbest.
# Her mu icin I3 yerel olarak maksimize edilir; bulunan kutunun I3'u ve Q1 gorunurlugu raporlanir.
import sys
import numpy as np
from scipy.optimize import minimize
import genel_qplex as G
from genel_q1_test import q1_visibility

def solve_fixed_mu(z0, mu_fix, gam_fix=None):
    def eqc(z):
        U, V, mu, gam = G.unpack(z)
        extra = [mu.ravel() - mu_fix.ravel()]
        if gam_fix is not None:
            extra.append(gam.ravel() - gam_fix.ravel())
        return np.concatenate([G.eq_cons(z)[:-2]] + extra)   # mu toplam=1 kisiti sabitlemeyle tekrar etmesin
    bounds = [(0, 1)]*(2*G.NV) + [(0, 1)]*12
    z0 = z0.copy(); z0[2*G.NV:2*G.NV+6] = mu_fix.ravel()     # baslangici sabit agirliklarla baslat
    r = minimize(lambda z: -G.I3(G.box(z)), z0, method='SLSQP', bounds=bounds,
                 constraints=[{'type': 'eq', 'fun': eqc}, {'type': 'ineq', 'fun': G.ineq_cons}],
                 options={'maxiter': 3000, 'ftol': 1e-12})
    z = r.x
    feas = max(np.abs(eqc(z)).max(), max(0, -G.ineq_cons(z).min()), max(0, -z.min()))
    return z, G.I3(G.box(z)), feas

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    tries = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    mus = [np.array([[.4, .3, .3], [.4, .3, .3]]), np.array([[.5, .3, .2], [.2, .3, .5]]),
           np.array([[.5, .25, .25], [1/3, 1/3, 1/3]]), np.array([[.6, .2, .2], [.6, .2, .2]]),
           np.array([[.45, .45, .1], [.1, .45, .45]])]
    for mu_fix in mus:
        best = (None, -9)
        for t in range(tries):
            z, val, feas = solve_fixed_mu(G.random_start(rng, rng.choice([0.05, 0.1, 0.2])), mu_fix)
            if feas < 1e-7 and val > best[1]:
                best = (z, val)
        if best[0] is None:
            print(mu_fix.tolist(), "uygulanabilir cozum yok"); continue
        U, V, mu, gam = G.unpack(best[0])
        P = G.box(best[0])
        print(f"mu={mu_fix.round(2).tolist()}  en iyi I3={best[1]:.6f}  |s-c|={np.linalg.norm(mu[0]@U[0]-G.c):.4f}"
              f"  gamma={gam.round(3).tolist()}  Q1 v*={q1_visibility(P):.6f}", flush=True)
