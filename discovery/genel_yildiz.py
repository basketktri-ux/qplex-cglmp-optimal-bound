# Duz nokta etrafinda YILDIZ: tepe = duz nokta, 16 ortant simpleksi (kose: duz, duz + s_i rho e_i).
# Her simplekste: kose basina dual theta_v (afin interpolasyon), simplekse ozgu sabit w  =>  Z(mu) afin.
# Duz kosede: Z X* = 0 (tam tekillik) + tumleyende marj;  diger koselerde: Z >= t I.  Amac <= T her kosede.
import sys, itertools
import numpy as np
import cvxpy as cp
import genel_dual as D
from genel_yerel import flat_primal, S0_expr, new_theta, obj_of, TAN, flat, T, N

Xs = flat_primal()
_ev, _V = np.linalg.eigh(Xs)
KR = _V[:, _ev > 1e-9]; PP = _V[:, _ev <= 1e-9]

def Zexpr(mu, th, w):
    v = D.vvec(mu)
    Z = S0_expr(th) - D.Cmat(mu) - (cp.reshape(w, (N, 1), order='C') @ v.reshape(1, N) + v.reshape(N, 1) @ cp.reshape(w, (1, N), order='C'))/2
    return (Z + Z.T)/2

def simplex_cert(verts, flat_idx=None, solver='CLARABEL'):
    w = cp.Variable(N); t = cp.Variable(); cons = []
    for k, mu in enumerate(verts):
        th = new_theta(); Z = Zexpr(mu, th, w)
        cons += [obj_of(th) <= T, th[2] >= 0, th[3] >= 0]
        if k == flat_idx:
            cons.append(PP.T @ Z @ KR == 0)
            Zkk = KR.T @ Z @ KR
            cons += [Zkk[i, j] == 0 for i in range(KR.shape[1]) for j in range(i, KR.shape[1])]
            Zpp = PP.T @ Z @ PP
            cons.append((Zpp + Zpp.T)/2 - t*np.eye(PP.shape[1]) >> 0)
        else:
            cons.append(Z - t*np.eye(N) >> 0)
    pr = cp.Problem(cp.Maximize(t), cons + [t <= 1])
    try:
        pr.solve(solver=solver)
    except Exception:
        return 'hata', None
    return pr.status, (None if t.value is None else float(t.value))

if __name__ == "__main__":
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    res = []
    for signs in itertools.product((1, -1), repeat=4):
        verts = [flat] + [flat + s*rho*TAN[i] for i, s in enumerate(signs)]
        st, tv = simplex_cert(verts, flat_idx=0)
        res.append(tv if tv is not None else -1)
        print(signs, st, tv, flush=True)
    print(f"rho={rho}: en kucuk marj = {min(res)}  (16 simpleksin hepsi > 0 ise yerel bolge kapali)")
