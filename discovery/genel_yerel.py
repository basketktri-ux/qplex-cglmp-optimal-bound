# Yerel sertifika: duz noktanin etrafinda capraz-politop (kose: duz +- rho*e_i, 4 teget yon).
# Dual carpanlar theta(mu) = theta_c + sum_i delta_i Theta_i (afin), w sabit  =>  Z(mu) afin  =>  koselerde PSD yeter.
# Duz noktada tekillik kacinilmaz: Z(mu_v) X* = 0 (X* = qplex orneginin tam Gram'i) dayatilir,
# geri kalan alt uzayda en buyuk marj t aranir.
import io, contextlib, importlib.util, sys
import numpy as np
import cvxpy as cp
from genel_alice import iA, iB, E0, PAIRS
import genel_dual as D

N = 13
T = 2 + 2/np.sqrt(3)
flat = np.full((2, 3), 1/3)
e1 = np.array([1, -1, 0])/np.sqrt(2); e2 = np.array([1, 1, -2])/np.sqrt(6)
TAN = [np.array([e1, 0*e1]), np.array([e2, 0*e2]), np.array([0*e1, e1]), np.array([0*e1, e2])]

def flat_primal():
    spec = importlib.util.spec_from_file_location("g", "cglmp_qplex_kontrol.py"); g = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(g)
    bl = lambda p: np.sqrt(18)*(np.array(p) - 1/9)
    vecs = np.zeros((N, 10)); vecs[E0, 0] = 1
    for x in range(2):
        for a in range(3):
            vecs[iA(x, a)] = np.concatenate([[1], np.sqrt(2)*bl(g.u(x, a))])
    for y in range(2):
        for b in range(3):
            vecs[iB(y, b)] = (1/3)*np.concatenate([[1], np.sqrt(2)*bl(g.v(y, b))])
    return vecs @ vecs.T

def S0_expr(th):
    y00, yA, zA, u, wy = th
    S = y00 * D.E(0, 0)
    for k, i in enumerate(D.ALICE):
        S = S + yA[k]*D.E(i, 0) + zA[k]*D.E(i, i)
    for k, (i, j) in enumerate(D.OFF):
        S = S - u[k]*D.E(i, j)
    for y in range(2):
        b = D.bvec(y)
        S = S - (cp.reshape(wy[y], (N, 1), order='C') @ b.reshape(1, N) + b.reshape(N, 1) @ cp.reshape(wy[y], (1, N), order='C'))/2
    return S

def new_theta():
    return (cp.Variable(), cp.Variable(6), cp.Variable(6), cp.Variable(len(D.OFF)), cp.Variable((2, N)))

def obj_of(th):
    return th[0] + cp.sum(th[1]) + 3*cp.sum(th[2])

def local_cert(rho, impose_kernel=True, solver='CLARABEL'):
    Xs = flat_primal()
    evals, evecs = np.linalg.eigh(Xs)
    Kr = evecs[:, evals > 1e-9]                  # X*'in menzili (Z bunu yok etmeli)
    Pperp = evecs[:, evals <= 1e-9]
    thc = new_theta(); Th = [new_theta() for _ in range(4)]
    w = cp.Variable(N); t = cp.Variable()
    cons = [obj_of(thc) <= T] + [obj_of(Ti) == 0 for Ti in Th]
    verts = [(s, i) for i in range(4) for s in (+1, -1)]
    for s, i in verts:
        mu = flat + s*rho*TAN[i]
        th = tuple(thc[k] + s*rho*Th[i][k] for k in range(5))
        v = D.vvec(mu)
        Z = S0_expr(th) - D.Cmat(mu) - (cp.reshape(w, (N, 1), order='C') @ v.reshape(1, N) + v.reshape(N, 1) @ cp.reshape(w, (1, N), order='C'))/2
        Z = (Z + Z.T)/2
        cons += [th[2] >= 0, th[3] >= 0]
        if impose_kernel:
            Zpk = Pperp.T @ Z @ Kr; Zkk = Kr.T @ Z @ Kr
            cons.append(Zpk == 0)
            cons += [Zkk[i, j] == 0 for i in range(Kr.shape[1]) for j in range(i, Kr.shape[1])]
            Zpp = Pperp.T @ Z @ Pperp
            cons.append((Zpp + Zpp.T)/2 - t*np.eye(Pperp.shape[1]) >> 0)
        else:
            cons.append(Z - t*np.eye(N) >> 0)
    pr = cp.Problem(cp.Maximize(t), cons + [t <= 1])
    pr.solve(solver=solver)
    return pr.status, t.value, Kr.shape[1]

if __name__ == "__main__":
    for rho in [float(r) for r in (sys.argv[1:] or ["0.01", "0.05", "0.1", "0.2"])]:
        st, tv, rk = local_cert(rho)
        print(f"rho={rho}: durum={st}, en buyuk marj t={tv}, rank(X*)={rk}", flush=True)
