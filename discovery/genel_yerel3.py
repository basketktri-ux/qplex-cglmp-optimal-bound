# Olceklenmis yerel sertifika:  M(rho,d) = [[Z0p + rho Z1pp(d) + rho^2 Z2pp(d), Z1pk(d) + rho Z2pk(d)],[., Z2kk(d)]]
# rho=0'da marj m0 tek LMI ile:  [[Z0p - m0 I, W~],[W~^T, K~ - m0 I]] >= 0   (|d|=1)
# Yeterli kosul:  m0 > rho (e1 + b2) + rho^2 e2 ;  amac: -c rho^2 + r3 rho^3 + r4 rho^4 <= 0 ; isaretli bilesenler.
import numpy as np, math, cvxpy as cp
import genel_yerel2 as L
TAU = [np.array([1,-1,0,0,0,0.]), np.array([1,1,-2,0,0,0.]), np.array([0,0,0,1,-1,0.]), np.array([0,0,0,1,1,-2.])]
L.TANV = TAU
L.B1 = [np.zeros(L.NY) for _ in range(4)]
for i in range(4): L.B1[i][L.IYA] = TAU[i]; L.B1[i][L.IZA] = 2*TAU[i]
L.B2 = [[np.zeros(L.NY) for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4): L.B2[i][j][L.IZA] = 3*TAU[i]*TAU[j]
KEYS = [(i, j) for i in range(4) for j in range(i, 4)]

def solve_curve(y0, Kr, P, act, inact, m_t, c_t, w=(1, 1, 0.05, 0.02, 0.02)):
    G = np.hstack([P, Kr]); Z0 = L.Aof(y0) - L.C; Z0p = (G.T @ Z0 @ G)[:8, :8]
    Y1 = cp.Variable((L.NY, 4)); Y2 = {k: cp.Variable(L.NY) for k in KEYS}
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    b0 = L.bvec(L.MU0); cons = [b0 @ Y1[:, i] == 0 for i in range(4)]
    Zh1 = [G.T @ L.Aexpr(Y1[:, i]) @ G for i in range(4)]
    Zh2 = {(i, j): G.T @ L.Aexpr(y2(i, j)) @ G for i in range(4) for j in range(4)}
    for i in range(4):
        cons += [Zh1[i][p, q] == 0 for p in range(8, 13) for q in range(p, 13)]
    W = cp.hstack([Zh1[i][:8, 8:] for i in range(4)])
    K = cp.bmat([[Zh2[(i, j)][8:, 8:] for j in range(4)] for i in range(4)])
    big = cp.bmat([[Z0p - m_t*np.eye(8), W], [W.T, (K + K.T)/2 - m_t*np.eye(20)]])
    cons.append((big + big.T)/2 >> 0)
    Q = cp.bmat([[cp.reshape(b0 @ y2(i, j) + (L.B1[i] @ Y1[:, j] + L.B1[j] @ Y1[:, i])/2 + L.B2[i][j] @ y0, (1, 1), order='C')
                  for j in range(4)] for i in range(4)])
    cons.append((Q + Q.T)/2 + c_t*np.eye(4) << 0)
    for k in inact:
        cons.append(Y1[k, :] == 0)
        Mk = cp.bmat([[cp.reshape(y2(i, j)[k], (1, 1), order='C') for j in range(4)] for i in range(4)])
        cons.append((Mk + Mk.T)/2 >> 0)
    e1 = cp.norm(cp.hstack([cp.vec(Zh1[i][:8, :8], order='C') for i in range(4)]))
    b2 = cp.norm(cp.hstack([cp.vec(Zh2[k][:8, 8:], order='C') for k in Zh2]))
    e2 = cp.norm(cp.hstack([cp.vec(Zh2[k][:8, :8], order='C') for k in Zh2]))
    r3 = cp.norm(cp.hstack([L.B1[i] @ y2(j, l) + L.B2[i][j] @ Y1[:, l] for i in range(4) for j in range(4) for l in range(4)]))
    sg = cp.norm(Y1[act, :], 'fro')
    pr = cp.Problem(cp.Minimize(w[0]*e1 + w[1]*b2 + w[2]*e2 + w[3]*r3 + w[4]*sg), cons)
    pr.solve(solver=cp.CLARABEL)
    return pr.status, Y1.value, {k: v.value for k, v in Y2.items()}

def radius(y0, Kr, P, act, Y1, Y2, m0=None, cc=None):
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    G = np.hstack([P, Kr]); Z0 = L.Aof(y0) - L.C; Z0p = (G.T @ Z0 @ G)[:8, :8]
    Zh1 = [G.T @ L.Aof(Y1[:, i]) @ G for i in range(4)]
    Zh2 = {(i, j): G.T @ L.Aof(y2(i, j)) @ G for i in range(4) for j in range(4)}
    W = np.hstack([Zh1[i][:8, 8:] for i in range(4)]); K = np.block([[Zh2[(i, j)][8:, 8:] for j in range(4)] for i in range(4)])
    if m0 is None:
        m0 = np.linalg.eigvalsh(np.block([[Z0p, W], [W.T, (K + K.T)/2]])).min()
    fro = lambda M: float(np.sum(M**2))
    e1 = math.sqrt(sum(fro(Zh1[i][:8, :8]) for i in range(4))); e2 = math.sqrt(sum(fro(Zh2[k][:8, :8]) for k in Zh2))
    b2 = math.sqrt(sum(fro(Zh2[k][:8, 8:]) for k in Zh2))
    b0 = L.bvec(L.MU0)
    Q = np.array([[b0 @ y2(i, j) + (L.B1[i] @ Y1[:, j] + L.B1[j] @ Y1[:, i])/2 + L.B2[i][j] @ y0 for j in range(4)] for i in range(4)])
    if cc is None: cc = -np.linalg.eigvalsh(Q).max()
    r3 = math.sqrt(sum((L.B1[i] @ y2(j, l) + L.B2[i][j] @ Y1[:, l])**2 for i in range(4) for j in range(4) for l in range(4)))
    r4 = math.sqrt(sum((L.B2[i][j] @ y2(l, m))**2 for i in range(4) for j in range(4) for l in range(4) for m in range(4)))
    def ok(rho):
        if m0 - rho*(e1 + b2) - rho**2*e2 <= 0: return False
        if -cc + r3*rho + r4*rho**2 > 0: return False
        for k in act:
            n2 = math.sqrt(sum(y2(i, j)[k]**2 for i in range(4) for j in range(4)))
            if y0[k] - np.linalg.norm(Y1[k])*rho - n2*rho**2 < 0: return False
        return True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi)/2; lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, dict(m0=m0, c=cc, e1=e1, e2=e2, b2=b2, r3=r3, r4=r4)

if __name__ == "__main__":
    y0, Kr, P, act, inact, t0, st = L.flat_dual()
    for m_t in (0.02, 0.05, 0.1, 0.2):
        st, Y1, Y2 = solve_curve(y0, Kr, P, act, inact, m_t, 0.1)
        if Y1 is None: print(m_t, st); continue
        rho, info = radius(y0, Kr, P, act, Y1, Y2)
        print(f"m_hedef={m_t}: {st}  rho={rho:.4g}  (mu-yaricapi {rho*math.sqrt(2):.4g})  ", {k: round(v, 4) for k, v in info.items()}, flush=True)
