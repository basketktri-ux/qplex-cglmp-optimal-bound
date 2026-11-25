# Dis bolge icin dal-sinir (branch-and-bound), Alice-homojen dual ile.
# Primal (sabit mu):  max sum +-X[a~,b~]  s.t. X>=0, X00=1, X[a~,e0]=mu_a, X[a~,a~]<=3mu_a^2, X_ij>=0,
#                     Bob: X(1_By - e0)=0,  Alice: X(1_A0 - 1_A1)=0
# Dual: Z(y) = y00 E00 + sum yA sym(E_a0) + sum zA E_aa - sum u sym(E_ij) - sum sym(w_y b_y^T) - sym(w c^T) - C >= 0
#       SINIR: f_y(mu) = y00 + yA.mu + 3 sum zA mu^2   (zA>=0 => mu'de konveks)  => kutuda maks KOSELERDE.
# Kutu: p=(mu0_0, mu0_1, mu1_0, mu1_1), mu^x_2 = 1 - mu^x_0 - mu^x_1.
import sys, time, pickle, itertools
import numpy as np
import cvxpy as cp

T = 2 + 2/np.sqrt(3)
N = 13; E0 = 0
PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
iA = lambda x, a: 1 + 3*x + a
iB = lambda y, b: 7 + 3*y + b
ALICE = [iA(x, a) for x in range(2) for a in range(3)]
OFF = [(i, j) for i in range(1, N) for j in range(i+1, N)]

def E(i, j):
    e = np.zeros((N, N)); e[i, j] = 1
    return (e + e.T)/2 if i != j else e
def symvec(w, b):
    return (cp.reshape(w, (N, 1), order='C') @ b.reshape(1, N) + b.reshape(N, 1) @ cp.reshape(w, (1, N), order='C'))/2

CMAT = np.zeros((N, N))
for x, y, g, l in PAIRS:
    for a in range(3):
        CMAT += E(iA(x, a), iB(y, (a-g) % 3)) - E(iA(x, a), iB(y, (a-l) % 3))
BV = []
for y in range(2):
    v = np.zeros(N); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1; BV.append(v)
CV = np.zeros(N); CV[[iA(0, a) for a in range(3)]] = 1; CV[[iA(1, a) for a in range(3)]] = -1

def build_problem(nvert=16):
    y00 = cp.Variable(); yA = cp.Variable(6); zA = cp.Variable(6, nonneg=True)
    u = cp.Variable(len(OFF), nonneg=True); wy = cp.Variable((2, N)); w = cp.Variable(N); t = cp.Variable()
    Z = y00*E(0, 0) - CMAT
    for k, i in enumerate(ALICE):
        Z = Z + yA[k]*E(i, 0) + zA[k]*E(i, i)
    Uoff = np.zeros((len(OFF), N, N))
    Z = Z - sum(u[k]*E(i, j) for k, (i, j) in enumerate(OFF))
    Z = Z - symvec(wy[0], BV[0]) - symvec(wy[1], BV[1]) - symvec(w, CV)
    Pmu = cp.Parameter((nvert, 6)); Pmu2 = cp.Parameter((nvert, 6))
    cons = [(Z + Z.T)/2 >> 0, y00 + Pmu @ yA + 3*(Pmu2 @ zA) <= t]
    pr = cp.Problem(cp.Minimize(t), cons)
    return pr, Pmu, Pmu2, (y00, yA, zA, u, wy, w)

def mu_of_p(p):
    return np.array([p[0], p[1], 1 - p[0] - p[1], p[2], p[3], 1 - p[2] - p[3]])

def box_vertices(lo, hi):
    return [mu_of_p([lo[i] if bit == 0 else hi[i] for i, bit in enumerate(bits)]) for bits in itertools.product((0, 1), repeat=4)]

PFLAT = np.full(4, 1/3)

def bb(d0=0.05, max_boxes=200000, tol=1e-7, start_div=4, log_every=500, euclid=None):
    pr, Pmu, Pmu2, vars_ = build_problem()
    step = 1/start_div
    queue = []
    for idx in itertools.product(range(start_div), repeat=4):
        lo = np.array(idx)*step; queue.append((lo, lo + step))
    certified, local_boxes, n_solved = [], [], 0
    t0 = time.time()
    while queue and n_solved < max_boxes:
        lo, hi = queue.pop()
        if lo[0] + lo[1] >= 1 or lo[2] + lo[3] >= 1:      # kutu tamamen alanin disinda (mu_2 <= 0 sinirinda bile)
            continue
        if euclid is not None:
            inner = max(np.linalg.norm(v - 1/3) for v in box_vertices(lo, hi)) <= euclid   # mu-uzayinda Oklid topu
        else:
            inner = np.all(np.maximum(np.abs(lo - PFLAT), np.abs(hi - PFLAT)) <= d0)
        if inner:
            local_boxes.append((lo, hi)); continue         # ic top: yerel argumana birakilir
        V = np.array(box_vertices(lo, hi))
        Pmu.value = V; Pmu2.value = V**2
        try:
            pr.solve(solver=cp.CLARABEL)
            val = pr.value
        except Exception:
            val = np.inf
        n_solved += 1
        if val is not None and val <= T - tol:
            certified.append((lo, hi, val, [v.value.copy() for v in vars_]))
        else:
            k = int(np.argmax(hi - lo)); mid = (lo[k] + hi[k])/2
            h1 = hi.copy(); h1[k] = mid; l2 = lo.copy(); l2[k] = mid
            queue += [(lo, h1), (l2, hi)]
        if n_solved % log_every == 0:
            print(f"  cozulen={n_solved} sertifikali={len(certified)} kuyruk={len(queue)} ic={len(local_boxes)} "
                  f"sure={time.time()-t0:.0f}s", flush=True)
    return certified, local_boxes, queue, n_solved, time.time() - t0

if __name__ == "__main__":
    d0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    maxb = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    eu = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cert, loc, q, ns, dt = bb(d0=d0, max_boxes=maxb, euclid=eu)
    print(f"BITTI d0={d0}: cozulen={ns}, sertifikali kutu={len(cert)}, ic kutu={len(loc)}, acik kuyruk={len(q)}, sure={dt:.0f}s")
    with open(f"bb_d0_{d0}.pkl", "wb") as f:
        pickle.dump({"cert": cert, "local": loc, "queue": q}, f)
