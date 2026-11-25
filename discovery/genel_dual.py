# Psi(mu) icin ACIK dual:  min  y00 + sum_i yA_i + 3 sum_i zA_i
#   S0 = y00 E00 + sum_i yA_i sym(E_{i0}) + sum_i zA_i E_ii - sum_{i<j} u_ij sym(E_ij) - sum_y sym(w_y (1_By - e0)^T)
#   Z(mu) = S0 - C(mu) - sym(w v(mu)^T) >= 0 ,  z,u >= 0
# Dual amaci mu'dan BAGIMSIZ. Sabit (y,z,u,w_y) icin kosul:  S0 - C(mu)  matrisi  v(mu)^perp  uzerinde PSD (Finsler).
import numpy as np
import cvxpy as cp
from genel_alice import iA, iB, E0, PAIRS

N = 13
def sym(a, b):
    return (np.outer(a, b) + np.outer(b, a)) / 2
def E(i, j):
    e = np.zeros((N, N)); e[i, j] = 1; return (e + e.T) / 2 if i != j else e

def Cmat(mu):
    C = np.zeros((N, N))
    for x, y, g, l in PAIRS:
        for a in range(3):
            C += mu[x][a] * (E(iA(x, a), iB(y, (a-g) % 3)) - E(iA(x, a), iB(y, (a-l) % 3)))
    return C
def vvec(mu):
    v = np.zeros(N)
    for a in range(3):
        v[iA(0, a)] = mu[0][a]; v[iA(1, a)] = -mu[1][a]
    return v
def bvec(y):
    v = np.zeros(N); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1; return v

ALICE = [iA(x, a) for x in range(2) for a in range(3)]
OFF = [(i, j) for i in range(1, N) for j in range(i+1, N)]

def solve_dual(mu):
    y00 = cp.Variable(); yA = cp.Variable(6); zA = cp.Variable(6, nonneg=True)
    u = cp.Variable(len(OFF), nonneg=True); wy = cp.Variable((2, N)); w = cp.Variable(N)
    S0 = y00 * E(0, 0)
    for k, i in enumerate(ALICE):
        S0 = S0 + yA[k] * E(i, 0) + zA[k] * E(i, i)
    for k, (i, j) in enumerate(OFF):
        S0 = S0 - u[k] * E(i, j)
    for y in range(2):
        b = bvec(y)
        S0 = S0 - (cp.reshape(wy[y], (N, 1), order='C') @ b.reshape(1, N) + b.reshape(N, 1) @ cp.reshape(wy[y], (1, N), order='C')) / 2
    v = vvec(mu)
    Z = S0 - Cmat(mu) - (cp.reshape(w, (N, 1), order='C') @ v.reshape(1, N) + v.reshape(N, 1) @ cp.reshape(w, (1, N), order='C')) / 2
    pr = cp.Problem(cp.Minimize(y00 + cp.sum(yA) + 3*cp.sum(zA)), [(Z + Z.T)/2 >> 0])
    pr.solve(solver=cp.CLARABEL)
    S0v = y00.value*E(0, 0)
    for k, i in enumerate(ALICE):
        S0v = S0v + yA.value[k]*E(i, 0) + zA.value[k]*E(i, i)
    for k, (i, j) in enumerate(OFF):
        S0v = S0v - u.value[k]*E(i, j)
    for y in range(2):
        S0v = S0v - sym(wy.value[y], bvec(y))
    return pr.value, S0v

def margin(S0, mu):
    """S0 - C(mu) matrisinin v(mu)^perp uzerindeki en kucuk oz degeri."""
    v = vvec(mu); M = S0 - Cmat(mu)
    Q, _ = np.linalg.qr(np.column_stack([v, np.eye(N)]))
    P = Q[:, 1:N]                      # v'ye dik ortonormal taban
    return np.linalg.eigvalsh(P.T @ M @ P).min()

if __name__ == "__main__":
    flat = np.full((2, 3), 1/3)
    val, S0 = solve_dual(flat)
    print("duz noktada dual deger:", val, "  2+2/sqrt3 =", 2 + 2/np.sqrt(3))
    print("duz noktada marj:", margin(S0, flat))
    rng = np.random.default_rng(0)
    for r in (0.01, 0.03, 0.1, 0.3, 1.0):
        ms = []
        for t in range(300):
            mu = np.clip(flat + r*rng.standard_normal((2, 3))/3, 1e-6, None); mu /= mu.sum(1, keepdims=True)
            ms.append(margin(S0, mu))
        print(f"yaricap ~{r}: min marj = {min(ms):+.3e}   (>=0 ise duz dual o bolgede de kanit)")
    ms = [margin(S0, rng.dirichlet(np.ones(3), size=2)) for _ in range(2000)]
    print("tum simpleks (2000 rastgele): min marj =", min(ms), "  negatif oran:", np.mean(np.array(ms) < -1e-9))
    np.save("duz_dual_S0.npy", S0)
