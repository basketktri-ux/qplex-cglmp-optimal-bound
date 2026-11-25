# Duz nokta etrafinda IKINCI MERTEBE yerel sertifika (Alice-homojen dual, uygun kume mu'dan bagimsiz).
#   y(delta) = y0 + Y1 delta + Y2(delta,delta),   Z(y) = A(y) - C   (A dogrusal),   f(delta) = b(mu(delta))^T y(delta)
# Kosullar (delta^2 mertebesine kadar):
#   (a) b0^T Y1 = 0                          (amacin 1. mertebesi yok)
#   (b) Kr^T A(Y1 e_i) Kr = 0                (Z0'in cekirdeginde 1. mertebe yok)
#   (c) [[Z0p, W],[W^T, K - eta I]] >= 0      (Schur 2. mertebe; W_i = P^T A(Y1 e_i) Kr, K_ij = Kr^T A(Y2_ij) Kr)
#   (d) Q + c I <= 0                          (amacin 2. mertebesi; Q_ij = b0^T Y2_ij + sym(B1_i^T Y1_j) + B2_ij^T y0)
#   (e) y0'da sifir olan isaret-kisitli bilesenler icin Y1=0, Y2_k >= 0
import numpy as np
import cvxpy as cp
from genel_yerel import flat_primal, TAN

T = 2 + 2/np.sqrt(3); N = 13; E0 = 0
PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
iA = lambda x, a: 1 + 3*x + a
iB = lambda y, b: 7 + 3*y + b
ALICE = [iA(x, a) for x in range(2) for a in range(3)]
OFF = [(i, j) for i in range(1, N) for j in range(i+1, N)]

def E(i, j):
    e = np.zeros((N, N)); e[i, j] = 1
    return (e + e.T)/2 if i != j else e
def S(a, b):
    return (np.outer(a, b) + np.outer(b, a))/2

C = np.zeros((N, N))
for x, y, g, l in PAIRS:
    for a in range(3):
        C += E(iA(x, a), iB(y, (a-g) % 3)) - E(iA(x, a), iB(y, (a-l) % 3))
BV = []
for y in range(2):
    v = np.zeros(N); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1; BV.append(v)
CV = np.zeros(N); CV[[iA(0, a) for a in range(3)]] = 1; CV[[iA(1, a) for a in range(3)]] = -1

# dual vektor y icin taban matrisleri A_k ve isaret kisitli indeksler
Amats, names, signed = [], [], []
Amats.append(E(0, 0)); names.append("y00")
for k, i in enumerate(ALICE):
    Amats.append(E(i, 0)); names.append(f"yA{k}")
for k, i in enumerate(ALICE):
    Amats.append(E(i, i)); names.append(f"zA{k}"); signed.append(len(Amats)-1)
for (i, j) in OFF:
    Amats.append(-E(i, j)); names.append(f"u{i},{j}"); signed.append(len(Amats)-1)
for y in range(2):
    for m in range(N):
        e = np.zeros(N); e[m] = 1; Amats.append(-S(e, BV[y])); names.append(f"wy{y},{m}")
for m in range(N):
    e = np.zeros(N); e[m] = 1; Amats.append(-S(e, CV)); names.append(f"w{m}")
Amats = np.array(Amats); NY = len(Amats)
IY00 = 0; IYA = list(range(1, 7)); IZA = list(range(7, 13))

def Aof(y):
    return np.tensordot(y, Amats, axes=1)
def Aexpr(yv):
    return sum(yv[k]*Amats[k] for k in range(NY))

def bvec(mu6):
    b = np.zeros(NY); b[IY00] = 1; b[IYA] = mu6; b[IZA] = 3*mu6**2; return b

MU0 = np.full(6, 1/3)
TANV = [t.reshape(6) for t in TAN]                     # 4 teget yon (R^6)
B1 = [np.zeros(NY) for _ in range(4)]
for i in range(4):
    B1[i][IYA] = TANV[i]; B1[i][IZA] = 6*MU0*TANV[i]
B2 = [[np.zeros(NY) for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        B2[i][j][IZA] = 3*TANV[i]*TANV[j]

def primal_flat():
    X = flat_primal().copy()
    for i in ALICE:                                   # a~ = A^/3
        X[i, :] /= 3; X[:, i] /= 3
    return X

def flat_dual():
    Xs = primal_flat()
    ev, V = np.linalg.eigh(Xs); Kr = V[:, ev > 1e-9]; P = V[:, ev <= 1e-9]
    yv = cp.Variable(NY); Z = Aexpr(yv) - C
    pr = cp.Problem(cp.Minimize(bvec(MU0) @ yv), [(Z + Z.T)/2 >> 0, yv[signed] >= 0]); pr.solve(solver=cp.CLARABEL)
    y0 = yv.value.copy()
    active = [k for k in signed if y0[k] > 1e-6]; inactive = [k for k in signed if k not in active]
    y0[inactive] = 0
    Z0 = Aof(y0) - C
    t = np.linalg.eigvalsh(P.T @ Z0 @ P).min()
    return y0, Kr, P, active, inactive, t, pr.status

def local_sdp(y0, Kr, P, active, inactive):
    Z0 = Aof(y0) - C; Z0p = P.T @ Z0 @ P
    Y1 = cp.Variable((NY, 4)); Y2 = {(i, j): cp.Variable(NY) for i in range(4) for j in range(i, 4)}
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    eta = cp.Variable(); c = cp.Variable()
    b0 = bvec(MU0)
    cons = [b0 @ Y1[:, i] == 0 for i in range(4)]
    for i in range(4):
        Ki = Kr.T @ Aexpr(Y1[:, i]) @ Kr
        cons += [Ki[a, b] == 0 for a in range(Kr.shape[1]) for b in range(a, Kr.shape[1])]
    W = cp.hstack([P.T @ Aexpr(Y1[:, i]) @ Kr for i in range(4)])                      # 8 x 20
    K = cp.bmat([[Kr.T @ Aexpr(y2(i, j)) @ Kr for j in range(4)] for i in range(4)])     # 20 x 20
    kk = 4*Kr.shape[1]
    big = cp.bmat([[Z0p, W], [W.T, (K + K.T)/2 - eta*np.eye(kk)]])
    cons.append((big + big.T)/2 >> 0)
    Q = cp.bmat([[cp.reshape(b0 @ y2(i, j) + (B1[i] @ Y1[:, j] + B1[j] @ Y1[:, i])/2 + B2[i][j] @ y0, (1, 1), order='C')
                  for j in range(4)] for i in range(4)])
    cons.append((Q + Q.T)/2 + c*np.eye(4) << 0)
    for k in inactive:                                 # y0_k = 0 olan isaret-kisitli bilesenler
        cons.append(Y1[k, :] == 0)
        Mk = cp.bmat([[cp.reshape(y2(i, j)[k], (1, 1), order='C') for j in range(4)] for i in range(4)])
        cons.append((Mk + Mk.T)/2 >> 0)
    s = cp.Variable()
    cons += [s <= eta, s <= c, s <= 1, cp.norm(Y1, 'fro') <= 1e3]
    pr = cp.Problem(cp.Maximize(s), cons); pr.solve(solver=cp.CLARABEL)
    return pr.status, s.value, eta.value, c.value, Y1.value, {k: v.value for k, v in Y2.items()}

if __name__ == "__main__":
    y0, Kr, P, active, inactive, t0, st = flat_dual()
    print("duz dual:", st, " tumleyen marji / aktif carpan alt siniri t =", t0, " aktif:", len(active), " inaktif:", len(inactive))
    print("  amac b0.y0 =", bvec(MU0) @ y0, " (T =", T, ")")
    g = [B1[i] @ y0 for i in range(4)]
    print("  y0'in mu-gradyani (sifir olmali):", np.round(g, 8))
    st, s, eta, c, Y1, Y2 = local_sdp(y0, Kr, P, active, inactive)
    print("yerel 2. mertebe SDP:", st, " s =", s, " eta =", eta, " c =", c)

def curve(y0, Y1, Y2, d):
    y = y0 + Y1 @ d
    for (i, j), v in Y2.items():
        y = y + (d[i]*d[j]*(1 if i == j else 2))*v
    return y

def check_radius(y0, Y1, Y2, rho, n=3000, seed=0):
    rng = np.random.default_rng(seed); worst = [np.inf, -np.inf, np.inf]
    for _ in range(n):
        d = rng.standard_normal(4); d *= rho/np.linalg.norm(d)
        mu = MU0 + sum(d[i]*TANV[i] for i in range(4))
        y = curve(y0, Y1, Y2, d)
        worst[0] = min(worst[0], np.linalg.eigvalsh(Aof(y) - C).min())
        worst[1] = max(worst[1], bvec(mu) @ y - T)
        worst[2] = min(worst[2], y[signed].min())
    return worst
