# Hakem denetimi: Teorem 11 sertifikalarinin BAGIMSIZ kontrolu (verify.py kullanilmaz).
# Z(y) makaledeki (5) tanimindan kurulur. Kapsama: ic kisim ayrikligi + alanla kesisimin tam hacmi.
import json, sys, math, time
from fractions import Fraction as F
from itertools import product
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from k3 import Q3, ZERO as QZ, ONE as QONE

D = sys.argv[1]
N = 13
iA = lambda x, a: 1 + 3*x + a
iB = lambda y, b: 7 + 3*y + b
CLS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
PAIRS_OFF = [(i, j) for i in range(1, N) for j in range(i+1, N)]
T_VECS = []
for y in range(2):
    t = [0]*N; t[0] = -1
    for b in range(3): t[iB(y, b)] = 1
    T_VECS.append(t)
t = [0]*N
for a in range(3): t[iA(0, a)] = 1; t[iA(1, a)] = -1
T_VECS.append(t)

def build_Z(y, objective, zero, half):
    Z = [[zero]*N for _ in range(N)]
    def E(i, j, val):                  # val * E_ij, E_ij = (e_i e_j^T + e_j e_i^T)/2
        if i == j: Z[i][i] = Z[i][i] + val
        else:
            Z[i][j] = Z[i][j] + val*half; Z[j][i] = Z[j][i] + val*half
    E(0, 0, y[0])
    for k in range(6):
        E(1+k, 0, y[1+k]); E(1+k, 1+k, y[7+k])
    for k, (i, j) in enumerate(PAIRS_OFF):
        E(i, j, -y[13+k])
    for jt, tv in enumerate(T_VECS):
        w = y[79+13*jt: 92+13*jt]
        for i in range(N):
            for j in range(N):
                if tv[j]: Z[i][j] = Z[i][j] - w[i]*tv[j]*half
                if tv[i]: Z[i][j] = Z[i][j] - w[j]*tv[i]*half
    if objective:
        for x, yy, g, l in CLS:
            for a in range(3):
                E(iA(x, a), iB(yy, (a-g) % 3), -1)      # -C
                E(iA(x, a), iB(yy, (a-l) % 3), +1)
    return Z

def ldl_pd(M, sign=lambda v: (v > 0) - (v < 0)):
    n = len(M); A = [r[:] for r in M]
    for k in range(n):
        if sign(A[k][k]) <= 0: return False
        for i in range(k+1, n):
            f = A[i][k]/A[k][k]
            for j in range(k, n): A[i][j] = A[i][j] - f*A[k][j]
    return True

def lt_T(f):                               # f < 2 + 2/sqrt3  (tam)
    return f <= 2 or 3*(f - 2)**2 < 4

def area(a, b, c, d):                      # [a,b]x[c,d] ile {x,y>=0, x+y<=1}'in kesisim alani
    pts = sorted(set([a, b] + [x for x in (1 - d, 1 - c) if a < x < b]))
    s = F(0)
    for x0, x1 in zip(pts, pts[1:]):
        t0, t1 = min(d, 1 - x0), min(d, 1 - x1)
        s += (x1 - x0)*((max(t0, c) - c) + (max(t1, c) - c))/2
    return s

def verts(lo, hi):
    for p in product(*zip(lo, hi)):
        yield (p[0], p[1], 1 - p[0] - p[1], p[2], p[3], 1 - p[2] - p[3])

def step3():
    t0 = time.time()
    bx = json.load(open(D + "/boxes.json"))
    outer = [(tuple(map(F, lo)), tuple(map(F, hi)), list(map(F, y))) for lo, hi, y in bx["outer"]]
    inner = [(tuple(map(F, lo)), tuple(map(F, hi))) for lo, hi in bx["inner"]]
    R2 = F(1, 125)**2
    worst = F(0)
    for lo, hi, y in outer:
        assert len(y) == 118
        assert all(v >= 0 for v in y[7:79]), "isaret"
        assert ldl_pd(build_Z(y, True, F(0), F(1, 2))), "Z > 0 degil"
        for mu in verts(lo, hi):
            f = y[0] + sum(y[1+i]*m + 3*y[7+i]*m*m for i, m in enumerate(mu))
            assert lt_T(f), "kose degeri T'yi asiyor"
            worst = max(worst, f)
    for lo, hi in inner:
        assert all(sum((m - F(1, 3))**2 for m in mu) <= R2 for mu in verts(lo, hi)), "ic kutu topun disinda"
    boxes = [(lo, hi) for lo, hi, _ in outer] + inner
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            (l1, h1), (l2, h2) = boxes[i], boxes[j]
            assert not all(max(l1[k], l2[k]) < min(h1[k], h2[k]) for k in range(4)), "ic kisimlar cakisiyor"
    assert all(0 <= l[k] < h[k] <= 1 for l, h in boxes for k in range(4))
    vol = sum(area(l[0], h[0], l[1], h[1])*area(l[2], h[2], l[3], h[3]) for l, h in boxes)
    print(f"[Adim 3] {len(outer)} dis kutu: isaretler, Z>0 (tam LDL), 16 kosede f<T (tam). En buyuk kose degeri {float(worst):.10f} < T={2+2/math.sqrt(3):.10f}")
    print(f"[Adim 3] {len(inner)} ic kutu ||mu-mu_bar||<=1/125 icinde. {len(boxes)} kutunun ic kisimlari ayrik; alanla kesisim hacmi = {vol} (alanin hacmi 1/4)  [{time.time()-t0:.1f}s]")
    assert vol == F(1, 4)

def q(pair): return Q3(F(pair[0]), F(pair[1]))

def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum((A[i][k]*B[k][j] for k in range(m) if not A[i][k].iszero() and not B[k][j].iszero()), QZ) for j in range(p)] for i in range(n)]

def rank_q3(M):
    A = [r[:] for r in M]; m, n = len(A), len(A[0]); r = 0
    for col in range(n):
        p = next((i for i in range(r, m) if not A[i][col].iszero()), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]
        for i in range(r+1, m):
            if not A[i][col].iszero():
                f = A[i][col]/A[r][col]; A[i] = [A[i][j] - f*A[r][j] for j in range(n)]
        r += 1
    return r

def pd_margin(B, m):
    """B - m I > 0: rasyonel yuvarlama B', tam isaret testiyle ||B-B'||_F <= tau, B' - (m+tau) I icin tam LDL."""
    n = len(B)
    assert all(B[i][j] == B[j][i] for i in range(n) for j in range(n))
    Bq = [[Q3(F(float(B[min(i, j)][max(i, j)])).limit_denominator(10**8)) for j in range(n)] for i in range(n)]
    D2 = sum(((B[i][j] - Bq[i][j])*(B[i][j] - Bq[i][j]) for i in range(n) for j in range(n)), QZ)
    tau = F(1, 10**6)
    while (Q3(tau*tau) - D2).sign() < 0: tau *= 2
    return ldl_pd([[Bq[i][j] - (Q3(m + tau) if i == j else QZ) for j in range(n)] for i in range(n)], sign=lambda v: v.sign()), tau

def sqrt_ub(val):
    assert val.sign() >= 0
    u = F(math.ceil(math.sqrt(max(0.0, float(val)))*10**8) + 1, 10**8)
    while (Q3(u*u) - val).sign() < 0: u += F(1, 10**8)
    return u

def step4():
    t0 = time.time()
    d = json.load(open(D + "/local.json"))
    G = [[q(e) for e in r] for r in d["basis"]]
    y0 = [q(e) for e in d["constant"]]
    Y1 = [[q(e) for e in r] for r in d["linear"]]
    keys = [(i, j) for i in range(4) for j in range(i, 4)]
    Y2 = {k: [Q3(F(e)) for e in r] for k, r in zip(keys, d["quadratic"])}
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    assert rank_q3(G) == 13, "G tersinir degil"
    Gt = [list(r) for r in zip(*G)]
    hat = lambda Z: matmul(matmul(Gt, Z), G)
    Z0 = hat(build_Z(y0, True, QZ, Q3(F(1, 2))))
    Z1 = [hat(build_Z(Y1[i], False, QZ, Q3(F(1, 2)))) for i in range(4)]
    Z2 = {k: hat(build_Z(Y2[k], False, QZ, Q3(F(1, 2)))) for k in keys}
    z2 = lambda i, j: Z2[(min(i, j), max(i, j))]
    assert all(Z0[i][j].iszero() for i in range(N) for j in range(N) if i >= 8 or j >= 8), "Z0'in pk/kk bloklari sifir degil"
    assert all(Z1[i][p][r].iszero() for i in range(4) for p in range(8, N) for r in range(8, N)), "Z1_kk sifir degil"
    big = [Z0[p][:8] + [Z1[i][p][8+r] for i in range(4) for r in range(5)] for p in range(8)]
    big += [[Z1[i][p][8+r] for p in range(8)] + [z2(i, j)[8+r][8+s] for j in range(4) for s in range(5)] for i in range(4) for r in range(5)]
    ok, tau = pd_margin(big, F(7, 50)); assert ok, "28x28 marj 7/50 dogrulanamadi"
    tau_vec = [(1,-1,0,0,0,0), (1,1,-2,0,0,0), (0,0,0,1,-1,0), (0,0,0,1,1,-2)]
    b0 = [QONE] + [Q3(F(1, 3))]*12 + [QZ]*105
    B1 = [[QZ]*118 for _ in range(4)]; B2 = {}
    for i in range(4):
        for a in range(6): B1[i][1+a] = Q3(tau_vec[i][a]); B1[i][7+a] = Q3(2*tau_vec[i][a])
    for i in range(4):
        for j in range(4):
            v = [QZ]*118
            for a in range(6): v[7+a] = Q3(3*tau_vec[i][a]*tau_vec[j][a])
            B2[(i, j)] = v
    dot = lambda u, v: sum((s*t for s, t in zip(u, v) if not s.iszero() and not t.iszero()), QZ)
    assert dot(b0, y0) == Q3(2, F(2, 3)), "f(0) != T"
    assert all((dot(b0, Y1[i]) + dot(B1[i], y0)).iszero() for i in range(4)), "dogrusal terim sifir degil"
    Q = [[dot(b0, y2(i, j)) + (dot(B1[i], Y1[j]) + dot(B1[j], Y1[i]))*Q3(F(1, 2)) + dot(B2[(i, j)], y0) for j in range(4)] for i in range(4)]
    ok, _ = pd_margin([[-Q[i][j] for j in range(4)] for i in range(4)], F(7, 25)); assert ok, "-Q >= 7/25 dogrulanamadi"
    ss = lambda it: sum((v*v for v in it), QZ)
    e1 = sqrt_ub(ss(Z1[i][p][r] for i in range(4) for p in range(8) for r in range(8)))
    e2 = sqrt_ub(ss(z2(i, j)[p][r] for i in range(4) for j in range(4) for p in range(8) for r in range(8)))
    b2 = sqrt_ub(ss(z2(i, j)[p][8+r] for i in range(4) for j in range(4) for p in range(8) for r in range(5)))
    r3 = sqrt_ub(ss(dot(B1[i], y2(j, k)) + dot(B2[(i, j)], Y1[k]) for i in range(4) for j in range(4) for k in range(4)))
    r4 = sqrt_ub(ss(dot(B2[(i, j)], y2(k, l)) for i in range(4) for j in range(4) for k in range(4) for l in range(4)))
    R = F(1, 125)
    print(f"[Adim 4] kendi ust sinirlarim: e1<={float(e1):.4f} e2<={float(e2):.4f} b2<={float(b2):.5f} r3<={float(r3):.3f} r4<={float(r4):.3f} (tau={float(tau):.1e})")
    assert R*(e1 + b2) + R*R*e2 < F(7, 50), "olcekli matris kosulu"
    assert R*r3 + R*R*r4 < F(7, 25), "amac kalan terim kosulu"
    n_in, n_act = 0, 0
    for k in range(7, 79):
        if y0[k].iszero():
            n_in += 1
            assert all(Y1[i][k].iszero() for i in range(4)), "inaktif carpanda dogrusal terim"
            assert ldl_pd([[y2(i, j)[k] for j in range(4)] for i in range(4)], sign=lambda v: v.sign()), "inaktif Y2 blogu PD degil"
        else:
            n_act += 1
            n1 = sqrt_ub(ss(Y1[i][k] for i in range(4))); n2 = sqrt_ub(ss(y2(i, j)[k] for i in range(4) for j in range(4)))
            assert (y0[k] - Q3(R*n1 + R*R*n2)).sign() > 0, "aktif carpan isareti"
    assert [sum(s*t for s, t in zip(tau_vec[i], tau_vec[j])) for i in range(4) for j in range(4)] == [2,0,0,0, 0,6,0,0, 0,0,2,0, 0,0,0,6]
    print(f"[Adim 4] G tersinir (rank 13); Z0 ve Z1 cekirdek bloklari sifir; 28x28 >= 7/50 I; -Q >= 7/25 I; f(0)=T; dogrusal terim 0;"
          f" {n_in} inaktif + {n_act} aktif isaret kisiti tamam. R=1/125 icin iki kalan-terim esitsizligi KESIN. [{time.time()-t0:.1f}s]")

if __name__ == "__main__":
    step3(); step4()
    print("SONUC: Adim 3 ve Adim 4 kendi bagimsiz denetleyicimle dogrulandi.")
