# Yerel topun TITIZ ispati: duz nokta etrafinda kuadratik dual egrisi, Q(sqrt3)'te tam dogrulama + acik kalan-terim sinirlari.
# Teget taban (rasyonel): tau1=((1,-1,0),0), tau2=((1,1,-2),0), tau3=(0,(1,-1,0)), tau4=(0,(1,1,-2));  mu = duz + sum delta_i tau_i
# ||mu - duz||^2 = 2d1^2 + 6d2^2 + 2d3^2 + 6d4^2 >= 2|delta|^2   =>  Oklid mu-topu yaricap r  icinde |delta| <= r/sqrt2.
import sys, math, pickle
from fractions import Fraction as Fr
from itertools import product
import numpy as np
from q3 import Q3, ZERO, ONE, R3, matmul, transpose, ldl_pd, rref_solver
import genel_yerel2 as L

T_EX = Q3(2, Fr(2, 3))                       # 2 + 2/sqrt3
TAU = [np.array([1, -1, 0, 0, 0, 0.]), np.array([1, 1, -2, 0, 0, 0.]),
       np.array([0, 0, 0, 1, -1, 0.]), np.array([0, 0, 0, 1, 1, -2.])]
TAU_EX = [[Q3(int(v)) for v in t] for t in TAU]

# ---------- 1) tam SIC vektorleri ve X* ----------
def cmul(p, q): return (p[0]*q[0] - p[1]*q[1], p[0]*q[1] + p[1]*q[0])
def cadd(p, q): return (p[0] + q[0], p[1] + q[1])
def wpow(m):
    m %= 3
    return [(ONE, ZERO), (Q3(Fr(-1, 2)), Q3(0, Fr(1, 2))), (Q3(Fr(-1, 2)), Q3(0, Fr(-1, 2)))][m]
h = (Q3(Fr(1, 4), Fr(1, 4)), Q3(Fr(1, 4), Fr(-1, 4)))
k = (Q3(Fr(-1, 2)), Q3(Fr(1, 2)))
hb = (h[0], -h[1])
PH = [(h, k), (hb, h)]
GRID = list(product(range(3), repeat=2))
def u_ex(x, a):
    return [(ONE + wpow(ij[x] - a)[0]) / 9 for ij in GRID]
def v_ex(y, b):
    z0, z1 = PH[y]; out = []
    for i, j in GRID:
        s = cadd(cmul(z0, wpow(i)), cmul(z1, wpow(j)))
        out.append((ONE + cmul(wpow(-b), s)[0]) / 9)
    return out
def vec13():
    V = [[ONE] + [ZERO]*9]
    for x in range(2):
        for a in range(3):
            V.append([Q3(Fr(1, 3))] + [Fr(6, 3)*(p - Q3(Fr(1, 9))) for p in u_ex(x, a)])   # a~ = A^/3 = (1,6(u-c))/3
    for y in range(2):
        for b in range(3):
            V.append([Q3(Fr(1, 3))] + [Fr(6, 3)*(p - Q3(Fr(1, 9))) for p in v_ex(y, b)])   # b~ = (1,6(v-c))/3
    return V
VX = vec13()
XS = matmul(VX, transpose(VX))
CEX = [[Q3(Fr(L.C[i, j]).limit_denominator(4)) for j in range(13)] for i in range(13)]
AMATS = [[[Q3(Fr(L.Amats[kk][i, j]).limit_denominator(4)) for j in range(13)] for i in range(13)] for kk in range(L.NY)]

def nullspace_basis(M):
    """M'nin cekirdegi icin tam taban (sutunlar)."""
    m, n = len(M), len(M[0]); A = [r[:] for r in M]; piv, r = [], 0
    for col in range(n):
        p = next((i for i in range(r, m) if not A[i][col].iszero()), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; pv = A[r][col]; A[r] = [x/pv for x in A[r]]
        for i in range(m):
            if i != r and not A[i][col].iszero():
                f = A[i][col]; A[i] = [A[i][j] - f*A[r][j] for j in range(n)]
        piv.append(col); r += 1
    free = [c for c in range(n) if c not in piv]; basis = []
    for fc in free:
        v = [ZERO]*n; v[fc] = ONE
        for i, pc in enumerate(piv): v[pc] = -A[i][fc]
        basis.append(v)
    return basis, piv

def Aof_ex(y):
    Z = [[ZERO]*13 for _ in range(13)]
    for kk, yk in enumerate(y):
        if yk.iszero(): continue
        Ak = AMATS[kk]
        for i in range(13):
            for j in range(13):
                if not Ak[i][j].iszero(): Z[i][j] = Z[i][j] + yk*Ak[i][j]
    return Z

def frob2(M):
    return sum((x*x for row in M for x in row), ZERO)

def fup(x):                                   # Q3 >= 0 icin guvenli ust float
    return float(x)*(1 + 1e-12) + 1e-300

def run():
    # X* ve tam bazlar
    obj = sum((CEX[i][j]*XS[i][j] for i in range(13) for j in range(13)), ZERO)
    assert obj == T_EX, f"<C,X*> != T : {obj}"
    ker, piv = nullspace_basis(XS)              # X*'in cekirdegi (P, 8 boyut)
    KR = [[XS[i][c] for i in range(13)] for c in piv]    # menzil: X*'in pivot sutunlari (5 boyut)
    P = transpose(ker); Kr = transpose(KR)      # 13x8, 13x5
    print("rank X* =", len(piv), " cekirdek boyutu =", len(ker))
    # sayisal egri (rasyonel teget tabaninda)
    L.TANV = [t.copy() for t in TAU]
    L.B1 = [np.zeros(L.NY) for _ in range(4)]
    for i in range(4):
        L.B1[i][L.IYA] = TAU[i]; L.B1[i][L.IZA] = 6*L.MU0*TAU[i]
    L.B2 = [[np.zeros(L.NY) for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            L.B2[i][j][L.IZA] = 3*TAU[i]*TAU[j]
    y0n, Krn, Pn, act, inact, t0, st = L.flat_dual()
    st, s, eta, c, Y1n, Y2n = L.local_sdp(y0n, Krn, Pn, act, inact)
    print("sayisal yerel SDP:", st, "eta =", eta, "c =", c)
    b0 = [Q3(1)] + [Q3(Fr(1, 3))]*6 + [Q3(Fr(1, 3))]*6 + [ZERO]*(L.NY - 13)       # b(duz)
    def B1ex(i):
        v = [ZERO]*L.NY
        for a in range(6):
            v[1+a] = TAU_EX[i][a]; v[7+a] = Q3(2)*TAU_EX[i][a]
        return v
    def B2ex(i, j):
        v = [ZERO]*L.NY
        for a in range(6): v[7+a] = Q3(3)*TAU_EX[i][a]*TAU_EX[j][a]
        return v
    dot = lambda a, b: sum((x*y for x, y in zip(a, b) if not x.iszero() and not y.iszero()), ZERO)
    # 2) tam y0:  (A(y)-C)Kr = 0, inaktif=0, b0.y = T, B1_i.y = 0
    rows, rhs = [], []
    AK = [matmul(AMATS[kk], Kr) for kk in range(L.NY)]
    CK = matmul(CEX, Kr)
    for i in range(13):
        for jj in range(5):
            rows.append([AK[kk][i][jj] for kk in range(L.NY)]); rhs.append(CK[i][jj])
    for kk in inact:
        r = [ZERO]*L.NY; r[kk] = ONE; rows.append(r); rhs.append(ZERO)
    rows.append(b0); rhs.append(T_EX)
    for i in range(4):
        rows.append(B1ex(i)); rhs.append(ZERO)
    y0, rk = rref_solver(rows, rhs, y0n)
    Z0 = [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(Aof_ex(y0), CEX)]
    assert all(x.iszero() for row in matmul(Z0, Kr) for x in row), "Z0 Kr != 0"
    assert all(y0[kk].sign() > 0 for kk in act), "aktif carpan pozitif degil"
    G = [P[i] + Kr[i] for i in range(13)]       # 13x13 taban [P | Kr]
    GT = transpose(G)
    Zh0 = matmul(matmul(GT, Z0), G)
    Z0p = [r[:8] for r in Zh0[:8]]
    lam_lo = Q3(Fr(1, 10**6))
    lam = Fr(float(np.linalg.eigvalsh(np.array([[float(x) for x in r] for r in Z0p])).min())*0.99).limit_denominator(10**8)
    lam_lo = Q3(lam)
    assert ldl_pd([[Z0p[i][j] - (lam_lo if i == j else ZERO) for j in range(8)] for i in range(8)]), "Z0p - lam I PD degil"
    a_inv = 1/float(lam_lo)
    print("tam y0: rank", rk, " Z0 Kr = 0 OK, Z0p >= lam I, lam =", float(lam_lo))
    # 3) tam Y1:  b0.y = 0, Kr^T A(y) Kr = 0 (ust ucgen), inaktif = 0
    rows1 = [b0[:]]
    KAK = [matmul(transpose(Kr), AK[kk]) for kk in range(L.NY)]
    for i in range(5):
        for jj in range(i, 5):
            rows1.append([KAK[kk][i][jj] for kk in range(L.NY)])
    for kk in inact:
        r = [ZERO]*L.NY; r[kk] = ONE; rows1.append(r)
    Y1 = [rref_solver(rows1, [ZERO]*len(rows1), Y1n[:, i])[0] for i in range(4)]
    # 4) Y2 rasyonel; inaktif bilesenlerde 4x4 matrisler tam PD (gerekirse kucuk kaydirma)
    keys = [(i, j) for i in range(4) for j in range(i, 4)]
    Y2 = {kk: [Q3(Fr(float(x)).limit_denominator(10**12)) for x in Y2n[kk]] for kk in keys}
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    for kk in inact:
        M = [[y2(i, j)[kk] for j in range(4)] for i in range(4)]
        if not ldl_pd(M):
            for i in range(4): Y2[(i, i)][kk] = Y2[(i, i)][kk] + Q3(Fr(1, 10**9))
            M = [[y2(i, j)[kk] for j in range(4)] for i in range(4)]
            assert ldl_pd(M), "inaktif Y2 PD yapilamadi"
    # 5) tam LMI (c): [[Z0p, W],[W^T, K - eta' I]] >= 0
    Zh1 = [matmul(matmul(GT, Aof_ex(Y1[i])), G) for i in range(4)]
    for i in range(4):
        assert all(Zh1[i][p][q].iszero() for p in range(8, 13) for q in range(8, 13)), "Z1_kk != 0"
    Zh2 = {(i, j): matmul(matmul(GT, Aof_ex(y2(i, j))), G) for i in range(4) for j in range(4)}
    W = [[Zh1[i][p][8+q] for i in range(4) for q in range(5)] for p in range(8)]          # 8 x 20
    K = [[Zh2[(i, j)][8+q][8+r] for j in range(4) for r in range(5)] for i in range(4) for q in range(5)]
    big = [Z0p[p] + W[p] for p in range(8)] + [[W[p][m] for p in range(8)] + K[m] for m in range(20)]
    bigf = np.array([[float(x) for x in r] for r in big])
    # Schur tumleyeninin en kucuk oz degeri -> eta'
    Zp = bigf[:8, :8]; Wf = bigf[:8, 8:]; Kf = bigf[8:, 8:]
    eta_num = np.linalg.eigvalsh(Kf - Wf.T @ np.linalg.solve(Zp, Wf)).min()
    eta_p = Q3(Fr(eta_num*0.98).limit_denominator(10**8))
    big2 = [[big[i][j] - (eta_p if (i == j and i >= 8) else ZERO) for j in range(28)] for i in range(28)]
    assert ldl_pd(big2), "LMI (c) tam dogrulanamadi"
    # 6) tam (d): -Q - c' I >= 0
    Q = [[dot(b0, y2(i, j)) + (dot(B1ex(i), Y1[j]) + dot(B1ex(j), Y1[i]))/2 + dot(B2ex(i, j), y0) for j in range(4)] for i in range(4)]
    c_num = -np.linalg.eigvalsh(np.array([[float(x) for x in r] for r in Q])).max()
    c_p = Q3(Fr(c_num*0.98).limit_denominator(10**8))
    assert ldl_pd([[-Q[i][j] - (c_p if i == j else ZERO) for j in range(4)] for i in range(4)]), "(d) tam dogrulanamadi"
    # 1. mertebe terimler tam sifir mi
    assert all(dot(b0, Y1[i]).iszero() and dot(B1ex(i), y0).iszero() for i in range(4))
    print(f"tam dogrulandi: eta' = {float(eta_p):.5f}, c' = {float(c_p):.5f}")
    # 7) kalan-terim sinirlari (Frobenius normlari tam hesaplanip yukari yuvarlanir)
    sq = lambda x: math.sqrt(fup(x))
    e1 = sq(sum((frob2([r[:8] for r in Zh1[i][:8]]) for i in range(4)), ZERO))
    e2 = sq(sum((frob2([r[:8] for r in Zh2[(i, j)][:8]]) for i in range(4) for j in range(4)), ZERO))
    b1 = sq(sum((frob2([r[8:] for r in Zh1[i][:8]]) for i in range(4)), ZERO))
    b2 = sq(sum((frob2([r[8:] for r in Zh2[(i, j)][:8]]) for i in range(4) for j in range(4)), ZERO))
    r3 = sq(sum(((dot(B1ex(i), y2(j, l)) + dot(B2ex(i, j), Y1[l]))**2 for i in range(4) for j in range(4) for l in range(4)), ZERO))
    r4 = sq(sum((dot(B2ex(i, j), y2(l, m))**2 for i in range(4) for j in range(4) for l in range(4) for m in range(4)), ZERO))
    eta_f, c_f = float(eta_p)*(1 - 1e-9), float(c_p)*(1 - 1e-9)
    def ok(rho):
        E = e1*rho + e2*rho**2
        if a_inv*E >= 1: return False
        R = (b1**2*rho**2*a_inv**2*E + (2*b1*b2*rho**3 + b2**2*rho**4)*a_inv)/(1 - a_inv*E)
        if eta_f*rho**2 - R < 0: return False
        if -c_f + r3*rho + r4*rho**2 > 0: return False
        for kk in act:
            n1 = math.sqrt(sum(float(Y1[i][kk])**2 for i in range(4))) * (1 + 1e-9)
            n2 = math.sqrt(sum(float(y2(i, j)[kk])**2 for i in range(4) for j in range(4))) * (1 + 1e-9)
            if float(y0[kk])*(1 - 1e-12) - n1*rho - n2*rho**2 < 0: return False
        return True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    print(f"normlar: a={a_inv:.3g} e1={e1:.3g} e2={e2:.3g} b1={b1:.3g} b2={b2:.3g} r3={r3:.3g} r4={r4:.3g}")
    print(f"TITIZ YEREL YARICAP (delta koordinatlarinda): rho = {lo:.6g}   ->  mu-uzayinda Oklid yaricapi en az {lo*math.sqrt(2):.6g}")
    return lo

if __name__ == "__main__":
    run()
