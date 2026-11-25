# YEREL TOPUN TITIZ ISPATI (olceklenmis argumanla), Q(sqrt3) tam aritmetik.
# Girdi: yerel_egri_olcekli.pkl (sayisal y0, Y1, Y2).  Cikti: titiz yaricap rho (delta koordinatlari) ve mu-uzayi yaricapi sqrt2*rho.
#
# Iddia: |delta| <= rho icin y(delta) = y0 + Y1 delta + sum_{ij} delta_i delta_j Y2_{ij} dual-uygun ve
#        f(delta) = b(mu(delta))^T y(delta) <= T = 2+2/sqrt3.   Dolayisiyla bu topta Psi(mu) <= T.
# Ispat adimlari (hepsi asagida tam olarak denetlenir):
#   (1) Z0 = A(y0)-C,  Z0 Kr = 0 (Kr = X*'in menzili),  Gt^T Z0 Gt'nin pp blogu >= lam I
#   (2) Kr^T A(Y1_i) Kr = 0,  b0.Y1_i = 0,  B1_i.y0 = 0   (birinci mertebe terimler TAM sifir)
#   (3) [[Z0p - m0 I, W],[W^T, K - m0 I]] >= 0  =>  M0(d) >= m0 I  (|d|=1)
#   (4) -Q - c I >= 0   (amacin ikinci mertebesi <= -c|delta|^2)
#   (5) m0 - rho(e1+b2) - rho^2 e2 > 0,  -c + r3 rho + r4 rho^2 <= 0,  isaretli bilesenler >= 0
import sys, math, pickle, time, os
from fractions import Fraction as Fr
import numpy as np
from q3 import Q3, ZERO, ONE, matmul, transpose, ldl_pd, rref_solver
import genel_yerel_tam as GT0            # tam SIC vektorleri, X*, A matrisleri, C
import genel_yerel2 as L

TAU = GT0.TAU; TAU_EX = GT0.TAU_EX; T_EX = GT0.T_EX
NY = L.NY
t_start = time.time()
def log(*a):
    print(f"[{time.time()-t_start:6.0f}s]", *a, flush=True)

def frac(x, den):
    return Q3(Fr(float(x)).limit_denominator(den))

def dot(a, b):
    return sum((x*y for x, y in zip(a, b) if not x.iszero() and not y.iszero()), ZERO)

def Aof(y):
    Z = [[ZERO]*13 for _ in range(13)]
    for k, yk in enumerate(y):
        if yk.iszero(): continue
        Ak = GT0.AMATS[k]
        for i in range(13):
            for j in range(13):
                if not Ak[i][j].iszero():
                    Z[i][j] = Z[i][j] + yk*Ak[i][j]
    return Z

def b0_ex():
    return [ONE] + [Q3(Fr(1, 3))]*12 + [ZERO]*(NY - 13)
def B1ex(i):
    v = [ZERO]*NY
    for a in range(6):
        v[1+a] = TAU_EX[i][a]; v[7+a] = Q3(2)*TAU_EX[i][a]
    return v
def B2ex(i, j):
    v = [ZERO]*NY
    for a in range(6):
        v[7+a] = Q3(3)*TAU_EX[i][a]*TAU_EX[j][a]
    return v


def pd_margin(B, m, den=10**6):
    """B (Q3, simetrik) icin B - m I > 0 oldugunu TITIZ olarak dogrula:
       B' = rasyonel yuvarlama; tau^2 >= ||B - B'||_F^2 (tam isaret testi); B' - (m+tau) I > 0 (tam LDL, kucuk paydalar)."""
    n = len(B)
    Bq = [[Q3(Fr(float(B[i][j])).limit_denominator(den)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            Bq[i][j] = Bq[j][i]
    D2 = sum(((B[i][j] - Bq[i][j])*(B[i][j] - Bq[i][j]) for i in range(n) for j in range(n)), ZERO)
    tau = Fr(1, 10**4)
    while (Q3(tau*tau) - D2).sign() <= 0:
        tau *= 2
        if tau > 1: return False, None
    ok = ldl_pd([[Bq[i][j] - (Q3(Fr(m) + tau) if i == j else ZERO) for j in range(n)] for i in range(n)])
    return ok, float(tau)

def fro_ub(M):
    """Q3 girdili matrisin Frobenius normu icin titiz ust sinir: float donusum hatasi acikca eklenir."""
    s, e = 0.0, 0.0
    for r in M:
        for x in r:
            f = float(x); s += f*f
            e += (4e-16*(abs(float(x.a)) + 2*abs(float(x.b))) + 1e-300)**2
    return (math.sqrt(s) + math.sqrt(e))*(1 + 1e-12)

def main():
    data = pickle.load(open("yerel_egri_olcekli.pkl", "rb"))
    y0n, Y1n, Y2n, act, inact = data["y0"], data["Y1"], data["Y2"], data["act"], data["inact"]
    XS = GT0.XS; CEX = GT0.CEX
    assert sum((CEX[i][j]*XS[i][j] for i in range(13) for j in range(13)), ZERO) == T_EX
    ker, piv = GT0.nullspace_basis(XS)
    Ks = transpose([[XS[i][c] for i in range(13)] for c in piv])            # 13x5 basit menzil tabani
    Ns = transpose(ker)                                                      # 13x8 basit cekirdek tabani
    log("rank X* =", len(piv))
    # --- iyi kosullu tam taban: sayisal ortonormal tabanlara en yakin, tam olarak cekirdek/menzilde ---
    Xf = np.array([[float(x) for x in r] for r in XS]); ev, Vf = np.linalg.eigh(Xf)
    Pf, Kf = Vf[:, ev <= 1e-9], Vf[:, ev > 1e-9]
    Nsf = np.array([[float(x) for x in r] for r in Ns]); Ksf = np.array([[float(x) for x in r] for r in Ks])
    cP = np.linalg.lstsq(Nsf, Pf, rcond=None)[0]; cK = np.linalg.lstsq(Ksf, Kf, rcond=None)[0]
    cPe = [[frac(x, 10**4) for x in r] for r in cP]; cKe = [[frac(x, 10**4) for x in r] for r in cK]
    Pg = matmul(Ns, cPe); Kg = matmul(Ks, cKe)
    G = [Pg[i] + Kg[i] for i in range(13)]; Gt = transpose(G)
    # --- (1) tam y0: inaktifler 0, (A(y)-C)Ks = 0, b0.y = T, B1_i.y = 0 ---
    U = [k for k in range(NY) if k not in inact]
    AK = {k: matmul(GT0.AMATS[k], Ks) for k in U}; CK = matmul(CEX, Ks)
    rows, rhs = [], []
    for i in range(13):
        for jj in range(5):
            rows.append([AK[k][i][jj] for k in U]); rhs.append(CK[i][jj])
    b0 = b0_ex()
    rows.append([b0[k] for k in U]); rhs.append(T_EX)
    for i in range(4):
        B = B1ex(i); rows.append([B[k] for k in U]); rhs.append(ZERO)
    sol, rk = rref_solver(rows, rhs, [y0n[k] for k in U])
    y0 = [ZERO]*NY
    for k, v in zip(U, sol): y0[k] = v
    Z0 = [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(Aof(y0), CEX)]
    assert all(x.iszero() for r in matmul(Z0, Ks) for x in r), "(1) Z0 Kr != 0"
    assert all(y0[k].sign() > 0 for k in act), "(1) aktif carpan pozitif degil"
    assert dot(b0, y0) == T_EX and all(dot(B1ex(i), y0).iszero() for i in range(4))
    Zh0 = matmul(matmul(Gt, Z0), G); Z0p = [r[:8] for r in Zh0[:8]]
    log("(1) tam y0 bulundu, rank", rk)
    # --- (2) tam Y1 ---
    KAK = {k: matmul(transpose(Ks), AK[k]) for k in U}
    rows1 = [[b0[k] for k in U]]
    for i in range(5):
        for jj in range(i, 5):
            rows1.append([KAK[k][i][jj] for k in U])
    Y1 = []
    for i in range(4):
        s1, _ = rref_solver(rows1, [ZERO]*len(rows1), [Y1n[k, i] for k in U])
        v = [ZERO]*NY
        for k, x in zip(U, s1): v[k] = x
        Y1.append(v)
    assert all(dot(b0, Y1[i]).iszero() for i in range(4))
    log("(2) tam Y1 bulundu")
    # --- Y2 rasyonel, inaktiflerde 4x4 PD ---
    keys = [(i, j) for i in range(4) for j in range(i, 4)]
    Y2 = {kk: [frac(x, 10**8) for x in Y2n[kk]] for kk in keys}
    y2 = lambda i, j: Y2[(min(i, j), max(i, j))]
    for k in inact:
        if not ldl_pd([[y2(i, j)[k] for j in range(4)] for i in range(4)]):
            for i in range(4): Y2[(i, i)][k] = Y2[(i, i)][k] + Q3(Fr(1, 10**9))
            assert ldl_pd([[y2(i, j)[k] for j in range(4)] for i in range(4)]), "inaktif Y2 PD degil"
    # --- (3) olceklenmis LMI ---
    Zh1 = [matmul(matmul(Gt, Aof(Y1[i])), G) for i in range(4)]
    for i in range(4):
        assert all(Zh1[i][p][q].iszero() for p in range(8, 13) for q in range(8, 13)), "(2) Z1_kk != 0"
    log("Z1 hesaplandi")
    Zh2 = {}
    for (i, j) in keys:
        Zh2[(i, j)] = matmul(matmul(Gt, Aof(Y2[(i, j)])), G); Zh2[(j, i)] = Zh2[(i, j)]
    log("Z2 hesaplandi")
    W = [[Zh1[i][p][8+q] for i in range(4) for q in range(5)] for p in range(8)]
    K = [[Zh2[(i, j)][8+q][8+r] for j in range(4) for r in range(5)] for i in range(4) for q in range(5)]
    big = [Z0p[p] + W[p] for p in range(8)] + [[W[p][m] for p in range(8)] + K[m] for m in range(20)]
    bigf = np.array([[float(x) for x in r] for r in big])
    m0n = np.linalg.eigvalsh((bigf + bigf.T)/2).min()
    m0 = Q3(Fr(m0n*0.97).limit_denominator(10**6))
    okc, tau3 = pd_margin(big, Fr(m0n*0.97).limit_denominator(10**6))
    assert okc, "(3) LMI tam dogrulanamadi"
    log(f"(3) LMI tam dogrulandi, m0 = {float(m0):.5f}")
    # --- (4) ---
    Q = [[dot(b0, y2(i, j)) + (dot(B1ex(i), Y1[j]) + dot(B1ex(j), Y1[i]))/2 + dot(B2ex(i, j), y0) for j in range(4)] for i in range(4)]
    cn = -np.linalg.eigvalsh(np.array([[float(x) for x in r] for r in Q])).max()
    cc = Q3(Fr(cn*0.97).limit_denominator(10**6))
    okd, tau4 = pd_margin([[-Q[i][j] for j in range(4)] for i in range(4)], Fr(cn*0.97).limit_denominator(10**6))
    assert okd, "(4) tam dogrulanamadi"
    log(f"(4) tam dogrulandi, c = {float(cc):.5f}")
    # --- (5) kalan terim sinirlari: tam degerlerin float'a cevrilmesi + guvenlik payi ---
    SAFE = 1 + 1e-9
    e1 = math.sqrt(sum(fro_ub([r[:8] for r in Zh1[i][:8]])**2 for i in range(4)))*SAFE
    e2 = math.sqrt(sum(fro_ub([r[:8] for r in Zh2[(i, j)][:8]])**2 for i in range(4) for j in range(4)))*SAFE
    b2 = math.sqrt(sum(fro_ub([r[8:] for r in Zh2[(i, j)][:8]])**2 for i in range(4) for j in range(4)))*SAFE
    r3 = fro_ub([[dot(B1ex(i), y2(j, l)) + dot(B2ex(i, j), Y1[l]) for i in range(4) for j in range(4) for l in range(4)]])*SAFE
    r4 = fro_ub([[dot(B2ex(i, j), y2(l, m)) for i in range(4) for j in range(4) for l in range(4) for m in range(4)]])*SAFE
    m0f, cf = float(m0)/SAFE, float(cc)/SAFE   # m0, c: tam dogrulanan rasyonel marjlar
    sgn = [(float(y0[k])/SAFE, math.sqrt(sum(float(Y1[i][k])**2 for i in range(4)))*SAFE,
            math.sqrt(sum(float(y2(i, j)[k])**2 for i in range(4) for j in range(4)))*SAFE) for k in act]
    def ok(rho):
        if m0f - rho*(e1 + b2) - rho**2*e2 <= 0: return False
        if -cf + r3*rho + r4*rho**2 > 0: return False
        return all(a - b*rho - c*rho**2 >= 0 for a, b, c in sgn)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi)/2; lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    log(f"normlar: e1={e1:.4g} e2={e2:.4g} b2={b2:.4g} r3={r3:.4g} r4={r4:.4g}")
    log(f"TITIZ YEREL YARICAP: rho = {lo:.6g} (delta);  mu-uzayinda Oklid yaricapi >= {lo*math.sqrt(2):.6g}")
    log("Dis bolge (kutu bolme) ic topu: 0.008  ->  ORTUSME " + ("VAR: genel teorem kapanir" if lo*math.sqrt(2) >= 0.008 else "YOK: kutu bolmeyi daha kucuk ic topla tekrarla"))
    pickle.dump({"rho": lo, "m0": float(m0), "c": float(cc), "e1": e1, "e2": e2, "b2": b2, "r3": r3, "r4": r4},
                open("yerel_titiz_sonuc3.pkl", "wb"))

if __name__ == "__main__":
    main()
