# Dis bolge kutu sertifikalarinin TAM aritmetikle dogrulanmasi.
# Her kutu: Z(y) >= eps I ile yeniden coz -> y'yi rasyonele yuvarla -> tam Z'yi kur -> tam LDL (tum pivotlar > 0)
#           -> 16 kosede f_y(mu_v) = y00 + yA.mu + 3 zA.mu^2 <= T_alt  (T_alt rasyonel, T_alt < 2+2/sqrt3).
import sys, pickle, time
from fractions import Fraction as Fr
import numpy as np
import cvxpy as cp
import genel_bb as B

T_ALT = Fr(3154700538, 10**9)          # 3.154700538 < 2 + 2/sqrt3 = 3.15470053837...
assert T_ALT**0 == 1
EPS = 1e-6

def build_margin():
    N = B.N
    y00 = cp.Variable(); yA = cp.Variable(6); zA = cp.Variable(6, nonneg=True)
    u = cp.Variable(len(B.OFF), nonneg=True); wy = cp.Variable((2, N)); w = cp.Variable(N); t = cp.Variable()
    Z = y00*B.E(0, 0) - B.CMAT
    for k, i in enumerate(B.ALICE):
        Z = Z + yA[k]*B.E(i, 0) + zA[k]*B.E(i, i)
    Z = Z - sum(u[k]*B.E(i, j) for k, (i, j) in enumerate(B.OFF))
    Z = Z - B.symvec(wy[0], B.BV[0]) - B.symvec(wy[1], B.BV[1]) - B.symvec(w, B.CV)
    Pmu = cp.Parameter((16, 6)); Pmu2 = cp.Parameter((16, 6))
    cons = [(Z + Z.T)/2 - EPS*np.eye(N) >> 0, y00 + Pmu @ yA + 3*(Pmu2 @ zA) <= t]
    return cp.Problem(cp.Minimize(t), cons), Pmu, Pmu2, (y00, yA, zA, u, wy, w)

def fr(x, den=10**10):
    return Fr(float(x)).limit_denominator(den)

def exact_Z(y00, yA, zA, u, wy, w):
    N = B.N; h = Fr(1, 2)
    Z = [[Fr(0)]*N for _ in range(N)]
    def add(i, j, val):
        Z[i][j] += val
    add(0, 0, y00)
    for k, i in enumerate(B.ALICE):
        add(i, 0, yA[k]*h); add(0, i, yA[k]*h); add(i, i, zA[k])
    for k, (i, j) in enumerate(B.OFF):
        add(i, j, -u[k]*h); add(j, i, -u[k]*h)
    for vec, bv in ((wy[0], B.BV[0]), (wy[1], B.BV[1]), (w, B.CV)):
        for i in range(N):
            for j in range(N):
                if bv[j] != 0:
                    add(i, j, -vec[i]*Fr(int(bv[j]))*h)
                if bv[i] != 0:
                    add(i, j, -vec[j]*Fr(int(bv[i]))*h)
    for i in range(N):                                          # - C (girdiler 0, +-1/2)
        for j in range(N):
            if B.CMAT[i, j] != 0:
                Z[i][j] -= Fr(B.CMAT[i, j]).limit_denominator(4)
    return Z

def ldl_pd(Z):
    n = len(Z); M = [row[:] for row in Z]
    for k in range(n):
        if M[k][k] <= 0:
            return False
        for i in range(k+1, n):
            f = M[i][k] / M[k][k]
            if f:
                for j in range(k, n):
                    M[i][j] -= f*M[k][j]
    return True

def verify_box(pr, Pmu, Pmu2, vs, lo, hi):
    V = np.array(B.box_vertices(lo, hi)); Pmu.value = V; Pmu2.value = V**2
    pr.solve(solver=cp.CLARABEL)
    if pr.value is None or pr.value > float(T_ALT) - 1e-9:
        return False, "marjli SDP yeterli degil"
    y00, yA, zA, u, wy, w = [v.value for v in vs]
    y00 = fr(y00); yA = [fr(x) for x in yA]; zA = [max(Fr(0), fr(x)) for x in zA]
    u = [max(Fr(0), fr(x)) for x in u]; wy = [[fr(x) for x in row] for row in wy]; w = [fr(x) for x in w]
    if not ldl_pd(exact_Z(y00, yA, zA, u, wy, w)):
        return False, "tam LDL basarisiz"
    for bits in range(16):                                      # tam kose koordinatlari (ikili kesirler)
        p = [Fr(float(lo[i] if not (bits >> (3-i)) & 1 else hi[i])) for i in range(4)]
        mu = [p[0], p[1], 1 - p[0] - p[1], p[2], p[3], 1 - p[2] - p[3]]
        f = y00 + sum(yA[k]*mu[k] for k in range(6)) + 3*sum(zA[k]*mu[k]**2 for k in range(6))
        if f > T_ALT:
            return False, "kose degeri T_alt'i asiyor"
    return True, "ok"

if __name__ == "__main__":
    data = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "bb_d0_0.0.pkl", "rb"))
    pr, Pmu, Pmu2, vs = build_margin()
    ok, fails, t0 = 0, [], time.time()
    for n, (lo, hi, val, _) in enumerate(data["cert"]):
        good, why = verify_box(pr, Pmu, Pmu2, vs, lo, hi)
        if good: ok += 1
        else: fails.append((lo, hi, why))
        if (n+1) % 200 == 0:
            print(f"  {n+1}/{len(data['cert'])} dogrulandi={ok} basarisiz={len(fails)} sure={time.time()-t0:.0f}s", flush=True)
    print(f"SONUC: {ok}/{len(data['cert'])} kutu tam aritmetikle dogrulandi; basarisiz {len(fails)}; ic (yerel) kutu {len(data['local'])}; acik kuyruk {len(data['queue'])}")
    for lo, hi, why in fails[:10]:
        print("  basarisiz:", np.round(lo, 5).tolist(), np.round(hi, 5).tolist(), why)
    pickle.dump(fails, open("bb_basarisiz.pkl", "wb"))
