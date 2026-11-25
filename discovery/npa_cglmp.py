# NPA hiyerarsisi (Q1, Q1+AB, Q2) ile CGLMP3 ust sinirlari ve qplex kutusunun uyeligi.
# Gerekli: numpy, scipy, cvxpy (Clarabel). Calistir:  <venv>/bin/python npa_cglmp.py
from itertools import product
from fractions import Fraction as Fr
import importlib.util, io, contextlib, os
import numpy as np
import cvxpy as cp

# --- qplex kutusu: rehberin kodundan (cglmp_qplex_kontrol.py) ---
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("g", os.path.join(here, "cglmp_qplex_kontrol.py"))
g = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(g)
BOX = {(a, b, x, y): g.probability(x, y, a, b) for a, b, x, y in product(range(3), range(3), range(2), range(2))}
PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))   # (x, y, +1 sinifi, -1 sinifi), k = a-b mod 3

def I3_of(p):
    q = lambda x, y, k: sum(p(a, (a - k) % 3, x, y) for a in range(3))
    return sum(q(x, y, gp) - q(x, y, lm) for x, y, gp, lm in PAIRS)

# --- NPA kelimeleri: projektorler (x, a), a in {0,1}; son sonuc normalizasyondan ---
def reduce(seq):
    out = []
    for op in seq:
        if out and out[-1][0] == op[0]:          # ayni ayar: P_a P_a' = delta P_a
            if out[-1][1] != op[1]:
                return None
            continue
        out.append(op)
    return tuple(out)

def monomials(level):
    A = [((x, a),) for x in range(2) for a in range(2)]
    one = [()]
    mons = [((), ())] + [(w, ()) for w in A] + [((), w) for w in A]
    if level in ("1+AB", "2"):
        mons += [(wa, wb) for wa in A for wb in A]
    if level == "2":
        AA = [((x, a), (1 - x, a2)) for x in range(2) for a in range(2) for a2 in range(2)]
        mons += [(w, ()) for w in AA] + [((), w) for w in AA]
    return mons

def build(level):
    mons = monomials(level)
    n = len(mons)
    classes, zeros = {}, []
    for i, (ai, bi) in enumerate(mons):
        for j, (aj, bj) in enumerate(mons):
            aw, bw = reduce(ai[::-1] + aj), reduce(bi[::-1] + bj)
            if aw is None or bw is None:
                zeros.append((i, j)); continue
            key = min((aw, bw), (aw[::-1], bw[::-1]))
            classes.setdefault(key, []).append((i, j))
    return mons, n, classes, zeros

def prob_exprs(G, classes):
    e = lambda key: G[classes[key][0]]
    pA = lambda a, x: e((((x, a),), ())) if a < 2 else 1 - e((((x, 0),), ())) - e((((x, 1),), ()))
    pB = lambda b, y: e(((), ((y, b),))) if b < 2 else 1 - e(((), ((y, 0),))) - e(((), ((y, 1),)))
    def p(a, b, x, y):
        if a < 2 and b < 2:
            return e((((x, a),), ((y, b),)))
        if a < 2:
            return pA(a, x) - p(a, 0, x, y) - p(a, 1, x, y)
        return pB(b, y) - p(0, b, x, y) - p(1, b, x, y)
    return p

def npa_problem(level, positivity=False):
    mons, n, classes, zeros = build(level)
    G = cp.Variable((n, n), symmetric=True)
    cons = [G >> 0, G[0, 0] == 1]
    for pos in classes.values():
        cons += [G[pos[0]] == G[q] for q in pos[1:]]
    cons += [G[z] == 0 for z in zeros]
    p = prob_exprs(G, classes)
    if positivity:
        cons += [p(a, b, x, y) >= 0 for a, b, x, y in product(range(3), range(3), range(2), range(2))]
    return G, cons, p, (mons, n, classes, zeros)

def max_I3(level, positivity=False):
    G, cons, p, _ = npa_problem(level, positivity)
    pr = cp.Problem(cp.Maximize(I3_of(p)), cons)
    pr.solve(solver=cp.CLARABEL)
    return pr.value

def box_visibility(level, positivity=False):
    """max v: v*kutu + (1-v)*duz dagilim seviyede uygulanabilir mi?"""
    G, cons, p, _ = npa_problem(level, positivity)
    v = cp.Variable()
    cons += [p(a, b, x, y) == v * BOX[a, b, x, y] + (1 - v) / 9
             for a, b, x, y in product(range(3), range(3), range(2), range(2))]
    pr = cp.Problem(cp.Maximize(v), cons + [v <= 2])
    pr.solve(solver=cp.CLARABEL)
    return float(v.value)

# --- Q1+AB ust siniri icin TAM aritmetikle dogrulanan dual sertifika ---
def rigorous_bound(level):
    """lam - I3 = <Z,G> + sum_w s_w * val(w);  Z >= 0 (tam LDL), |val(w)| <= 1  =>  I3 <= lam - s_1 + sum|s_w|."""
    mons, n, classes, zeros = build(level)
    # I3'u moment girdileri cinsinden katsayilara ac (her kelime sinifinin ilk konumuna yaz)
    keys = list(classes)
    coef = {k: 0.0 for k in keys}
    const = 0.0
    def lin(a, b, x, y):  # p(a,b|x,y) = sabit + sum c_k * val(k)
        if a < 2 and b < 2:
            return 0.0, {(((x, a),), ((y, b),)): 1.0}
        def pAl(a_, x_):
            if a_ < 2: return 0.0, {(((x_, a_),), ()): 1.0}
            return 1.0, {(((x_, 0),), ()): -1.0, (((x_, 1),), ()): -1.0}
        def pBl(b_, y_):
            if b_ < 2: return 0.0, {((), ((y_, b_),)): 1.0}
            return 1.0, {((), ((y_, 0),)): -1.0, ((), ((y_, 1),)): -1.0}
        if a < 2:
            c0, d = pAl(a, x); c1, d1 = lin(a, 0, x, y); c2, d2 = lin(a, 1, x, y)
            out = dict(d)
            for dd in (d1, d2):
                for k, v in dd.items(): out[k] = out.get(k, 0) - v
            return c0 - c1 - c2, out
        c0, d = pBl(b, y); c1, d1 = lin(0, b, x, y); c2, d2 = lin(1, b, x, y)
        out = dict(d)
        for dd in (d1, d2):
            for k, v in dd.items(): out[k] = out.get(k, 0) - v
        return c0 - c1 - c2, out
    for x, y, gp, lm in PAIRS:
        for a in range(3):
            for k_, sgn in ((gp, 1), (lm, -1)):
                c0, d = lin(a, (a - k_) % 3, x, y)
                const += sgn * c0
                for k, v in d.items(): coef[k] += sgn * v
    # Dual SDP: min lam  s.t.  Z = lam*E00 - C - N >= 0, N sinif toplamlari sifir (sifir kelimeler serbest)
    lam = cp.Variable()
    Nv = cp.Variable((n, n), symmetric=True)
    C = np.zeros((n, n))
    for k, v in coef.items():
        i, j = classes[k][0]
        C[i, j] += v / 2; C[j, i] += v / 2
    E00 = np.zeros((n, n)); E00[0, 0] = 1
    cons = [lam * E00 - C - Nv >> 0]
    for k, pos in classes.items():
        cons.append(sum(Nv[q] for q in pos) == 0)
    pr = cp.Problem(cp.Minimize(lam), cons)
    pr.solve(solver=cp.CLARABEL)
    Z = lam.value * E00 - C - Nv.value
    Z = (Z + Z.T) / 2
    Z = Z + (max(0.0, -np.linalg.eigvalsh(Z).min()) * 1.5 + 1e-9) * np.eye(n)   # kesin pozitif yap
    Zr = [[Fr(Z[i, j]).limit_denominator(10**9) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i):
            Zr[i][j] = Zr[j][i]
    # tam LDL^T: tum pivotlar > 0  =>  Zr kesin pozitif tanimli
    M = [row[:] for row in Zr]
    for k in range(n):
        piv = M[k][k]
        if piv <= 0:
            return None
        for i in range(k + 1, n):
            f = M[i][k] / piv
            if f:
                for j in range(k, n):
                    M[i][j] -= f * M[k][j]
    # lam_r - I3 = <Zr,G> + sum_w s_w val(w);  s_w = sum_{class} (lam_r E00 - C - Zr)_ij
    lam_r = Fr(float(lam.value)).limit_denominator(10**6)   # bound'da sadelesir; sadece okunurluk icin
    Cr = [[Fr(C[i, j]) for j in range(n)] for i in range(n)]
    s_one, s_abs = Fr(0), Fr(0)
    for k, pos in classes.items():
        s = sum((lam_r if (i, j) == (0, 0) else 0) - Cr[i][j] - Zr[i][j] for i, j in pos)
        if k == ((), ()):
            s_one = s
        else:
            s_abs += abs(s)
    # sifir kelimeli girdiler G'de 0 oldugu icin katkisi yok
    bound = lam_r + Fr(const) - s_one + s_abs
    return float(lam.value) + const, bound

if __name__ == "__main__":
    print("qplex kutusunun I3 degeri:", I3_of(lambda a, b, x, y: BOX[a, b, x, y]))
    print("\nNPA ust sinirlari (CGLMP3, I3):")
    print("  Q1 (pozitiflik ekli)  :", round(max_I3("1", True), 7), "  analitik 2+2/sqrt3 =", round(2 + 2 / 3**0.5, 7))
    print("  Q1 (pozitifliksiz)    :", round(max_I3("1", False), 7))
    print("  Q1+AB (almost-quantum):", round(max_I3("1+AB"), 7), "  kuantum 1+sqrt(11/3) =", round(1 + (11 / 3)**0.5, 7))
    print("  Q2                    :", round(max_I3("2"), 7))
    print("\nqplex kutusu icin en buyuk gorunurluk v (v*kutu + (1-v)*gurultu uygulanabilir):")
    print("  Q1 (pozitiflik ekli)  : v* =", round(box_visibility("1", True), 6))
    print("  Q1+AB                 : v* =", round(box_visibility("1+AB"), 6))
    num, bound = rigorous_bound("1+AB")
    print(f"\nQ1+AB dual sertifikasi: sayisal {num:.7f}, TAM aritmetikle kanitlanan I3 <= {float(bound):.7f}")
    print("  => qplex kutusu (3.1547) Q1+AB disinda:", bound < Fr(3154, 1000))
