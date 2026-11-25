# Bob homojen (saflik yok), Alice homojen a~ = mu (1, sqrt2 A); tek konvekslik engeli: |a~|^2 <= 3 mu^2.
# Alice'in 6 agirligi uzerinde moment gevsetmeleri:
#   seviye 1  : m=E[mu], U=E[mu mu^T];  saflik  X_aa <= 3 U_aa
#   seviye 1.5: her mu_c ile lokalize Gram X^(c)=E[mu_c X] ve 3. moment V_cab=E[mu_c mu_a mu_b] (derece-2 moment matrisi ile)
import itertools, sys
import numpy as np
import cvxpy as cp

PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
N = 13; E0 = 0
iA = lambda x, a: 1 + 3*x + a          # Gram indeksi
iB = lambda y, b: 7 + 3*y + b
aw = lambda x, a: 3*x + a              # agirlik indeksi (0..5)
ABLK = [[0, 1, 2], [3, 4, 5]]

def gram_cons(X, mvec, Udiag_fn, scale=1.0):
    """Gram X icin kisitlar; mvec[a] = X[a~,e0] hedefi, Udiag_fn(a) = saflik sag tarafi (3*E[..mu_a^2])."""
    c = [X >> 0, X[E0, E0] == scale]
    for x in range(2):
        for a in range(3):
            i = iA(x, a)
            c += [X[i, E0] == mvec[aw(x, a)], X[i, i] <= Udiag_fn(aw(x, a))]
    c += [X[i, j] >= 0 for i in range(1, N) for j in range(i+1, N)]
    for y in range(2):
        v = np.zeros(N); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1
        c.append(X @ v == 0)
    v = np.zeros(N); v[[iA(0, a) for a in range(3)]] = 1; v[[iA(1, a) for a in range(3)]] = -1
    c.append(X @ v == 0)
    return c

def objective(X):
    return sum(X[iA(x, a), iB(y, (a-g) % 3)] - X[iA(x, a), iB(y, (a-l) % 3)] for x, y, g, l in PAIRS for a in range(3))

def level1():
    M = cp.Variable((7, 7), symmetric=True); m, U = M[0, 1:], M[1:, 1:]
    X = cp.Variable((N, N), symmetric=True)
    cons = [M >> 0, M[0, 0] == 1, m >= 0, U >= 0]
    for blk in ABLK:
        cons.append(sum(m[a] for a in blk) == 1)
        cons += [sum(U[a, j] for a in blk) == m[j] for j in range(6)]
    cons += gram_cons(X, m, lambda a: 3*U[a, a])
    pr = cp.Problem(cp.Maximize(objective(X)), cons); pr.solve(solver=cp.CLARABEL)
    return pr.value, pr.status

def level15():
    # derece-2 moment matrisi: indeks {1, mu_a, mu_a mu_b (a<=b)}
    mons = [()] + [(a,) for a in range(6)] + [tuple(sorted(p)) for p in itertools.combinations_with_replacement(range(6), 2)]
    n = len(mons)
    Mm = cp.Variable((n, n), symmetric=True)
    key = {}
    cons = [Mm >> 0, Mm[0, 0] == 1]
    for i, p in enumerate(mons):
        for j, q in enumerate(mons):
            k = tuple(sorted(p + q))
            if k in key:
                cons.append(Mm[i, j] == Mm[key[k]])
            else:
                key[k] = (i, j)
    mom = lambda *idx: Mm[key[tuple(sorted(idx))]]
    cons += [mom(*k) >= 0 for k in key if len(k) >= 1]
    for blk in ABLK:                           # sum_{a in blk} mu_a = 1, tum dusuk derece momentlerle carpilmis
        for k in key:
            if len(k) <= 3:
                cons.append(sum(mom(*(k + (a,))) for a in blk) == mom(*k) if len(k) > 0 else sum(mom(a) for a in blk) == 1)
    X = cp.Variable((N, N), symmetric=True)
    cons += gram_cons(X, [mom(a) for a in range(6)], lambda a: 3*mom(a, a))
    Xc = [cp.Variable((N, N), symmetric=True) for _ in range(6)]
    for c in range(6):                         # mu_c ile lokalize
        cons += gram_cons(Xc[c], [mom(c, a) for a in range(6)], lambda a, c=c: 3*mom(c, a, a), scale=mom(c))
    for blk in ABLK:
        cons.append(sum(Xc[c] for c in blk) == X)
    # lokalize moment matrisleri (mu_c >= 0): indeks {1, mu_a}
    for c in range(6):
        L = cp.bmat([[mom(c) if (i == 0 and j == 0) else (mom(c, j-1) if i == 0 else (mom(c, i-1) if j == 0 else mom(c, i-1, j-1)))
                      for j in range(7)] for i in range(7)])
        cons.append(L >> 0)
    pr = cp.Problem(cp.Maximize(objective(X)), cons); pr.solve(solver=cp.CLARABEL)
    return pr.value, pr.status

if __name__ == "__main__":
    T = 2 + 2/np.sqrt(3)
    print("hedef 2+2/sqrt3 =", T)
    print("seviye 1  :", level1())
    print("seviye 1.5:", level15())
