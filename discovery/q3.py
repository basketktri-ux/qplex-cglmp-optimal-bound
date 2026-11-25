# Q(sqrt3) cisminde tam aritmetik + kucuk dogrusal cebir araclari.
from fractions import Fraction as Fr
import math

class Q3:
    __slots__ = ("a", "b")          # a + b*sqrt3
    def __init__(s, a=0, b=0):
        s.a = a if isinstance(a, Fr) else Fr(a); s.b = b if isinstance(b, Fr) else Fr(b)
    @staticmethod
    def c(x):
        return x if isinstance(x, Q3) else Q3(x)
    def __add__(s, o): o = Q3.c(o); return Q3(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = Q3.c(o); return Q3(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return Q3.c(o) - s
    def __mul__(s, o):
        o = Q3.c(o); return Q3(s.a*o.a + 3*s.b*o.b, s.a*o.b + s.b*o.a)
    __rmul__ = __mul__
    def __neg__(s): return Q3(-s.a, -s.b)
    def __pow__(s, n):
        assert isinstance(n, int) and n >= 0
        r = Q3(1)
        for _ in range(n): r = r*s
        return r
    def conj(s): return Q3(s.a, -s.b)
    def norm(s): return s.a*s.a - 3*s.b*s.b
    def __truediv__(s, o):
        o = Q3.c(o); n = o.norm()
        if n == 0: raise ZeroDivisionError
        t = s*o.conj(); return Q3(t.a/n, t.b/n)
    def __rtruediv__(s, o): return Q3.c(o)/s
    def sign(s):
        a, b = s.a, s.b
        if a >= 0 and b >= 0: return 0 if (a == 0 and b == 0) else 1
        if a <= 0 and b <= 0: return -1
        d = a*a - 3*b*b
        if d == 0: return 0
        return (1 if a > 0 else -1) if d > 0 else (1 if b > 0 else -1)
    def iszero(s): return s.a == 0 and s.b == 0
    def __float__(s): return float(s.a) + float(s.b)*math.sqrt(3)
    def __repr__(s): return f"({s.a}+{s.b}r3)"
    def __eq__(s, o): o = Q3.c(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))

ZERO, ONE = Q3(0), Q3(1)
R3 = Q3(0, 1)

def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum((A[i][k]*B[k][j] for k in range(m) if not A[i][k].iszero() and not B[k][j].iszero()), ZERO)
             for j in range(p)] for i in range(n)]

def transpose(A):
    return [list(r) for r in zip(*A)]

def ldl_pd(M):
    """Tam LDL^T: tum pivotlar > 0 ise True (kesin pozitif tanimli)."""
    n = len(M); A = [row[:] for row in M]
    for k in range(n):
        if A[k][k].sign() <= 0:
            return False
        for i in range(k+1, n):
            if A[i][k].iszero(): continue
            f = A[i][k]/A[k][k]
            for j in range(k, n):
                if not A[k][j].iszero():
                    A[i][j] = A[i][j] - f*A[k][j]
    return True

def rref_solver(Arows, rhs, float_guess):
    """A y = rhs (Q3) dogrusal sistemini coz: serbest degiskenler float_guess'in rasyonellestirilmisi, bagimlilar tam.
       Pivotlar float buyuklugune gore secilir."""
    m = len(Arows); n = len(Arows[0])
    A = [Arows[i][:] + [rhs[i]] for i in range(m)]
    piv_cols, r = [], 0
    for col in range(n):
        best, bv = None, 0.0
        for i in range(r, m):
            v = abs(float(A[i][col]))
            if v > bv + 1e-300 and not A[i][col].iszero():
                best, bv = i, v
        if best is None: continue
        A[r], A[best] = A[best], A[r]
        pv = A[r][col]
        A[r] = [x/pv for x in A[r]]
        for i in range(m):
            if i != r and not A[i][col].iszero():
                f = A[i][col]
                A[i] = [A[i][j] - f*A[r][j] for j in range(n+1)]
        piv_cols.append(col); r += 1
        if r == m: break
    for i in range(r, m):                      # tutarlilik
        if not A[i][n].iszero():
            raise ValueError("tutarsiz sistem")
    free = [c for c in range(n) if c not in piv_cols]
    y = [None]*n
    for c in free:
        y[c] = Q3(Fr(float(float_guess[c])).limit_denominator(10**12))
    for i, c in enumerate(piv_cols):
        y[c] = A[i][n] - sum((A[i][f]*y[f] for f in free if not A[i][f].iszero()), ZERO)
    return y, len(piv_cols)
