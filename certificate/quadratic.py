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

