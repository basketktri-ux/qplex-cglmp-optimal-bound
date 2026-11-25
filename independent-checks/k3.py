# Sifirdan yazilmis Q(sqrt3) aritmetigi (q3.py'den bagimsiz): a + b*sqrt(3), a,b rasyonel.
from fractions import Fraction as F
class Q3:
    __slots__ = ('a', 'b')
    def __init__(s, a=0, b=0): s.a = F(a); s.b = F(b)
    @staticmethod
    def _c(o): return o if isinstance(o, Q3) else Q3(o)
    def __add__(s, o): o = Q3._c(o); return Q3(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = Q3._c(o); return Q3(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return Q3._c(o) - s
    def __neg__(s): return Q3(-s.a, -s.b)
    def __mul__(s, o): o = Q3._c(o); return Q3(s.a*o.a + 3*s.b*o.b, s.a*o.b + s.b*o.a)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = Q3._c(o); n = o.a*o.a - 3*o.b*o.b
        if n == 0: raise ZeroDivisionError
        return Q3((s.a*o.a - 3*s.b*o.b)/n, (s.b*o.a - s.a*o.b)/n)
    def iszero(s): return s.a == 0 and s.b == 0
    def sign(s):
        sa = (s.a > 0) - (s.a < 0); sb = (s.b > 0) - (s.b < 0)
        if sb == 0 or sa == sb: return sa
        if sa == 0: return sb
        return sa if s.a*s.a > 3*s.b*s.b else sb      # zit isaret; a^2 = 3b^2 imkansiz
    def __eq__(s, o): o = Q3._c(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def __float__(s): return float(s.a) + float(s.b)*3**0.5
ZERO = Q3(); ONE = Q3(1)
