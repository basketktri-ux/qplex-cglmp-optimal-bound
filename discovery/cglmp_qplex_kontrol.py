# CGLMP / qplex kontrolleri — saf Python 3, ek paket gerekmez.
# 1) Rehberin s.10-11 kodu  2) 81 deterministik strateji  3) Esit marjinal SDP: 12x12 -> 4x4 indirgeme
# 4) Dual sertifika: I3 <= 2 + 2/sqrt3, tam (Q(sqrt3)) aritmetikle dogrulama  5) Q1 uyelik sertifikasi

from cmath import exp
from math import pi, sqrt
from itertools import product

outcomes = range(3)
omega = exp(2j*pi/3)
r = sqrt(3)
h, k = (1+r + 1j*(1-r))/4, (-1+1j)/2
phases = ((h, k), (h.conjugate(), h))
grid = list(product(outcomes, repeat=2))
pairs = ((0, 0, 0, 2), (1, 0, 2, 0),
         (1, 1, 0, 2), (0, 1, 0, 1))

def u(x, a):
    return [(1 + (omega**(ij[x]-a)).real)/9 for ij in grid]

def v(y, b):
    z0, z1 = phases[y]
    return [(1 + (omega**(-b)*(z0*omega**i+z1*omega**j)).real)/9
            for i, j in grid]

def dot(p, q):
    return sum(p_i*q_i for p_i, q_i in zip(p, q))

def probability(x, y, a, b):
    return 4*dot(u(x, a), v(y, b)) - 1/3

def q(x, y, k, raw_sum=False):
    shift = (lambda a: k-a) if raw_sum else (lambda a: a-k)
    return sum(probability(x, y, a, shift(a) % 3) for a in outcomes)

def score(raw_sum=False):
    return sum(q(x,y,g,raw_sum)-q(x,y,l,raw_sum) for x,y,g,l in pairs)

eps = 1e-12
vectors = [f(s, a) for f in (u,v) for s in range(2) for a in outcomes]
assert all(abs(sum(p)-1) < eps and min(p) >= -eps for p in vectors)
assert all(1/12-eps <= dot(p,q) <= 1/6+eps
           for p, q in product(vectors, repeat=2))
for x, y in product(range(2), repeat=2):
    for i in outcomes:
        assert abs(sum(probability(x,y,i,b) for b in outcomes)-1/3) < eps
        assert abs(sum(probability(x,y,a,i) for a in outcomes)-1/3) < eps
    assert all(probability(x,y,a,b) >= -eps
               for a,b in product(outcomes, repeat=2))
    print((x,y), tuple(round(q(x,y,i), 10) for i in outcomes))
assert abs(score() - (2+2*r/3)) < eps
assert abs(score(raw_sum=True)) < eps
print("CGLMP duzeltilmis =", score(), "| ham toplam =", score(True))

# --- 81 deterministik strateji kontrolu (rehberde kodu yok) ---
from collections import Counter
def det_score(a0,a1,b0,b1):
    A, B = (a0,a1), (b0,b1)
    s = 0
    for x,y,g,l in pairs:
        e = (A[x]-B[y]) % 3
        s += (e == g) - (e == l)
    return s
print("81 deterministik:", sorted(Counter(det_score(*t) for t in product(range(3), repeat=4)).items()))

print("\n--- Esit marjinal SDP gevsetmesi ---")

from math import cos, sin
w = exp(2j*pi/3); c = 1/9
def four(vecs):  # X = sum_a w^a (v_a - c)
    return [sum(w**a*(vecs[a][i]-c) for a in range(3)) for i in range(9)]
X = [four([u(x,a) for a in range(3)]) for x in range(2)]
Y = [four([v(y,b) for b in range(3)]) for y in range(2)]
vecs = [[2*t for t in V] for V in X+Y]       # sira: a0,a1,b0,b1 (birim norm)
herm = lambda p,q: sum(pi_*qi.conjugate() for pi_,qi in zip(p,q))
K = [[herm(p,q) for q in vecs] for p in vecs]
phi = {(0,0):-pi/6,(1,0):5*pi/6,(1,1):-pi/6,(0,1):pi/6}
F = sum((exp(1j*phi[x,y])*K[x][2+y]).real for (x,y) in phi)
print("diag K:", [round(K[i][i].real,12) for i in range(4)])
print("I3 = (2/sqrt3) F =", 2/sqrt(3)*F, " (dogrudan:", score(), ")")
tri = lambda z: min((w**m*z).real for m in range(3))
for (x,y) in phi: print(f"  kappa_{x}{y}: |k|={abs(K[x][2+y]):.6f}  min_m Re(w^m k)={tri(K[x][2+y]):.6f}")
print("  <a0,a1>, <b0,b1> =", round(abs(K[0][1]),12), round(abs(K[2][3]),12))
active = [(j,k,m) for j in range(4) for k in range(j+1,4) for m in range(3)
          if abs((w**m*K[j][k]).real + 0.5) < 1e-9]
print("aktif ucgen kisitlari:", active)

def lam_max(H):  # 4x4 Hermitsel -> 8x8 reel simetrik, Jacobi
    n = len(H); A = [[0.0]*(2*n) for _ in range(2*n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = A[i+n][j+n] = H[i][j].real
            A[i+n][j] = H[i][j].imag; A[i][j+n] = -H[i][j].imag
    N = 2*n
    for _ in range(100):
        off = sum(A[i][j]**2 for i in range(N) for j in range(N) if i != j)
        if off < 1e-26: break
        for p in range(N):
            for q in range(p+1, N):
                if abs(A[p][q]) < 1e-30: continue
                th = 0.5*__import__('math').atan2(2*A[p][q], A[q][q]-A[p][p])
                cs, sn = cos(th), sin(th)
                for r in range(N):
                    arp, arq = A[r][p], A[r][q]
                    A[r][p], A[r][q] = cs*arp - sn*arq, sn*arp + cs*arq
                for r in range(N):
                    apr, aqr = A[p][r], A[q][r]
                    A[p][r], A[q][r] = cs*apr - sn*aqr, sn*apr + cs*aqr
    return max(A[i][i] for i in range(N))

Wm = [[0j]*4 for _ in range(4)]
for (x,y),f in phi.items():
    Wm[2+y][x] += exp(1j*f)/2; Wm[x][2+y] += exp(-1j*f)/2
def Emat(j,k,m):
    E = [[0j]*4 for _ in range(4)]; E[k][j] = w**m/2; E[j][k] = (w**m).conjugate()/2; return E
Esum = [[sum(Emat(*a)[i][j] for a in active) for j in range(4)] for i in range(4)]
def dual(lam):  # F <= 4*y + (1/2)*sum(lambda), y = lam_max(W + lam*sum E)
    M = [[Wm[i][j] + lam*Esum[i][j] for j in range(4)] for i in range(4)]
    return 4*lam_max(M) + 0.5*lam*len(active)
lo, hi = 0.0, 3.0
for _ in range(80):  # altin oran aramasi
    a = lo + 0.382*(hi-lo); b = lo + 0.618*(hi-lo)
    if dual(a) < dual(b): hi = b
    else: lo = a
lam = (lo+hi)/2
print(f"\nDual: lambda* = {lam:.10f}  ->  F <= {dual(lam):.12f}   (1+sqrt3 = {1+sqrt(3):.12f})")
print(f"I3 ust siniri = {2/sqrt(3)*dual(lam):.12f}   (2+2/sqrt3 = {2+2/sqrt(3):.12f})")

print("\n--- Tam aritmetik sertifika ---")

from fractions import Fraction as Fr

from itertools import permutations, combinations
class R3:  # p + q*sqrt3
    def __init__(s,p,q=0): s.p, s.q = Fr(p), Fr(q)
    def __add__(s,o): return R3(s.p+o.p, s.q+o.q)
    def __sub__(s,o): return R3(s.p-o.p, s.q-o.q)
    def __mul__(s,o): return R3(s.p*o.p+3*s.q*o.q, s.p*o.q+s.q*o.p)
    def __neg__(s): return R3(-s.p,-s.q)
    def sign(s):
        if s.p >= 0 and s.q >= 0: return 0 if (s.p==0 and s.q==0) else 1
        if s.p <= 0 and s.q <= 0: return -1
        d = s.p*s.p - 3*s.q*s.q
        return (1 if s.p > 0 else -1) if d > 0 else ((1 if s.q > 0 else -1) if d < 0 else 0)
    def __repr__(s): return f"({s.p} + {s.q}*sqrt3)"
class C3:  # re + i*im, re,im in Q(sqrt3)
    def __init__(s,re,im=None): s.re, s.im = re, (im if im is not None else R3(0))
    def __add__(s,o): return C3(s.re+o.re, s.im+o.im)
    def __sub__(s,o): return C3(s.re-o.re, s.im-o.im)
    def __mul__(s,o): return C3(s.re*o.re-s.im*o.im, s.re*o.im+s.im*o.re)
    def conj(s): return C3(s.re, -s.im)
Z0 = C3(R3(0)); h = Fr(1,2)
w  = [C3(R3(1)), C3(R3(-h), R3(0,h)), C3(R3(-h), R3(0,-h))]        # 1, omega, omega^2
eip = {(0,0): C3(R3(0,h),R3(-h)), (1,0): C3(R3(0,-h),R3(h)),        # e^{i phi}
       (1,1): C3(R3(0,h),R3(-h)), (0,1): C3(R3(0,h),R3(h))}
half = C3(R3(h))
M = [[Z0]*4 for _ in range(4)]
def add(i,j,z): M[i][j] = M[i][j] + z
for (x,y),e in eip.items():               # W
    add(2+y, x, e*half); add(x, 2+y, e.conj()*half)
lam = C3(R3(-h, h))                        # (sqrt3 - 1)/2
for (j,k,m) in [(0,2,1),(0,3,2),(1,2,0),(1,3,1)]:   # lambda * E_aktif
    add(k, j, lam*w[m]*half); add(j, k, lam*w[m].conj()*half)
Z = [[(half if i==j else Z0) - M[i][j] for j in range(4)] for i in range(4)]
def det(S):
    tot = Z0
    for p in permutations(range(len(S))):
        sgn = 1
        for a in range(len(p)):
            for b in range(a+1,len(p)):
                if p[a] > p[b]: sgn = -sgn
        t = C3(R3(sgn))
        for r,cidx in enumerate(p): t = t*S[r][cidx]
        tot = tot + t
    return tot
ok = True; ranks = []
for n in range(1,5):
    for idx in combinations(range(4), n):
        d = det([[Z[i][j] for j in idx] for i in idx])
        assert d.im.p == 0 and d.im.q == 0      # Hermitsel minor reel olmali
        s = d.re.sign(); ok &= (s >= 0)
        if s > 0: ranks.append(n)
        print(idx, d.re, "isaret", s)
print("TUM ESAS MINORLER >= 0 :", ok, "| rank(Z) =", max(ranks))
print("Dual deger = 4*(1/2) + (1/2)*4*lambda = 1 + sqrt3  =>  I3 <= (2/sqrt3)(1+sqrt3) = 2 + 2/sqrt3")

print("\n--- Q1 (NPA seviye 1 = makroskopik yerellik) sertifikasi ---")
# Q1 <=> 1 ve tum A_{a|x}, B_{b|y} icin bir Gram temsili: <1,A>=p(a|x), <A,A'>=delta p(a|x) (ayni x),
# <A,B>=p(ab|xy); farkli ayarlar arasi A-A, B-B serbest.  Aday: A_{a|x} -> e0/3 + 2(u^x_a - c)
def lift(p): return [1/3] + [2*(t - 1/9) for t in p]
ops = [("1", [1.0] + [0.0]*9)]
ops += [(("A",x,a), lift(u(x,a))) for x in range(2) for a in range(3)]
ops += [(("B",y,b), lift(v(y,b))) for y in range(2) for b in range(3)]
idx = {n:i for i,(n,_) in enumerate(ops)}
G = [[sum(p*q for p,q in zip(P,Q)) for _,Q in ops] for _,P in ops]
err = abs(G[0][0]-1)
for P in "AB":
    for s in range(2):
        for o in range(3):
            i = idx[(P,s,o)]; err = max(err, abs(G[0][i]-1/3))
            for o2 in range(3):
                err = max(err, abs(G[i][idx[(P,s,o2)]] - (1/3 if o==o2 else 0)))
for x,y,a,b in product(range(2), range(2), range(3), range(3)):
    err = max(err, abs(G[idx[("A",x,a)]][idx[("B",y,b)]] - probability(x,y,a,b)))
print("NPA-1 kisitlarindan en buyuk sapma:", err)
print("Moment matrisi min oz degeri (Gram => PSD):", -lam_max([[-t for t in row] for row in G]))
print("=> qplex kutusu Q1 icinde; I3 =", score(), "= Q1 ust siniri 2+2/sqrt3 =", 2+2/sqrt(3))
