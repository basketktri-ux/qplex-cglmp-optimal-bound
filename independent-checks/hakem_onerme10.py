# Hakem denetimi: Onerme 10 (Ek B) - biased.json'dan bagimsiz tam rasyonel kontrol. Kullanim: python3 -I hakem_onerme10.py <certificate klasoru>
import json, sys
from fractions import Fraction as F
from itertools import product
d = json.load(open(sys.argv[1] + "/biased.json"))
vec = lambda r: [F(n, r[0]) for n in r[1:]]
U = [vec(r) for r in d["alice_states"]]; V = [vec(r) for r in d["bob_states"]]
A = [U[0:3], U[3:6]]; B = [V[0:3], V[3:6]]
mu = [F(3, 5), F(1, 5), F(1, 5)]; ga = [[F(1, 3)]*3, [F(3, 8), F(5, 16), F(5, 16)]]
dot = lambda x, y: sum(a*b for a, b in zip(x, y))
assert all(r[0] > 0 and min(r[1:]) > 0 and sum(r[1:]) == r[0] for r in d["alice_states"] + d["bob_states"])
ov = [dot(x, y) for x in U + V for y in U + V]
print("ic carpimlar (oz-ciftler dahil): min", min(ov), "max", max(ov), "| (1/12,1/6) icinde:", F(1, 12) < min(ov) and max(ov) < F(1, 6))
bar = lambda w, S: [sum(w[a]*S[a][k] for a in range(3)) for k in range(9)]
assert bar(mu, A[0]) == bar(mu, A[1]), "ortak onsel"
assert all(bar(ga[y], B[y]) == [F(1, 9)]*9 for y in range(2)), "Bob normalizasyonu"
P = \
    {(x, y, a, b): mu[a]*ga[y][b]*(36*dot(A[x][a], B[y][b]) - 3) for x in range(2) for y in range(2) for a in range(3) for b in range(3)}
assert min(P.values()) > 0 and all(sum(P[x, y, a, b] for a in range(3) for b in range(3)) == 1 for x in range(2) for y in range(2))
pA = {(x, y, a): sum(P[x, y, a, b] for b in range(3)) for x in range(2) for y in range(2) for a in range(3)}
pB = {(x, y, b): sum(P[x, y, a, b] for a in range(3)) for x in range(2) for y in range(2) for b in range(3)}
assert all(pA[x, 0, a] == pA[x, 1, a] for x in range(2) for a in range(3)), "Alice sinyalsizlik"
assert all(pB[0, y, b] == pB[1, y, b] for y in range(2) for b in range(3)), "Bob sinyalsizlik"
print("min p =", min(P.values()), "| sinyalsizlik iki taraf icin tam | Alice marjinali", [pA[0, 0, a] for a in range(3)])
CLS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
I3 = sum(P[x, y, a, (a - g) % 3] - P[x, y, a, (a - l) % 3] for x, y, g, l in CLS for a in range(3))
print("I3 =", I3)
Hm = [[F(v) for v in r] for r in d["witness"]]
assert all(Hm[i][j] == Hm[j][i] for i in range(9) for j in range(9))
M = [r[:] for r in Hm]; minors, det = [], F(1)
for k in range(9):
    det *= M[k][k]; minors.append(det)
    for i in range(k+1, 9):
        f = M[i][k]/M[k][k]
        for j in range(k, 9): M[i][j] -= f*M[k][j]
print("ardisik bas minorler:", [int(m) for m in minors], "| hepsi > 0:", all(m > 0 for m in minors))
idx = [None, ('A', 0, 0), ('A', 0, 1), ('A', 1, 0), ('A', 1, 1), ('B', 0, 0), ('B', 0, 1), ('B', 1, 0), ('B', 1, 1)]
def gamma(i, j):
    if i == 0 and j == 0: return F(1)
    if i == 0 or j == 0:
        p, s, o = idx[max(i, j)]; return pA[s, 0, o] if p == 'A' else pB[0, s, o]
    (p1, s1, o1), (p2, s2, o2) = idx[i], idx[j]
    if p1 == p2 and s1 == s2: return (pA[s1, 0, o1] if p1 == 'A' else pB[0, s1, o1]) if o1 == o2 else F(0)
    if p1 == p2: return None                      # ayni tarafin farkli ayarlari: bilinmiyor
    if p1 == 'B': (s1, o1), (s2, o2) = (s2, o2), (s1, o1)
    return P[s1, s2, o1, o2]
unk = [(i, j) for i in range(9) for j in range(9) if gamma(i, j) is None]
print("bilinmeyen giris sayisi:", len(unk), "| H orada sifir:", all(Hm[i][j] == 0 for i, j in unk))
print("Tr(H Gamma) =", sum(Hm[i][j]*gamma(i, j) for i in range(9) for j in range(9) if gamma(i, j) is not None))
