"""Verify the fixed CGLMP witnesses using Python 3 standard-library arithmetic."""
import json
from fractions import Fraction as F
from itertools import combinations_with_replacement, product
from math import isqrt
from pathlib import Path

from biased import check_biased
from quadratic import Q3, ZERO, matmul, transpose, ldl_pd

HERE = Path(__file__).resolve().parent
HALF = F(1, 2)
CENTER = F(1, 3)
R = F(1, 125)
N = 13
AXES = range(4)
PAIRS = list(combinations_with_replacement(AXES, 2))
CLASSES = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
TANGENTS = ((1,-1,0,0,0,0), (1,1,-2,0,0,0), (0,0,0,1,-1,0), (0,0,0,1,1,-2))
SQRT3_APPROX = F(isqrt(3 * 10**100), 10**50)
GRID = 10**7

def check(condition, message):
    if not condition:
        raise ValueError(message)

def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))

def dual_matrix(y, include_objective):
    """Form Z from (4), with equality matrices symmetrized in the trace pairing."""
    check(len(y) == 118, 'dual dimension')
    zero = y[0]*0
    m = [[zero]*N for _ in range(N)]
    def sym(i, j, x):
        h = HALF*x
        m[i][j] += h
        m[j][i] += h
    m[0][0] = y[0]
    for i in range(1, 7):
        sym(0, i, y[i])
        m[i][i] += y[6+i]
    k = 13
    for i in range(1,N):
        for j in range(i+1,N):
            sym(i,j,-y[k])
            k += 1
    for entries in (((0,-1),(7,1),(8,1),(9,1)), ((0,-1),(10,1),(11,1),(12,1)),
                    ((1,1),(2,1),(3,1),(4,-1),(5,-1),(6,-1))):
        for i in range(N):
            for j, sign in entries:
                sym(i,j,-sign*y[k])
            k += 1
    if include_objective:
        for x, bsetting, plus, minus in CLASSES:
            for a in range(3):
                for diff, sign in ((plus,-1),(minus,1)):
                    sym(1+3*x+a,7+3*bsetting+(a-diff)%3,sign)
    return m

def positive_rational(matrix):
    """Positive pivots of successive symmetric Schur complements."""
    m = [row[:] for row in matrix]
    for k in range(len(m)):
        if m[k][k] <= 0:
            return False
        for i in range(k+1,len(m)):
            f = m[i][k] / m[k][k]
            for j in range(k+1,i+1):
                m[i][j] -= f*m[j][k]
    return True

def vertices(lo,hi):
    for a,b,c,d in product(*zip(lo,hi)):
        yield (a,b,1-a-b,c,d,1-c-d)

def check_outer():
    data = read('boxes.json')
    cap = F(3154700538,10**9)
    check(3*(cap-2)**2 < 4 and cap > 2, 'rational cap')
    leaves = set()
    for lo,hi,y in data['outer']:
        lo, hi = tuple(map(F,lo)), tuple(map(F,hi))
        y = list(map(F,y))
        check(all(v >= 0 for v in y[7:79]), 'negative multiplier')
        check(positive_rational(dual_matrix(y,True)), 'outer matrix')
        for mu in vertices(lo,hi):
            value = y[0]+sum(y[1+i]*q+3*y[7+i]*q*q for i,q in enumerate(mu))
            check(value <= cap,'vertex above bound')
        leaves.add((lo,hi))
    for lo,hi in data['inner']:
        lo,hi = tuple(map(F,lo)),tuple(map(F,hi))
        for mu in vertices(lo,hi):
            check(sum((q-CENTER)**2 for q in mu) <= R**2, 'inner radius')
        leaves.add((lo,hi))
    visited = set()
    def cover(lo,hi,depth=0):
        if (lo,hi) in leaves:
            visited.add((lo,hi))
            return
        if lo[0]+lo[1] >= 1 or lo[2]+lo[3] >= 1:
            return
        check(depth < 64, 'gap in cover')
        k = max(AXES,key=lambda i: hi[i]-lo[i])
        mid = (lo[k]+hi[k])/2
        cover(lo,hi[:k]+(mid,)+hi[k+1:],depth+1)
        cover(lo[:k]+(mid,)+lo[k+1:],hi,depth+1)
    for start in product(AXES,repeat=4):
        cover(tuple(F(i,4) for i in start),tuple(F(i+1,4) for i in start))
    check(visited==leaves and len(leaves)==len(data['inner'])+len(data['outer']), 'cover bookkeeping')
    # A finite union of closed boxes covering the interior contains its closure.
    print('Outer region verified:',len(data['outer']),'certified boxes and',len(data['inner']),'inner boxes.',flush=True)

def read_q(pair):
    return Q3(F(pair[0]),F(pair[1]))

def grid_integer(q):
    return round(GRID * (q.a + q.b*SQRT3_APPROX))

def norm_upper(coeffs):
    """Enclose each magnitude by a rational, then bound the Euclidean norm."""
    total = 0
    for q in coeffs:
        n = abs(grid_integer(q))+1
        h = F(n,GRID)
        check((h-q).sign()>=0 and (h+q).sign()>=0,'coefficient enclosure')
        total += n*n
    return F(isqrt(total)+1,GRID)

def lower_eigenvalue(m,margin):
    n = len(m)
    error = F(1,GRID)
    approx = [[F(grid_integer(x),GRID) for x in row] for row in m]
    for i,j in product(range(n),repeat=2):
        diff = m[i][j]-approx[i][j]
        check(m[i][j]==m[j][i] and (error+diff).sign()>=0 and (error-diff).sign()>=0,'matrix enclosure')
    for i in range(n):
        approx[i][i] -= margin+n*error  # spectral norm <= Frobenius norm <= n*error
    check(positive_rational(approx),'matrix margin')

def check_local():
    d = read('local.json')
    g = [[read_q(q) for q in row] for row in d['basis']]
    y0 = list(map(read_q,d['constant']))
    linear = [list(map(read_q,row)) for row in d['linear']]
    check(len(PAIRS) == len(d['quadratic']), 'quadratic coefficient count')
    quad = {key:[Q3(F(v)) for v in row] for key,row in zip(PAIRS,d['quadratic'])}
    def y2(i,j):
        return quad[min(i,j),max(i,j)]
    gt = transpose(g)
    lower_eigenvalue(matmul(gt,g),F(1,2))
    def transform(y,objective=False):
        return matmul(matmul(gt,dual_matrix(y,objective)),g)
    z0 = transform(y0,True)
    z1 = [transform(y) for y in linear]
    z2 = {key:transform(y) for key,y in quad.items()}
    def second(i,j):
        return z2[min(i,j),max(i,j)]
    check(all(z0[k][l].iszero() for k in range(8,N) for l in range(N)), 'constant kernel')
    check(all(z[k][l].iszero() for z in z1 for k,l in product(range(8,N),repeat=2)), 'linear kernel')
    big = [z0[k][:8]+[z1[i][k][l] for i in AXES for l in range(8,N)] for k in range(8)]
    big += [[z1[i][k][l] for k in range(8)]+[second(i,j)[l][m] for j in AXES for m in range(8,N)]
            for i in AXES for l in range(8,N)]
    lower_eigenvalue(big,F(14,100))
    def b0(y):
        return y[0]+sum(y[1:13],ZERO)*CENTER
    def b1(i,y):
        return sum(((y[1+k]+2*y[7+k])*t for k,t in enumerate(TANGENTS[i])),ZERO)
    def b2(i,j,y):
        return sum((3*y[7+k]*TANGENTS[i][k]*TANGENTS[j][k] for k in range(6)),ZERO)
    check(b0(y0)==Q3(2,F(2,3)), 'constant objective')
    check(all((b0(linear[i])+b1(i,y0)).iszero() for i in AXES), 'linear objective')
    negative_q = [[-b0(y2(i,j))-(b1(i,linear[j])+b1(j,linear[i]))*HALF-b2(i,j,y0) for j in AXES] for i in AXES]
    lower_eigenvalue(negative_q,F(28,100))
    pp = range(8)
    e1 = norm_upper(z1[i][k][l] for i in AXES for k,l in product(pp,repeat=2))
    e2 = norm_upper(second(i,j)[k][l] for i,j in product(AXES,repeat=2) for k,l in product(pp,repeat=2))
    off = norm_upper(second(i,j)[k][l] for i,j in product(AXES,repeat=2) for k in pp for l in range(8,N))
    r3 = norm_upper(b1(i,y2(j,k))+b2(i,j,linear[k]) for i,j,k in product(AXES,repeat=3))
    r4 = norm_upper(b2(i,j,y2(k,l)) for i,j,k,l in product(AXES,repeat=4))
    check(e1 <= 11 and e2 <= 5 and off <= F(1,50) and r3 <= 26 and r4 <= 16, 'published remainder bounds')
    check(F(14,100) > R*(e1+off)+R*R*e2,'scaled matrix remainder')
    check(F(28,100) > R*r3+R*R*r4,'objective remainder')
    for k in range(7,79):
        if y0[k].iszero():
            check(all(row[k].iszero() for row in linear) and ldl_pd([[y2(i,j)[k] for j in AXES] for i in AXES]),'inactive sign')
        else:
            v = norm_upper(row[k] for row in linear)
            w = norm_upper(y2(i,j)[k] for i,j in product(AXES,repeat=2))
            check((y0[k]-R*v-R*R*w).sign()>0, 'active sign')
    for i,j in product(AXES,repeat=2):
        dot = sum(a*b for a,b in zip(TANGENTS[i],TANGENTS[j]))
        check(dot == ((2,6,2,6)[i] if i==j else 0),'tangent metric')
    # ||mu - uniform|| >= sqrt(2)*||delta||, so the local proof covers the inner boxes.
    print('Local upper bound verified at |delta| <= 1/125. The two regions cover the full domain.',flush=True)

def check_attainment():
    """Evaluate the explicit Gupta–Pienaar model, including simplex and germ inequalities."""
    one = Q3(1)
    w = [(one,ZERO),(Q3(-HALF),Q3(0,HALF)),(Q3(-HALF),Q3(0,-HALF))]
    h = (Q3(F(1,4),F(1,4)), Q3(F(1,4),-F(1,4)))
    k = (Q3(-HALF),Q3(HALF))
    # The lower-left block of 2(W + lambda*sum E) is the unitary D of Theorem 1.
    def conjugate(z):
        return z[0], -z[1]
    def times(z,t):
        return z[0]*t[0]-z[1]*t[1], z[0]*t[1]+z[1]*t[0]
    lam = Q3(-HALF,HALF)
    phases = ((Q3(0,HALF), Q3(-HALF)), (Q3(0,-HALF),Q3(HALF)),
              (Q3(0,HALF), Q3(-HALF)), (Q3(0,HALF),Q3(HALF)))
    for phase,m,expected in zip(phases,(1,0,1,2),(h,k,h,conjugate(h))):
        check(tuple(v+lam*t for v,t in zip(phase,w[m])) == expected, 'Fourier dual block')
    d = ((h,k),(conjugate(h),h))
    for i,j in product(range(2),repeat=2):
        terms = [times(d[i][n],conjugate(d[j][n])) for n in range(2)]
        check(tuple(sum(v,ZERO) for v in zip(*terms)) == (Q3(int(i==j)),ZERO), 'unitary dual block')
    amplitudes = [((one,ZERO),(ZERO,ZERO)),((ZERO,ZERO),(one,ZERO)),
                  (h,k),((h[0],-h[1]),h)]
    states = []
    for a,b in amplitudes:
        basis = []
        for outcome in range(3):
            row = []
            for r,s in product(range(3),repeat=2):
                u,v = w[(r-outcome)%3],w[(s-outcome)%3]
                row.append((1+a[0]*u[0]-a[1]*u[1]+b[0]*v[0]-b[1]*v[1])/9)
            check(sum(row,ZERO)==one and all(q.sign()>=0 for q in row), 'state in simplex')
            basis.append(row)
        check(all(sum(column,ZERO)==CENTER for column in zip(*basis)), 'equal-weight average state')
        states.append(basis)
    def dot(a,b):
        return sum((p*q for p,q in zip(a,b)),ZERO)
    ensemble = [vec for setting in states for vec in setting]
    for a,b in product(ensemble,repeat=2):
        val = dot(a,b)
        check((val-F(1,12)).sign()>=0 and (F(1,6)-val).sign()>=0, 'germ inequalities')
    def prob(x,y,a,b):
        return 4*dot(states[x][a],states[2+y][b])-CENTER
    score = sum((prob(x,y,a,(a-g)%3)-prob(x,y,a,(a-l)%3)
                 for x,y,g,l in CLASSES for a in range(3)), ZERO)
    check(score==Q3(2,F(2,3)), 'attained score')
    print('Attaining model verified: I3 = 2 + 2/sqrt(3).',flush=True)


if __name__ == '__main__':
    check_attainment()
    check_biased()
    check_outer()
    check_local()
