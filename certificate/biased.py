"""Exact non-membership in Q1 for the rational example in Appendix B."""
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def dot(a, b):
    return sum(u*v for u,v in zip(a,b))

def barycentre(weights, vectors):
    return tuple(dot(weights,column) for column in zip(*vectors))

def states(rows):
    require(len(rows)==6, "six response vectors")
    result=[]
    for denominator,*numerators in rows:
        require(denominator>0 and len(numerators)==9 and min(numerators)>=0
                and sum(numerators)==denominator, "simplex condition")
        result.append(tuple(F(n,denominator) for n in numerators))
    return result[:3],result[3:]

def leading_determinants(matrix):
    m=[[F(x) for x in row] for row in matrix]
    det=F(1)
    for k in range(len(m)):
        pivot=m[k][k]
        require(pivot>0, "nonpositive witness pivot")
        det*=pivot
        yield det
        for i in range(k+1,len(m)):
            ratio=m[i][k]/pivot
            for j in range(k+1,len(m)):
                m[i][j]-=ratio*m[k][j]

def check_biased():
    data=json.loads(Path(__file__).with_name("biased.json").read_text())
    A,B=states(data["alice_states"]),states(data["bob_states"])
    H=data["witness"]
    mu=(F(3,5),F(1,5),F(1,5))
    gamma=((F(1,3),)*3,(F(3,8),F(5,16),F(5,16)))
    require(barycentre(mu,A[0])==barycentre(mu,A[1]), "common prior")
    for y in range(2):
        require(barycentre(gamma[y],B[y])==(F(1,9),)*9, "measurement normalisation")
    ensemble=A[0]+A[1]+B[0]+B[1]
    overlaps=[dot(u,v) for u,v in product(ensemble,repeat=2)]
    require(F(1,12)<=min(overlaps) and max(overlaps)<=F(1,6), "germ interval")
    P={(x,y,a,b): mu[a]*gamma[y][b]*(36*dot(A[x][a],B[y][b])-3)
       for x,y,a,b in product(range(2),range(2),range(3),range(3))}
    require(min(P.values())>=0, "probability sign")
    for x,y in product(range(2),repeat=2):
        require(sum(P[x,y,a,b] for a,b in product(range(3),repeat=2))==1, "total probability")
    pa=lambda x,a: sum(P[x,0,a,b] for b in range(3))
    pb=lambda y,b: sum(P[0,y,a,b] for a in range(3))
    for x,y,a in product(range(2),range(2),range(3)):
        require(sum(P[x,y,a,b] for b in range(3))==pa(x,a), "Alice marginal")
        require(sum(P[x,y,b,a] for b in range(3))==pb(y,a), "Bob marginal")
    # None marks an unspecified inner product between distinct local settings.
    m=[[None]*9 for _ in range(9)]
    m[0][0]=1
    for offset, marginal in ((1,pa),(5,pb)):
        for t in range(2):
            first=offset+2*t
            for a in range(2):
                i=first+a
                m[0][i]=m[i][0]=m[i][i]=marginal(t,a)
            m[first][first+1]=m[first+1][first]=0
    for x,y,a,b in product(range(2),repeat=4):
        i,j=1+2*x+a,5+2*y+b
        m[i][j]=m[j][i]=P[x,y,a,b]
    require(len(H)==9 and all(len(row)==9 for row in H), "witness dimensions")
    for i,j in product(range(9),repeat=2):
        require(H[i][j]==H[j][i] and (m[i][j] is not None or H[i][j]==0), "witness support")
    determinants=tuple(leading_determinants(H))
    require(len(determinants)==9 and min(determinants)>0, "positive definite witness")
    pairing=sum(H[i][j]*m[i][j] for i,j in product(range(9),repeat=2) if m[i][j] is not None)
    require(pairing==-F(1189,900), "separation from Q1")
    classes=((0,0,0,2),(1,0,2,0),(1,1,0,2),(0,1,0,1))
    score=sum(P[x,y,a,(a-g)%3]-P[x,y,a,(a-l)%3] for x,y,g,l in classes for a in range(3))
    require(score==F(391,144), "CGLMP score")
    print("Biased rational model verified: I3 = 391/144; Q1 witness pairing = -1189/900.",flush=True)

if __name__=="__main__":
    check_biased()
