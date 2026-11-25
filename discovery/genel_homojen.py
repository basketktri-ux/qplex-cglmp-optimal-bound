# Agirliklari vektorlerin icine gomen TEK SDP (genel durum icin ust sinir):
#   a'_{x,a} = mu^x_a (1, sqrt2 A^x_a),   b'_{y,b} = gam^y_b (1, sqrt2 B^y_b),   e0 = (1, 0)
#   p(a,b|x,y) = a'.b'      qplex ikili kosullari (>= -1/2)  <=>  tum ikili ic carpimlar >= 0
#   sum_b b'_{y,b} = e0,   sum_a a'_{0,a} = sum_a a'_{1,a} = sigma',  sigma'.e0 = 1
#   (|A|,|B| <= 1 kosulu yerine blok toplamlarindan cikan |w|^2 <= w.(blok toplami) kullanilir; SIC pozitifligi atilir)
import numpy as np
import cvxpy as cp

PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
E0 = 0
iA = lambda x, a: 1 + 3*x + a
iB = lambda y, b: 7 + 3*y + b

def build(with_cone=False):
    X = cp.Variable((13, 13), symmetric=True)
    cons = [X >> 0, X[E0, E0] == 1]
    cons += [X[i, j] >= 0 for i in range(13) for j in range(i+1, 13)]
    for y in range(2):                      # sum_b b'_{y,b} - e0 = 0
        v = np.zeros(13); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1
        cons.append(X @ v == 0)
    v = np.zeros(13); v[[iA(0, a) for a in range(3)]] = 1; v[[iA(1, a) for a in range(3)]] = -1
    cons.append(X @ v == 0)                 # Alice iki ayarda ayni sigma'
    cons.append(sum(X[iA(0, a), E0] for a in range(3)) == 1)
    obj = 0
    for x, y, g, l in PAIRS:
        for a in range(3):
            obj += X[iA(x, a), iB(y, (a-g) % 3)] - X[iA(x, a), iB(y, (a-l) % 3)]
    return X, cons, obj

if __name__ == "__main__":
    X, cons, obj = build()
    pr = cp.Problem(cp.Maximize(obj), cons)
    pr.solve(solver=cp.CLARABEL)
    print("homojen SDP ust siniri:", pr.value, "  2+2/sqrt3 =", 2 + 2/np.sqrt(3))
    Xv = X.value
    print("agirliklar (w.e0):", np.round(Xv[E0, 1:], 4).tolist())
    print("normlar |w|^2     :", np.round(np.diag(Xv)[1:], 4).tolist())
