# Boyut indirgeme: Bob homojen (agirliklari vektorlerin icinde, saflik kisiti YOK), Alice agirliklari mu sabit.
#   A^_{x,a} = (1, sqrt2 A),  b~_{y,b} = gam (1, sqrt2 B),  e0 = (1,0)
#   p(a,b|x,y) = mu^x_a * (A^_{x,a} . b~_{y,b})
# Psi(mu) = bu SDP'nin degeri; her gercek konfigurasyon icin I3 <= Psi(mu). Hedef: max_mu Psi(mu) = 2+2/sqrt3.
import sys
import numpy as np
import cvxpy as cp

PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
E0 = 0
iA = lambda x, a: 1 + 3*x + a
iB = lambda y, b: 7 + 3*y + b

def psi(mu, return_problem=False):
    X = cp.Variable((13, 13), symmetric=True)
    cons = [X >> 0, X[E0, E0] == 1]
    for x in range(2):
        for a in range(3):
            i = iA(x, a)
            cons += [X[i, E0] == 1, X[i, i] <= 3]
    cons += [X[i, j] >= 0 for i in range(1, 13) for j in range(i+1, 13)]
    for y in range(2):
        v = np.zeros(13); v[[iB(y, b) for b in range(3)]] = 1; v[E0] = -1
        cons.append(X @ v == 0)
    v = np.zeros(13)
    for a in range(3):
        v[iA(0, a)] = mu[0][a]; v[iA(1, a)] = -mu[1][a]
    cons.append(X @ v == 0)
    obj = 0
    for x, y, g, l in PAIRS:
        for a in range(3):
            obj += mu[x][a] * (X[iA(x, a), iB(y, (a-g) % 3)] - X[iA(x, a), iB(y, (a-l) % 3)])
    pr = cp.Problem(cp.Maximize(obj), cons)
    pr.solve(solver=cp.CLARABEL)
    return (pr, X, cons) if return_problem else pr.value

def mu_of(t):
    t = np.asarray(t).reshape(2, 3)
    w = np.exp(t - t.max(1, keepdims=True)); return w / w.sum(1, keepdims=True)

if __name__ == "__main__":
    flat = np.full((2, 3), 1/3)
    print("Psi(duz) =", psi(flat), "  2+2/sqrt3 =", 2 + 2/np.sqrt(3))
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    vals = []
    for t in range(int(sys.argv[2]) if len(sys.argv) > 2 else 300):
        th = rng.standard_normal(6) * rng.choice([0.1, 0.5, 1.5, 4.0])
        vals.append((psi(mu_of(th)), th))
    vals.sort(key=lambda v: -v[0])
    print("rastgele mu, en buyuk 6 Psi:", [round(v[0], 6) for v in vals[:6]])
    from scipy.optimize import minimize
    for k in range(3):
        r = minimize(lambda th: -psi(mu_of(th)), vals[k][1], method='Nelder-Mead',
                     options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-11})
        print(f"NM {k}: Psi = {-r.fun:.8f}   mu = {mu_of(r.x).round(4).tolist()}")
