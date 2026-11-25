# Duz sertifikanin genel agirliklarda verdigi esitsizlik:
#   (sqrt3/2) I3 <= sqrt3 + |sigma'|^2 - (3/2)(offA + offB)  , |sigma'|^2 = 1 + 2|S|^2
#   offA = sum_x sum_{a<a'} mu mu' (1 + 2 A.A'),  offB = sum_y sum_{b<b'} gam gam' (1 + 2 B.B')
# Yeniden olceklemeyle (t): (sqrt3/2) I3 <= sqrt(N_A N_B) + sqrt3 - 1,  N_A = 2|sigma'|^2 - 3 offA,  N_B = 2 - 3 offB
# Bu betik, gercek qplex konfigurasyonlarinda (SIC koordinatlari) bu sinirlarin I3'u gercekten ustten sinirladigini
# ve 2+2/sqrt3'e gore ne kadar acik kaldigini olcer.
import sys, numpy as np
import genel_qplex as G

def bloch(p):            # SIC olasiligi -> Bloch (|n|<=1, <p,q>_Q = (1+2 n.n')/3)
    return np.sqrt(18) * (p - 1/9)

def bounds(z):
    U, V, mu, gam = G.unpack(z)
    A = bloch(U); B = bloch(V)
    S = mu[0] @ A[0]
    offA = sum(mu[x][a]*mu[x][b]*(1 + 2*A[x][a] @ A[x][b]) for x in range(2) for a in range(3) for b in range(a+1, 3))
    offB = sum(gam[y][a]*gam[y][b]*(1 + 2*B[y][a] @ B[y][b]) for y in range(2) for a in range(3) for b in range(a+1, 3))
    sig2 = 1 + 2*S @ S
    b1 = (2/np.sqrt(3)) * (np.sqrt(3) + sig2 - 1.5*(offA + offB))
    NA, NB = 2*sig2 - 3*offA, 2 - 3*offB
    b2 = (2/np.sqrt(3)) * (np.sqrt(max(NA, 0)*max(NB, 0)) + np.sqrt(3) - 1)
    return G.I3(G.box(z)), b1, b2, S @ S, offA, offB

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    T = 2 + 2/np.sqrt(3)
    rows = []
    for t in range(n):
        z, val, feas = G.solve(G.random_start(rng, rng.choice([0.02, 0.05, 0.1, 0.2])))
        if feas < 1e-7:
            rows.append(bounds(z))
    rows = np.array(rows)
    print("konfig sayisi:", len(rows))
    print("sinir gecerli mi (I3 <= b1, I3 <= b2):", bool(np.all(rows[:, 0] <= rows[:, 1] + 1e-7)), bool(np.all(rows[:, 0] <= rows[:, 2] + 1e-7)))
    ok1 = rows[:, 1] <= T + 1e-9; ok2 = rows[:, 2] <= T + 1e-9
    print(f"duz sertifika 2+2/sqrt3'u kanitliyor: b1 ile {ok1.sum()}/{len(rows)}, b2 ile {ok2.sum()}/{len(rows)}")
    for r in rows[np.argsort(-rows[:, 0])][:8]:
        print(f"  I3={r[0]:.4f}  b1={r[1]:.4f}  b2={r[2]:.4f}  |S|^2={r[3]:.4f}  offA={r[4]:.4f}  offB={r[5]:.4f}")
