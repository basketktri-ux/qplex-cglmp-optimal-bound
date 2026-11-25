# GENEL DURUM icin tek SDP: agirlik-momentli, O(8)-simetrize Shor gevsetmesi.
# Homojen vektorler: a~_{x,a} = (mu, sqrt2 mu A),  b~_{y,b} = (gam, sqrt2 gam B);  w = 12 agirlik, perp = Bloch kisimlari.
#   p(a,b|x,y) = a0*b0 + a~perp . b~perp
#   Degiskenler:  M = [[1, m^T],[m, W]] >= 0  (agirliklarin 1. ve 2. momentleri),   Gam = perp Gram >= 0
#   Gecerli kisitlar (her gercek konfigurasyon m=w, W=ww^T, Gam=Gram ile saglar):
#     agirliklar >= 0, blok toplamlari 1 (+ RLT: sum_blok W_ij = m_j),
#     Bob: Gam * 1_blok = 0 ;  Alice: Gam (1_A0 - 1_A1) = 0
#     qplex ikili:  W_ij + Gam_ij >= 0  (i != j)
#     koni (|A|<=1) RLT:  2 W_ij - Gam_ij >= 0  (tum i,j; i=j saflik)
import numpy as np
import cvxpy as cp

PAIRS = ((0, 0, 0, 2), (1, 0, 2, 0), (1, 1, 0, 2), (0, 1, 0, 1))
iA = lambda x, a: 3*x + a
iB = lambda y, b: 6 + 3*y + b
BLOCKS = [[iA(0, a) for a in range(3)], [iA(1, a) for a in range(3)],
          [iB(0, b) for b in range(3)], [iB(1, b) for b in range(3)]]

def build(extra_rlt=True):
    M = cp.Variable((13, 13), symmetric=True)
    Gm = cp.Variable((12, 12), symmetric=True)
    m, W = M[0, 1:], M[1:, 1:]
    cons = [M >> 0, Gm >> 0, M[0, 0] == 1, m >= 0, W >= 0]
    for blk in BLOCKS:
        cons.append(sum(m[i] for i in blk) == 1)
        if extra_rlt:
            for j in range(12):
                cons.append(sum(W[i, j] for i in blk) == m[j])
    for blk in BLOCKS[2:]:
        v = np.zeros(12); v[blk] = 1
        cons.append(Gm @ v == 0)
    v = np.zeros(12); v[BLOCKS[0]] = 1; v[BLOCKS[1]] = -1
    cons.append(Gm @ v == 0)
    for i in range(12):
        for j in range(12):
            if i < j:
                cons.append(W[i, j] + Gm[i, j] >= 0)
            if i <= j:
                cons.append(2*W[i, j] - Gm[i, j] >= 0)
    P = lambda x, y, a, b: W[iA(x, a), iB(y, b)] + Gm[iA(x, a), iB(y, b)]
    obj = 0
    for x, y, g, l in PAIRS:
        for a in range(3):
            obj += P(x, y, a, (a-g) % 3) - P(x, y, a, (a-l) % 3)
    return M, Gm, cons, obj

if __name__ == "__main__":
    M, Gm, cons, obj = build()
    pr = cp.Problem(cp.Maximize(obj), cons)
    pr.solve(solver=cp.CLARABEL)
    print("Shor gevsetmesi (genel agirliklar):", pr.value, "   2+2/sqrt3 =", 2 + 2/np.sqrt(3), "  durum:", pr.status)
    print("agirlik 1. momentleri m:", np.round(M.value[0, 1:], 4).tolist())
