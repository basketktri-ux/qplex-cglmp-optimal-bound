# Seviye 1.5: Shor gevsetmesine her agirlik w_k ile LOKALIZE momentler eklenir (O(8)-simetrize).
#   T_k[i,j] = E[w_k w_i w_j]  (tam simetrik 3-tensor),   Gam_k[i,j] = E[w_k z_i.z_j]
#   L_k = [[m_k, W_k.],[W_k.^T, T_k]] >= 0,  Gam_k >= 0
#   blok toplami=1 ile carpim:  sum_{k in blok} T_k = W,  sum_{k in blok} Gam_k = Gam
#   z-toplam kisitlari, ikili ve koni kisitlari w_k ile carpilmis halleriyle.
import numpy as np
import cvxpy as cp
import genel_shor as S

def build():
    M, Gm, cons, obj = S.build()
    m, W = M[0, 1:], M[1:, 1:]
    T = [cp.Variable((12, 12), symmetric=True) for _ in range(12)]
    Gk = [cp.Variable((12, 12), symmetric=True) for _ in range(12)]
    for k in range(12):
        Lk = cp.bmat([[cp.reshape(m[k], (1, 1)), cp.reshape(W[k, :], (1, 12))],
                      [cp.reshape(W[k, :], (12, 1)), T[k]]])
        cons += [Lk >> 0, Gk[k] >> 0, T[k] >= 0]
        for i in range(12):            # tam simetri: T_k[i,j] = T_i[k,j]
            if i > k:
                for j in range(12):
                    cons.append(T[k][i, j] == T[i][k, j])
        for blk in S.BLOCKS[2:]:
            v = np.zeros(12); v[blk] = 1
            cons.append(Gk[k] @ v == 0)
        v = np.zeros(12); v[S.BLOCKS[0]] = 1; v[S.BLOCKS[1]] = -1
        cons.append(Gk[k] @ v == 0)
        for i in range(12):
            for j in range(i, 12):
                if i < j:
                    cons.append(T[k][i, j] + Gk[k][i, j] >= 0)
                cons.append(2*T[k][i, j] - Gk[k][i, j] >= 0)
    for blk in S.BLOCKS:
        cons.append(sum(T[k] for k in blk) == W)
        cons.append(sum(Gk[k] for k in blk) == Gm)
    return M, Gm, T, Gk, cons, obj

if __name__ == "__main__":
    M, Gm, T, Gk, cons, obj = build()
    pr = cp.Problem(cp.Maximize(obj), cons)
    pr.solve(solver=cp.CLARABEL)
    print("Seviye 1.5 gevsetmesi:", pr.value, "  2+2/sqrt3 =", 2 + 2/np.sqrt(3), "  durum:", pr.status)
