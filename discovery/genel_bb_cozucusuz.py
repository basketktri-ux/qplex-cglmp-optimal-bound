# Adim 3'un COZUCUSUZ ve tamamen tam aritmetik dogrulamasi + sertifikalarin diske kaydi.
#  1) Kapsama: 852 + 12 kutunun ic kisimlari ayrik ve alan (Delta_2 x Delta_2) ile kesisim hacimleri toplami tam 1/4.
#  2) Yerel topun icindeki kutular: tum koseler mu-uzayinda ||mu - mu_bar||^2 <= rmu2 (genel_yerel_tam4.py'den, tam).
#  3) Diger her kutu: kayitli dual y rasyonele yuvarlanir; eI kaydirmasi
#        y00 += 3e,  zA += e,  u(Bob blok ici) += 2e,  w_y -= e (1_{B_y} + e0)   =>   Z(y) -> Z(y) + e I
#     (bu ozdesligi Gemini 4 Neon'un bagimsiz denetimi onerdi) ile Z(y) > 0 kesin LDL ile, 16 kosede f_y <= T tam olarak.
#  Cikti: bb_sertifikalar_tam.pkl  (yalnizca str/int/list; numpy gerekmez)
import sys, math, pickle, itertools, time
from fractions import Fraction as Fr
import numpy as np
import genel_bb as B

def leq_T(f):                       # f <= 2 + 2/sqrt3  (tam)
    return f <= 2 or 3*(f - 2)**2 <= 4

def Z_exact(y00, yA, zA, u, wy, w):
    N = B.N; h = Fr(1, 2); Z = [[Fr(0)]*N for _ in range(N)]
    Z[0][0] += y00
    for k, i in enumerate(B.ALICE):
        Z[i][0] += yA[k]*h; Z[0][i] += yA[k]*h; Z[i][i] += zA[k]
    for k, (i, j) in enumerate(B.OFF):
        Z[i][j] -= u[k]*h; Z[j][i] -= u[k]*h
    for vec, bv in ((wy[0], B.BV[0]), (wy[1], B.BV[1]), (w, B.CV)):
        for i in range(N):
            for j in range(N):
                if bv[j]: Z[i][j] -= vec[i]*int(bv[j])*h
                if bv[i]: Z[i][j] -= vec[j]*int(bv[i])*h
    for i in range(N):
        for j in range(N):
            if B.CMAT[i, j]: Z[i][j] -= Fr(B.CMAT[i, j]).limit_denominator(4)
    return Z

def ldl_pd(Z):
    n = len(Z); M = [r[:] for r in Z]
    for k in range(n):
        if M[k][k] <= 0: return False
        for i in range(k+1, n):
            f = M[i][k]/M[k][k]
            if f:
                for j in range(k, n): M[i][j] -= f*M[k][j]
    return True

def area_rect_tri(a, b, c, d):      # [a,b]x[c,d] ile {x,y>=0, x+y<=1} kesisiminin alani (tam)
    pts = sorted(set([a, b] + [x for x in (1 - d, 1 - c) if a < x < b]))
    A = Fr(0)
    for x0, x1 in zip(pts, pts[1:]):
        top0, top1 = min(d, 1 - x0), min(d, 1 - x1)
        if max(top0, top1) <= c: continue
        A += (x1 - x0)*((max(top0, c) - c) + (max(top1, c) - c))/2   # 1-x dogrusal, d ve c ile kirpilmis
    return A

def main():
    t0 = time.time()
    bb = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "bb_d0_0.0.pkl", "rb"))
    rmu2 = pickle.load(open("yerel_titiz_sonuc4.pkl", "rb"))["rmu2"]
    assert len(bb["queue"]) == 0
    boxes = [(tuple(Fr(float(x)) for x in lo), tuple(Fr(float(x)) for x in hi)) for lo, hi, _, _ in bb["cert"]]
    boxes += [(tuple(Fr(float(x)) for x in lo), tuple(Fr(float(x)) for x in hi)) for lo, hi in bb["local"]]
    # 1) kapsama
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            (l1, h1), (l2, h2) = boxes[i], boxes[j]
            assert not all(max(l1[k], l2[k]) < min(h1[k], h2[k]) for k in range(4)), "kutular cakisiyor"
    vol = sum(area_rect_tri(l[0], h[0], l[1], h[1])*area_rect_tri(l[2], h[2], l[3], h[3]) for l, h in boxes)
    assert vol == Fr(1, 4), f"hacim {vol} != 1/4"
    print(f"[kapsama] {len(boxes)} kutu, ic kisimlar ayrik, alanla kesisim hacmi tam 1/4", flush=True)
    def verts(l, h):
        out = []
        for bits in itertools.product((0, 1), repeat=4):
            p = [l[i] if b == 0 else h[i] for i, b in enumerate(bits)]
            out.append([p[0], p[1], 1 - p[0] - p[1], p[2], p[3], 1 - p[2] - p[3]])
        return out
    in_ball = lambda l, h: max(sum((m - Fr(1, 3))**2 for m in mu) for mu in verts(l, h)) <= rmu2
    # 2) yerel kutular
    assert all(in_ball(l, h) for l, h in boxes[len(bb["cert"]):]), "yerel kutu topun disinda"
    # 3) dis kutular
    D = 10**9; rq = lambda x: Fr(round(float(x)*D), D)
    bob_intra = [k for k, (i, j) in enumerate(B.OFF) if any(i in [B.iB(y, b) for b in range(3)] and j in [B.iB(y, b) for b in range(3)] for y in range(2))]
    certs, n_loc, n_ex = [], 0, 0
    for (lo, hi, val, duals), (l, h) in zip(bb["cert"], boxes):
        if in_ball(l, h):
            n_loc += 1; continue
        y00, yA, zA, u, wy, w = duals
        y00 = rq(y00); yA = [rq(x) for x in yA]; zA = [max(Fr(0), rq(x)) for x in zA]
        u = [max(Fr(0), rq(x)) for x in u]; wy = [[rq(x) for x in r] for r in wy]; w = [rq(x) for x in w]
        Zf = np.array([[float(x) for x in r] for r in Z_exact(y00, yA, zA, u, wy, w)])
        eps0 = Fr(max(0.0, -np.linalg.eigvalsh(Zf).min())*1.5 + 1e-9).limit_denominator(10**12)
        done = False
        for mult in (1, 2, 5, 20, 100):
            e = eps0*mult
            y00s = y00 + 3*e; zAs = [x + e for x in zA]; us = u[:]
            for k in bob_intra: us[k] += 2*e
            wys = [r[:] for r in wy]
            for y in range(2):
                wys[y][0] -= e
                for b in range(3): wys[y][B.iB(y, b)] -= e
            if not all(leq_T(y00s + sum(yA[k]*mu[k] for k in range(6)) + 3*sum(zAs[k]*mu[k]**2 for k in range(6))) for mu in verts(l, h)):
                break
            if ldl_pd(Z_exact(y00s, yA, zAs, us, wys, w)):
                done = True; break
        assert done, f"kutu dogrulanamadi: {l} {h}"
        n_ex += 1
        certs.append({"lo": [str(x) for x in l], "hi": [str(x) for x in h], "y00": str(y00s), "yA": [str(x) for x in yA],
                      "zA": [str(x) for x in zAs], "u": [str(x) for x in us], "wy": [[str(x) for x in r] for r in wys], "w": [str(x) for x in w]})
    pickle.dump({"rmu2": str(rmu2), "local_boxes": [([str(x) for x in l], [str(x) for x in h]) for l, h in boxes if in_ball(l, h)],
                 "certs": certs}, open("bb_sertifikalar_tam.pkl", "wb"))
    print(f"[Adim 3] {n_ex} kutu cozucusuz tam aritmetikle dogrulandi; {n_loc + len(bb['local'])} kutu yerel topun icinde. "
          f"Sertifikalar: bb_sertifikalar_tam.pkl  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
