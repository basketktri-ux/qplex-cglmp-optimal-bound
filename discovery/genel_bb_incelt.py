# Tam dogrulamada basarisiz kutulari bol ve yeniden dogrula (ic Oklid topu 0.008 yerel argumana birakilir).
import pickle, sys
import numpy as np
import genel_bb as B
import genel_bb_dogrula as D

RHO_IN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
fails = pickle.load(open("bb_basarisiz.pkl", "rb"))
pr, Pmu, Pmu2, vs = D.build_margin()
queue = [(lo, hi) for lo, hi, _ in fails]
ok, inner, rounds = 0, [], 0
while queue and rounds < 20000:
    lo, hi = queue.pop(); rounds += 1
    if max(np.linalg.norm(v - 1/3) for v in B.box_vertices(lo, hi)) <= RHO_IN:
        inner.append((lo, hi)); continue
    good, why = D.verify_box(pr, Pmu, Pmu2, vs, lo, hi)
    if good:
        ok += 1; continue
    k = int(np.argmax(hi - lo)); mid = (lo[k] + hi[k])/2
    h1 = hi.copy(); h1[k] = mid; l2 = lo.copy(); l2[k] = mid
    queue += [(lo, h1), (l2, hi)]
print(f"inceltme: {ok} alt kutu tam dogrulandi, {len(inner)} alt kutu ic topta, acik {len(queue)}, adim {rounds}")
pickle.dump({"ok": ok, "inner": inner, "open": queue}, open("bb_incelt.pkl", "wb"))
