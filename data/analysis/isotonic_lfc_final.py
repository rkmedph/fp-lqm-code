"""Final LFC-calibrated monotonicity test: 10000 null replicates per dataset and error model, independent seeds from
the 2000-replicate run (isotonic_lfc_results.json). Reports the final grid and the stability of individual decisions."""
import json, numpy as np
from multiprocessing import Pool
from isotonic_lfc import R, design, T, null, pval, bh, KS
SC = [0.0, 0.03, 0.05, 0.10]; B = 10000

def job(args):
    i, k, sc = args; D, y, s = design(R[i]); rng = np.random.default_rng(7_000_000 + 1000*i + int(100*k) + int(1000*sc))
    if sc == 0.0:   # T scales as 1/k^2 and its null distribution is invariant: one null serves all k
        nl = null(D, s, 0.0, B, rng); return [((i, kk, sc), float(pval(T(D, y, s, 0.0)/kk**2, nl))) for kk in KS]
    nl = null(D, k*s, sc, B, rng); return [((i, k, sc), float(pval(T(D, y, k*s, sc), nl)))]

if __name__ == "__main__":
    jobs = [(i, 1.0, 0.0) for i in range(len(R))] + [(i, k, sc) for i in range(len(R)) for k in KS for sc in SC[1:]]
    with Pool(2) as p: res = dict(x for out in p.map(job, jobs, chunksize=4) for x in out)
    old = json.load(open("isotonic_lfc_results.json"))["grid"]
    grid = {}
    for k in KS:
        for sc in SC:
            ps = np.array([res[(i, k, sc)] for i in range(len(R))]); po = np.array(old[f"{k:.3f}|{sc}"]["p"])
            se = np.sqrt(0.05*0.95/B)
            grid[f"{k:.3f}|{sc}"] = dict(p05=int((ps < .05).sum()), p01=int((ps < .01).sum()), BH=bh(ps), p=ps.tolist(),
                                       p05_B2000=int((po < .05).sum()), decisions_changed=int(((ps < .05) != (po < .05)).sum()),
                                       near_threshold=int((np.abs(ps - 0.05) < 3*se).sum()),
                                       max_abs_dp=float(np.max(np.abs(ps - po))))
    for k in KS:
        print(f"{k:5.2f} " + "  ".join(f"{grid[f'{k:.3f}|{sc}']['p05']:>3} ({grid[f'{k:.3f}|{sc}']['BH']:>2}) [B2000 {grid[f'{k:.3f}|{sc}']['p05_B2000']}, changed {grid[f'{k:.3f}|{sc}']['decisions_changed']}, near {grid[f'{k:.3f}|{sc}']['near_threshold']}]" for sc in SC), flush=True)
    json.dump(dict(B=B, grid=grid), open("isotonic_lfc_final.json", "w"), indent=1)
