"""Size of the LFC-calibrated test exactly at the boundary g''=0 (-ln S linear in D), 20 replicates of each of the
91 designs per normalization model (1820 datasets, Monte Carlo SE ~0.5 %)."""
import json, numpy as np
from multiprocessing import Pool
from isotonic_lfc import R, design, T, null, pval
def job(sc):
    rng = np.random.default_rng(int(1000*sc) + 5); rej = n = 0
    for i, r in enumerate(R):
        D, _, s = design(r); nl = null(D, s, sc, 2000, rng)
        for _ in range(20):
            y = -rng.uniform(0.05, 0.4)*D + (rng.normal(0, sc) if sc > 0 else 0) + rng.normal(0, s)
            rej += pval(T(D, y, s, sc), nl) < 0.05; n += 1
    return sc, dict(n=n, rate=rej/n, se=float(np.sqrt(0.05*0.95/n)))
if __name__ == "__main__":
    with Pool(2) as p: out = dict(p.map(job, [0.0, 0.03, 0.05, 0.10]))
    for k, v in out.items(): print(f"sigma_c={k}: size {v['rate']:.3f} (n={v['n']}, MC SE {v['se']:.3f})", flush=True)
    json.dump({str(k): v for k, v in out.items()}, open("isotonic_lfc_boundary.json", "w"), indent=1)
