"""Size of the LFC-calibrated monotonicity test at the boundary (-ln S linear) for all 16 error models
(k in {1/sqrt3,1,1.5,2} x sigma_c in {0,3,5,10%}); true errors = k x bars, offset ~ N(0, sigma_c^2) as assumed.
2000 null replicates per design and model, 20 simulated datasets per design (1820 per model)."""
import json, numpy as np
from multiprocessing import Pool
from isotonic_lfc import R, design, T, null, pval, KS
SC = [0.0, 0.03, 0.05, 0.10]
def job(args):
    k, sc = args; rng = np.random.default_rng(int(1000*k) + int(1000*sc) + 99); rej = n = 0
    for r in R:
        D, _, s = design(r); s = k*s; nl = null(D, s, sc, 2000, rng)
        for _ in range(20):
            y = -rng.uniform(0.05, 0.4)*D + (rng.normal(0, sc) if sc > 0 else 0) + rng.normal(0, s)
            rej += pval(T(D, y, s, sc), nl) < 0.05; n += 1
    return f"{k:.3f}|{sc}", dict(n=n, rate=rej/n)
if __name__ == "__main__":
    with Pool(2) as p: out = dict(p.map(job, [(k, sc) for k in KS for sc in SC]))
    for k in KS: print(f"{k:.2f} " + "  ".join(f"{out[f'{k:.3f}|{sc}']['rate']:.3f}" for sc in SC), flush=True)
    json.dump(out, open("isotonic_lfc_size16.json", "w"), indent=1)
