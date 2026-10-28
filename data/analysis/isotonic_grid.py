"""Monotonicity test of the 91 Polgár photon datasets under a grid of error models:
observation errors = digitised whisker half-width x k (k = 1/sqrt(3): whiskers are SD of 3 replicates and the point a
mean; k = 1: whiskers are the standard error; k = 1.5, 2: whiskers underestimate the scatter), combined with a common
normalisation error of SD sigma_c (0, 3, 5, 10 %, free). Counts of p < 0.05 and Benjamini-Hochberg rejections."""
import json, numpy as np
from multiprocessing import Pool
from isotonic_norm import test as test_norm
from isotonic_test import test as test_fixed

R = json.load(open("polgar_results.json"))
KS = [1/np.sqrt(3), 1.0, 1.5, 2.0]; SC = [0.0, 0.03, 0.05, 0.10, np.inf]

def bh(ps, q=0.05):
    o = np.argsort(ps); m = len(ps); t = 0
    for k, i in enumerate(o, 1):
        if ps[i] <= q*k/m: t = k
    return t

def run(args):
    k, sc = args; rng = np.random.default_rng(int(1000*k) + int(100*min(sc, 9)))
    ps = []
    for r in R:
        D = np.array(r["D"]); y = np.array(r["lnS"]); s = k*np.array(r["sig"])
        ps.append(test_fixed(D, y, s, B=600, rng=rng)[1] if sc == 0 else test_norm(D, y, s, sc, 600, rng)[2])
    ps = np.array(ps)
    return (k, sc), dict(p05=int((ps < .05).sum()), BH=bh(ps))

if __name__ == "__main__":
    jobs = [(k, sc) for k in KS for sc in SC]
    with Pool() as p: res = dict(p.map(run, jobs))
    out = {f"{k:.3f}|{sc}": v for (k, sc), v in res.items()}
    json.dump(out, open("isotonic_grid_results.json", "w"), indent=1)
    print("k \\ sigma_c   " + "  ".join(f"{s:>8}" for s in ["0", "3%", "5%", "10%", "free"]))
    for k in KS:
        print(f"{k:5.2f}        " + "  ".join(f"{res[(k, sc)]['p05']:>3} ({res[(k, sc)]['BH']:>2})" for sc in SC))
