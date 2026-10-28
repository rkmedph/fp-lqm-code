"""Sensitivity of the monotonicity test to the control normalisation.
Observed ln S_i = ln S_true(D_i) + l, with l = ln(c) the (unknown) error of normalising to the unirradiated control.
Then G_true(D) = (-ln S_obs + l)/D, and a negative l adds a decreasing term |l|/D that can mimic hypersensitivity.
Statistic: T = min_l [ isotonic RSS of G(l) + (l/sigma_c)^2 ], sigma_c = prior SD of the normalisation error
(inf = free nuisance parameter). RSS(l) is convex in l (squared distance of an affine path to a convex cone), so
Brent minimisation is exact. p-values by parametric bootstrap from the constrained fit."""
import json, sys
import numpy as np
from scipy.optimize import minimize_scalar
from isotonic_test import pava

def T_of(D, lnS, sig, sc):
    w = (D/sig)**2
    def obj(l):
        G = (-lnS + l)/D; f = pava(G, w)
        return float(np.sum(w*(G - f)**2)) + (0.0 if not np.isfinite(sc) else (l/sc)**2), f
    r = minimize_scalar(lambda l: obj(l)[0], bounds=(-1.0, 1.0), method="bounded", options=dict(xatol=1e-5))
    T, f = obj(r.x)
    return T, r.x, f

def test(D, lnS, sig, sc, B, rng):
    o = np.argsort(D); D, lnS, sig = D[o], lnS[o], sig[o]
    T, l, f = T_of(D, lnS, sig, sc)
    # H0 model: ln S_obs = -D*G(D) + l + eps, with G non-decreasing, eps independent per point and the normalisation
    # offset l common to all doses (fully correlated), l ~ N(0, sc^2): equivalent to GLS with covariance
    # diag(sig^2) + sc^2 * 1 1^T.  The offset is redrawn in every replicate (free: offset irrelevant, T is invariant).
    base = -D*f
    sims = [T_of(D, base + (rng.normal(0, sc) if np.isfinite(sc) else l) + rng.normal(0, sig), sig, sc)[0] for _ in range(B)]
    return T, l, float((1 + np.sum(np.array(sims) >= T))/(B + 1))

if __name__ == "__main__":
    R = json.load(open("polgar_results.json")); B = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.default_rng(7); out = {}
    for sc in (0.0, 0.03, 0.05, 0.10, np.inf):
        ps = []; ls = []
        for r in R:
            D = np.array(r["D"]); lnS = np.array(r["lnS"]); s = np.array(r["sig"])
            if sc == 0.0:
                from isotonic_test import test as t0
                T, p = t0(D, lnS, s, B=B, rng=rng); l = 0.0
            else:
                T, l, p = test(D, lnS, s, sc, B, rng)
            ps.append(p); ls.append(l)
        ps = np.array(ps); ls = np.array(ls)
        o = np.argsort(ps); m = len(ps); bh = 0
        for k, i in enumerate(o, 1):
            if ps[i] <= 0.05*k/m: bh = k
        key = "fixed (paper)" if sc == 0 else ("free" if not np.isfinite(sc) else f"prior SD {sc:.2f}")
        out[key] = dict(p_lt_05=int((ps < .05).sum()), p_lt_01=int((ps < .01).sum()), BH=bh, median_l=float(np.median(ls)),
                        frac_l_neg=float(np.mean(ls < 0)), p=ps.tolist(), l=ls.tolist())
        print(f"normalisation {key:14s}: p<0.05 {out[key]['p_lt_05']:2d}, p<0.01 {out[key]['p_lt_01']:2d}, BH {bh:2d}; median fitted ln c = {np.median(ls):+.3f}", flush=True)
    json.dump(out, open("isotonic_norm_results.json", "w"), indent=1)
