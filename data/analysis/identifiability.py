"""Likelihood profiles for the exact model on the densest available curves + an ideal synthetic curve.
Errors are rescaled so that LQ has chi2/dof = 1 (honest scale). For M in {1,2}, n_c in 2..30 and a grid in kappa/mu
(resp. F_r), the remaining continuous parameters are optimised. Profiles of derived quantities (alpha0, beta0,
alpha0/beta0, lambda_d) are lower envelopes over all optimised fits."""
import sys, json
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from discrete import survival_disc, ab_disc
from validation import lq_log

LN10 = np.log(10)
NC = list(range(2, 31))
KG = list(np.logspace(-4, 3, 15))
FG = list(0.5 + np.logspace(-5, 3, 15))


def datasets():
    H = {c["cell"]: c for c in json.load(open("/home/claude/rawdata/analysis/digitize/howard_curves.json"))}
    out = {}
    for cell in ("A549", "CHO"):
        c = H[cell]; out[f"Howard2018 {cell} 6MV"] = (np.array([p["D"] for p in c["points"]]), -np.log([p["SF"] for p in c["points"]]),
                                                      np.array([p["sigma_lnSF"] for p in c["points"]]))
    from energy_data import VERBAKEL
    p = np.array(VERBAKEL[("D384", "6 MV FF")])
    out["Verbakel2013 D384 6MV"] = (p[:, 0], -LN10*p[:, 1], LN10*np.hypot(np.maximum(p[:, 2], .05), .03))
    D = np.array([0.90, 1.82, 3.22, 5.00, 6.62, 8.34, 9.91, 11.44]); S = np.array([0.77, 0.60, 0.275, 0.085, 0.0193, 0.0061, 0.00182, 0.00051])
    out["Elkind-Sutton 1959"] = (D, -np.log(S), np.full(8, 0.10))
    for k, (D, y, s) in list(out.items()):           # honest error scale
        a, b, cov, c2 = lq_log(D, y, s); sc = max(1.0, np.sqrt(c2/(len(D) - 2)))
        out[k] = (D, y, s*sc)
    return out


def g(D, lam, F, k, nc, M):
    return -np.log(np.maximum(survival_disc(D, lam, 1.0, 2*F, k, nc, M), 1e-300))


def fit_fixed(args):
    D, y, s, M, nc, which, val = args
    best = None
    if which == "kappa":
        f = lambda p: (g(D, np.exp(p[0]), .5 + np.exp(p[1]), val, nc, M) - y)/s
        starts = [[np.log(.5/M), np.log(.5)], [np.log(2/M), np.log(3)], [np.log(1/M), np.log(.01)], [np.log(5/M), np.log(30)]]
        lb, ub = np.log([1e-3, 1e-6]), np.log([300, 1e4])
    else:
        f = lambda p: (g(D, np.exp(p[0]), val, np.exp(p[1]), nc, M) - y)/s
        starts = [[np.log(.5/M), np.log(.3)], [np.log(2/M), np.log(1)], [np.log(1/M), np.log(.01)], [np.log(5/M), np.log(30)]]
        lb, ub = np.log([1e-3, 1e-6]), np.log([300, 1e4])
    for p0 in starts:
        r = least_squares(f, p0, bounds=(lb, ub))
        if best is None or r.cost < best.cost: best = r
    if which == "kappa": lam, F, k = np.exp(best.x[0]), .5 + np.exp(best.x[1]), val
    else: lam, F, k = np.exp(best.x[0]), val, np.exp(best.x[1])
    a0, b0 = ab_disc(lam, 1.0, 2*F, k, nc, M)
    return dict(M=M, nc=nc, which=which, lam=lam, F=F, k=k, chi2=2*best.cost, alpha0=a0, beta0=b0)


def run():
    DS = datasets()
    # ideal synthetic curve generated from the best Howard A549 fit (computed below) is added after the first pass
    res = {}
    with Pool() as pool:
        for name, (D, y, s) in DS.items():
            jobs = [(D, y, s, M, nc, "kappa", k) for M in (1, 2) for nc in NC for k in KG] + \
                   [(D, y, s, M, nc, "F", F) for M in (1, 2) for nc in NC for F in FG]
            fits = pool.map(fit_fixed, jobs, chunksize=8)
            a, b, cov, c2 = lq_log(D, y, s)
            res[name] = dict(D=D.tolist(), y=y.tolist(), s=s.tolist(), chi2_LQ=c2, fits=fits)
            best = min(fits, key=lambda f: f["chi2"]); print(name, "LQ chi2 %.2f" % c2, "best exact %.2f" % best["chi2"], best, flush=True)
        # ideal data: 12 doses 0.1-10 Gy, 3 % error, noise-free, from the best A549 fit
        b = min(res["Howard2018 A549 6MV"]["fits"], key=lambda f: f["chi2"])
        D = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10.]); y = g(D, b["lam"], b["F"], b["k"], b["nc"], b["M"]); s = np.full(len(D), 0.03)
        jobs = [(D, y, s, M, nc, "kappa", k) for M in (1, 2) for nc in NC for k in KG] + [(D, y, s, M, nc, "F", F) for M in (1, 2) for nc in NC for F in FG]
        fits = pool.map(fit_fixed, jobs, chunksize=8)
        res["Ideal synthetic (A549 truth, 12 doses 0.1-10 Gy, 3 %)"] = dict(D=D.tolist(), y=y.tolist(), s=s.tolist(), chi2_LQ=lq_log(D, y, s)[3], fits=fits, truth=b)
    json.dump(res, open("identifiability_results.json", "w"), default=float)


if __name__ == "__main__":
    run()
