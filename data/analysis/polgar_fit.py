"""Low-dose test on Polgár et al 2022 (101 datasets with HRS/IRR; photons only).
Models fitted to -ln S with sigma(ln S) = half-width of the digitised whiskers / S (floor 1 %):
  LQ (2 par), induced-repair IR (4), LQ two-population mixture (4),
  exact homogeneous threshold model, M = 1, 2, 4 (3 par + n_c),
  exact model + hypersensitive fraction, M = 1 (5 par + n_c).
Model-free test: for a homogeneous population of our class -ln S is convex, so -ln S / D is non-decreasing;
a significantly negative slope of -ln S / D on D <= 1 Gy is incompatible with any homogeneous model of the class.
Output: polgar_results.json
"""
import sys, json, warnings
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, ".")
from discrete import survival_disc, ab_disc
from polgar_load import load

PHOTON = ("X-ray", "60Co γ radiation", "60Co γ-radiation", "137Cs γ-ray")
NC = range(2, 26)


def prep(r):
    D, S, lo, hi = r["D"], r["S"], r["Smin"], r["Smax"]
    m = (D > 0.005) & (S > 0)
    D, S, lo, hi = D[m], S[m], lo[m], hi[m]
    half = np.where(np.isfinite(lo) & np.isfinite(hi), (hi - lo)/2, np.nan)
    s = half/S
    s = np.where(np.isfinite(s), s, np.nan)
    if np.all(np.isnan(s)): s = np.full_like(S, 0.05)   # no whiskers: 5 % assumed
    s = np.where(np.isnan(s), np.nanmedian(s), s)
    s = np.maximum(s, 0.01)
    return D, -np.log(S), s


def best_of(f, starts, lb, ub):
    best = None
    for p0 in starts:
        try:
            r = least_squares(f, p0, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    return best


def fits(args):
    rid, D, y, s = args
    out = {}
    # LQ
    X = np.vstack([D, D*D]).T/s[:, None]; c, *_ = np.linalg.lstsq(X, y/s, rcond=None); r = y/s - X @ c
    out["LQ"] = dict(chi2=float(r @ r), k=2, par=c.tolist())
    # IR: alpha_r (1 + (alpha_s/alpha_r - 1) exp(-D/Dc)) D + beta D^2
    f = lambda p: (p[0]*(1 + (p[1]/p[0] - 1)*np.exp(-D/p[2]))*D + p[3]*D*D - y)/s
    b = best_of(f, [[0.2, 1.0, 0.2, 0.03], [0.1, 3.0, 0.1, 0.0], [0.3, 0.5, 0.5, 0.05]], [1e-4, 0, 0.005, 0], [5, 50, 10, 1])
    out["IR"] = dict(chi2=2*b.cost, k=4, par=b.x.tolist())
    # LQ mixture: resistant (1-f) LQ + sensitive f exp(-a_h D)
    f = lambda p: (-np.log((1 - p[3])*np.exp(-p[0]*D - p[1]*D*D) + p[3]*np.exp(-p[2]*D)) - y)/s
    b = best_of(f, [[0.1, 0.03, 5, 0.1], [0.2, 0.0, 20, 0.05], [0.05, 0.05, 2, 0.3]], [0, 0, 0, 0], [5, 1, 200, 0.99])
    out["LQ mixture"] = dict(chi2=2*b.cost, k=4, par=b.x.tolist())
    # exact homogeneous model
    lb = np.log([1e-3, 1e-6, 1e-5]); ub = np.log([200, 1e4, 1e4])
    for M in (1, 2, 4):
        best = None
        for nc in NC:
            f = lambda p: (-np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1.0, 2*(0.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y)/s
            b = best_of(f, [[np.log(0.5/M), np.log(0.5), np.log(0.3)], [np.log(2/M), np.log(3), np.log(1)], [np.log(1/M), np.log(.05), np.log(.01)]], lb, ub)
            if best is None or b.cost < best[0].cost: best = (b, nc)
        b, nc = best; lam, Fm, k = np.exp(b.x)
        a0, b0 = ab_disc(lam, 1.0, 2*(0.5 + Fm), k, nc, M)
        out[f"exact M={M}"] = dict(chi2=2*b.cost, k=4, nc=nc, par=[lam, 0.5 + Fm, k], alpha0=a0, beta0=b0)
    # exact model (M=1) + hypersensitive fraction
    best = None
    for nc in NC:
        def f(p):
            S0 = survival_disc(D, np.exp(p[0]), 1.0, 2*(0.5 + np.exp(p[1])), np.exp(p[2]), nc, 1)
            return (-np.log(np.maximum((1 - p[4])*S0 + p[4]*np.exp(-p[3]*D), 1e-300)) - y)/s
        b = best_of(f, [[np.log(0.5), np.log(0.5), np.log(0.3), 5, 0.1], [np.log(2), np.log(3), np.log(1), 20, 0.05]],
                    np.r_[lb, 0, 0], np.r_[ub, 200, 0.99])
        if best is None or b.cost < best[0].cost: best = (b, nc)
    b, nc = best
    out["exact M=1 + sensitive fraction"] = dict(chi2=2*b.cost, k=6, nc=nc, par=[float(np.exp(b.x[0])), float(0.5 + np.exp(b.x[1])), float(np.exp(b.x[2])), float(b.x[3]), float(b.x[4])])
    # model-free slope test of g = -lnS/D on D <= 1 Gy
    m = D <= 1.0
    if m.sum() >= 3:
        g = y[m]/D[m]; sg = s[m]/D[m]; X = np.vstack([np.ones(m.sum()), D[m]]).T/sg[:, None]
        cov = np.linalg.inv(X.T @ X); c = cov @ X.T @ (g/sg)
        out["slope_test"] = dict(n=int(m.sum()), slope=float(c[1]), se=float(np.sqrt(cov[1, 1])), z=float(c[1]/np.sqrt(cov[1, 1])))
    return rid, out


def main():
    R = [r for r in load() if r["radiation"] in PHOTON]
    jobs = []
    for r in R:
        D, y, s = prep(r); jobs.append((r["id"], D, y, s))
    with Pool() as p:
        res = dict(p.map(fits, jobs))
    meta = []
    for r, (rid, D, y, s) in zip(R, jobs):
        meta.append(dict(id=r["id"], cell=r["cell"].strip(), cell_info=r["cell_info"], radiation=r["radiation"], radiation_info=r["radiation_info"],
                         authors=r["authors"], year=r["year"], doi=r["doi"], figure=r["figure"], n=len(D), Dmin=float(D.min()), Dmax=float(D.max()),
                         D=D.tolist(), lnS=(-y).tolist(), sig=s.tolist(), fits_in_original=r["fits_in_original"], fits=res[rid]))
    json.dump(meta, open("polgar_results.json", "w"), indent=1, default=float)
    print(len(meta), "photon datasets fitted")


if __name__ == "__main__":
    main()
