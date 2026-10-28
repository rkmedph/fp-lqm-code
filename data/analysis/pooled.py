"""Pooled test on Li et al 2024 (4 beams per cell line):
does the beam quality act only as a rescaling of the lesion yield (lambda_d), i.e. S_b(D) = S_ref(r_b D)?
In the exact model this is the statement 'only lambda_d depends on beam quality'."""
import sys, json
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2dist
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc, ab_disc
from registry import build, sigma_lnS

FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # extra relative error on SF (added in quadrature)
BEAMS = ["6 MV", "192Ir", "225 kVp", "50 kVp"]


def data(cell):
    cs = {c["beam"]: c for c in build() if c["src"] == "LI24" and c["cell"] == cell}
    D = np.concatenate([cs[b]["D"] for b in BEAMS]); y = np.concatenate([-np.log(cs[b]["S"]) for b in BEAMS])
    s = np.concatenate([sigma_lnS(cs[b]) for b in BEAMS]); s = np.sqrt(s**2 + FLOOR**2); g = np.repeat(np.arange(4), 4)
    return D, y, s, g


def fit(f, p0, lb=None, ub=None):
    best = None
    for q in p0:
        r = least_squares(f, q, bounds=(lb, ub) if lb is not None else (-np.inf, np.inf))
        if best is None or r.cost < best.cost: best = r
    return best


def run(cell):
    D, y, s, g = data(cell); out = {}
    # L8: independent LQ
    f = lambda p: (p[g]*D + p[4+g]*D*D - y)/s
    r = fit(f, [np.r_[np.full(4, .3), np.full(4, .05)]]); out["LQ independent"] = (2*r.cost, 8, r.x)
    # L5: shared beta
    f = lambda p: (p[g]*D + p[4]*D*D - y)/s
    r = fit(f, [np.r_[np.full(4, .3), .05]]); out["LQ shared beta"] = (2*r.cost, 5, r.x)
    # R5: constant RBE (dose rescaling), reference = 6 MV
    f = lambda p: (p[0]*(np.r_[1, p[2:5]][g]*D) + p[1]*(np.r_[1, p[2:5]][g]*D)**2 - y)/s
    r = fit(f, [np.r_[.3, .05, 1, 1, 1]]); out["LQ dose-rescaling"] = (2*r.cost, 5, r.x)
    # X: exact model, lambda_d per beam, shared F_r, kappa/mu, n_c; M fixed
    for M in (1, 2, 4):
        best = None
        for nc in range(2, 21):
            def f(p):
                lam = np.exp(p[:4]); F = 0.5 + np.exp(p[4]); k = np.exp(p[5])
                pred = np.empty_like(y)
                for b in range(4):
                    m = g == b
                    pred[m] = -np.log(np.maximum(survival_disc(D[m], lam[b], 1.0, 2*F, k, nc, M), 1e-300))
                return (pred - y)/s
            starts = [np.r_[np.full(4, np.log(x/M)), np.log(F0), np.log(k0)] for x, F0, k0 in [(0.5, .5, .3), (2, 3, 1), (1, .05, .01), (5, 10, 3)]]
            r = fit(f, starts, np.r_[np.full(4, np.log(1e-3)), np.log(1e-6), np.log(1e-5)], np.r_[np.full(4, np.log(200)), np.log(200), np.log(200)])
            if best is None or r.cost < best[0].cost: best = (r, nc)
        r, nc = best; lam = np.exp(r.x[:4]); F = .5 + np.exp(r.x[4]); k = np.exp(r.x[5])
        ab = [ab_disc(l, 1.0, 2*F, k, nc, M) for l in lam]
        out[f"exact model, lambda_d per beam, M={M}"] = (2*r.cost, 7, dict(nc=nc, lam=lam.tolist(), F_r=F, kappa_mu=k,
                                                         alpha0=[a for a, _ in ab], beta0=[b for _, b in ab]))
    return out


if __name__ == "__main__":
    allres = {}
    for cell in ("HCT116", "HeLa", "PC3"):
        res = run(cell); allres[cell] = {}
        print(f"== {cell} (16 points)")
        for k, (c2, npar, par) in res.items():
            dof = 16 - npar; p = chi2dist.sf(c2, dof)
            allres[cell][k] = dict(chi2=c2, k=npar, dof=dof, p=p, AIC=c2 + 2*npar, par=par if isinstance(par, dict) else np.asarray(par).tolist())
            print(f"  {k:40s} chi2={c2:7.2f} k={npar} dof={dof:2d} p={p:.2g} AIC={c2+2*npar:7.2f}")
    json.dump(allres, open(f"pooled_results_floor{FLOOR:g}.json", "w"), indent=1, default=float)
