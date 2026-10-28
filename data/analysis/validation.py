"""Independent check of the model on raw literature survival curves (registry.py).

Steps
 1. reproduce the LQ fit reported in each source with the source's own fitting method;
 2. LQ fit in log space with stated uncertainties;
 3. exact threshold-crossing model with M = 1, 2, 4 (n_c scanned), small-dose alpha0, beta0 implied;
 4. dose-range stability of LQ coefficients (truncated ranges, leave-one-dose-out);
 5. comparison with the monograph table (no assumption that rows come from the same experiments);
 6. pooled test for Li et al: does beam quality act only as a rescaling of the lesion yield?
Outputs: validation_results.json, table_*.csv
"""
import sys, json, csv, itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2dist
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc, ab_disc
from registry import build, sigma_lnS, MONO

NC = range(2, 31)
MS = (1, 2, 4)


# ---------- LQ ----------
def lq_log(D, y, s):
    """Weighted LS for y = -lnS = aD + bD^2 with sigma s. Returns a, b, cov (unscaled), chi2."""
    X = np.vstack([D, D*D]).T / s[:, None]; z = y / s
    cov = np.linalg.inv(X.T @ X); c = cov @ X.T @ z; r = z - X @ c
    return c[0], c[1], cov, float(r @ r)


def lq_lin(D, S, sem):
    """Source-style fit: SF = exp(-aD-bD^2), weights 1/SEM^2 on the linear SF scale."""
    w = 1.0 / sem if sem is not None else np.ones_like(S)
    f = lambda p: (np.exp(-p[0]*D - p[1]*D*D) - S) * w
    r = least_squares(f, [0.2, 0.03], method="lm")
    J = r.jac; cov = np.linalg.inv(J.T @ J); chi2 = float(r.fun @ r.fun); dof = len(D) - 2
    se = np.sqrt(np.diag(cov) * (chi2/dof if dof > 0 else 1.0))  # Origin default: scale by reduced chi2
    return r.x, se


# ---------- exact model ----------
def neglogS(D, lam, F, k, nc, M):
    return -np.log(np.maximum(survival_disc(D, lam, 1.0, 2*F, k, nc, M), 1e-300))


def fit_model_nc(D, y, s, M, nc):
    best = None
    lb = np.log([1e-3, 1e-6, 1e-5]); ub = np.log([200, 200, 200])
    for p0 in ([np.log(0.5/M), np.log(0.5), np.log(0.3)], [np.log(2.0/M), np.log(3.0), np.log(1.0)],
               [np.log(1.0/M), np.log(0.05), np.log(0.01)], [np.log(5.0/M), np.log(10.0), np.log(3.0)]):
        f = lambda p: (neglogS(D, np.exp(p[0]), 0.5+np.exp(p[1]), np.exp(p[2]), nc, M) - y) / s
        r = least_squares(f, p0, bounds=(lb, ub))
        c2 = float(r.fun @ r.fun)
        if best is None or c2 < best[0]:
            best = (c2, np.exp(r.x))
    c2, (lam, Fm, k) = best
    a0, b0 = ab_disc(lam, 1.0, 2*(0.5+Fm), k, nc, M)
    at_bound = bool(Fm < 1e-5 or k < 1e-4 or lam > 199 or Fm > 199 or k > 199)
    return dict(nc=nc, chi2=c2, lam_d=lam, F_r=0.5+Fm, kappa_mu=k, alpha0=a0, beta0=b0, at_bound=at_bound)


def fit_model(D, y, s, M):
    prof = [fit_model_nc(D, y, s, M, nc) for nc in NC]
    c2min = min(p["chi2"] for p in prof); best = min(prof, key=lambda p: p["chi2"])
    ok = [p for p in prof if p["chi2"] <= c2min + 1.0]
    return dict(best=best, chi2=c2min, dof=len(D) - 4,
                nc_set=[p["nc"] for p in ok],
                alpha0_range=[min(p["alpha0"] for p in ok), max(p["alpha0"] for p in ok)],
                beta0_range=[min(p["beta0"] for p in ok), max(p["beta0"] for p in ok)],
                profile=[{k: p[k] for k in ("nc", "chi2", "alpha0", "beta0", "F_r", "kappa_mu", "lam_d")} for p in prof])


def main():
    curves = build(); res = []
    for c in curves:
        D = np.array(c["D"], float); S = np.array(c["S"], float); y = -np.log(S); s = sigma_lnS(c)
        r = dict(curve_id=c["curve_id"], cell=c["cell"], beam=c["beam"], n=len(D), D=D.tolist())
        # 1. reproduction of source fit
        if c["src_alpha"] is not None:
            sem = np.array(c["unc"]) / (np.sqrt(3) if c["unc_kind"].startswith("SD") else 1)
            variants = {"SEM-weighted, linear SF": lq_lin(D, S, sem), "unweighted, linear SF": lq_lin(D, S, None)}
            r["reproduction"] = {k: dict(alpha=v[0][0], beta=v[0][1], se_alpha=v[1][0], se_beta=v[1][1]) for k, v in variants.items()}
            r["source_LQ"] = dict(alpha=c["src_alpha"], se_alpha=c["src_alpha_se"], beta=c["src_beta"], se_beta=c["src_beta_se"])
        # 2. LQ in log space
        a, b, cov, c2 = lq_log(D, y, s); dof = len(D) - 2
        r["LQ"] = dict(alpha=a, beta=b, se_alpha=float(np.sqrt(cov[0, 0])), se_beta=float(np.sqrt(cov[1, 1])),
                       chi2=c2, dof=dof, p=float(chi2dist.sf(c2, dof)) if dof > 0 else None, AIC=c2 + 4)
        # 3. exact model
        r["model"] = {}
        for M in MS:
            fm = fit_model(D, y, s, M); fm["AIC"] = fm["chi2"] + 2*4
            r["model"][f"M{M}"] = fm
        # 4. stability: truncated dose ranges and leave-one-dose-out
        stab = []
        Ds = sorted(set(D))
        for dmax in Ds[1:]:
            m = D <= dmax
            if m.sum() >= 2:
                a_, b_, cov_, c2_ = lq_log(D[m], y[m], s[m])
                stab.append(dict(kind=f"D<={dmax:g}", n=int(m.sum()), alpha=a_, beta=b_,
                                 se_alpha=float(np.sqrt(cov_[0, 0])), se_beta=float(np.sqrt(cov_[1, 1]))))
        if len(D) >= 4:
            for i in range(len(D)):
                m = np.arange(len(D)) != i
                a_, b_, cov_, _ = lq_log(D[m], y[m], s[m])
                stab.append(dict(kind=f"drop {D[i]:g} Gy", n=int(m.sum()), alpha=a_, beta=b_,
                                 se_alpha=float(np.sqrt(cov_[0, 0])), se_beta=float(np.sqrt(cov_[1, 1]))))
        r["stability"] = stab
        # 5. monograph comparison
        key = "CHO" if c["cell"].startswith("CHO") else c["cell"]
        if key in MONO:
            am, bm = MONO[key]; d1 = D[0]
            pred = np.exp(-am*D - bm*D*D)
            r["monograph"] = dict(alpha=am, beta=bm, SF_pred=pred.tolist(), SF_obs=S.tolist(),
                                  z_lnS=((-np.log(pred) - y)/s).tolist(), chi2=float(np.sum(((-np.log(pred) - y)/s)**2)))
        res.append(r)
        print(f"{c['curve_id']:24s} LQ a={a:.3f}±{r['LQ']['se_alpha']:.3f} b={b:.4f}±{r['LQ']['se_beta']:.4f} chi2={c2:.2f}/{dof} | "
              + " ".join(f"M{M}:chi2={r['model'][f'M{M}']['chi2']:.2f}" for M in MS), flush=True)
    json.dump(res, open("validation_results.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    main()
