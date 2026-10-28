"""Nuryadi et al 2018, Supplementary Table 2: 20 in-house A549 experiments (100 kVp X-rays, 1.14 Gy/min, Faxitron RX-650;
cells seeded in triplicate 12 h before irradiation, 10 d incubation, colonies >=50 cells; SF normalised to unirradiated controls;
reported values are averages of the triplicate, two decimals, no uncertainties).

Common likelihood for every model and every curve:
  ln SF_obs ~ Normal(ln SF_model, sigma_i^2),  sigma_i^2 = s_r^2 + (delta/SF_obs)^2,
  delta = 0.005/sqrt(3) (rounding to 2 decimals), s_r = relative between-dish/experiment scatter (common to all curves,
  estimated by maximum likelihood on the individual LQ fits; sensitivity: s_r re-estimated under the exact model).
  SF = 0.00 is left-censored: P(SF_true < 0.005) = Phi((ln 0.005 - ln SF_model)/s_r).
Models per curve: LQ (alpha, beta free), LQ with beta >= 0, exact threshold model M = 1, 2, 4 (lambda_d, F_r, kappa/mu, n_c).
"""
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.stats import norm
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc, ab_disc

D = np.array([2.0, 4.0, 6.0, 8.0])
T2 = np.array([  # SF2, SF4, SF6, SF8, D10, D50 (Supplementary Table 2, p. 4)
 [0.80,0.53,0.28,0.10,8.1,4.2],[0.74,0.22,0.10,0.05,6.3,2.2],[0.65,0.38,0.20,0.10,7.9,3.0],[0.70,0.54,0.33,0.18,9.9,4.1],
 [0.64,0.37,0.22,0.10,8.1,3.0],[0.72,0.22,0.06,0.02,5.5,2.3],[0.87,0.29,0.11,0.05,6.5,2.8],[0.76,0.54,0.33,0.18,9.6,4.3],
 [0.72,0.47,0.24,0.08,7.7,3.8],[0.60,0.28,0.14,0.02,6.1,2.8],[0.63,0.26,0.05,0.01,5.3,2.6],[0.87,0.28,0.10,0.05,6.5,2.6],
 [0.49,0.32,0.14,0.05,6.6,2.5],[0.75,0.44,0.16,0.05,7.0,3.5],[0.87,0.55,0.31,0.11,8.2,4.5],[0.69,0.50,0.24,0.12,8.3,3.7],
 [0.65,0.38,0.18,0.10,7.9,2.9],[0.90,0.29,0.10,0.04,6.4,2.8],[0.47,0.28,0.12,0.04,6.2,2.3],[0.58,0.24,0.13,0.00,6.7,2.1]])
DELTA = 0.005/np.sqrt(3); LOD = 0.005
NC = range(2, 31)


def nll(lnSmod, S, sr):
    """-log likelihood (up to a constant) for one curve; censored where S == 0."""
    lnSmod = np.asarray(lnSmod, float); out = 0.0
    obs = S > 0
    if obs.any():
        sig = np.sqrt(sr**2 + (DELTA/S[obs])**2)
        out += np.sum(0.5*((np.log(S[obs]) - lnSmod[obs])/sig)**2 + np.log(sig))
    if (~obs).any():
        out -= np.sum(norm.logcdf((np.log(LOD) - lnSmod[~obs])/sr))
    return out


def fit_lq(S, sr, nonneg=False):
    best = None
    for p0 in ([0.15, 0.03], [0.05, 0.05], [0.3, 0.01]):
        bnds = [(-2, 3), (0 if nonneg else -1, 1)]
        r = minimize(lambda p: nll(-(p[0]*D + p[1]*D*D), S, sr), p0, method="L-BFGS-B", bounds=bnds)
        if best is None or r.fun < best.fun: best = r
    return dict(nll=best.fun, alpha=best.x[0], beta=best.x[1], k=2)


def fit_exact(S, sr, M):
    prof = []
    for nc in NC:
        best = None
        for p0 in ([np.log(0.5/M), np.log(0.5), np.log(0.3)], [np.log(2/M), np.log(3), np.log(1)], [np.log(1/M), np.log(.05), np.log(.01)],
                   [np.log(5/M), np.log(10), np.log(3)]):
            f = lambda p: nll(np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1, 2*(0.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)), S, sr)
            r = minimize(f, p0, method="L-BFGS-B", bounds=[(np.log(1e-3), np.log(200)), (np.log(1e-6), np.log(1e4)), (np.log(1e-5), np.log(1e4))])
            if best is None or r.fun < best.fun: best = r
        lam, Fm, k = np.exp(best.x); a0, b0 = ab_disc(lam, 1, 2*(0.5 + Fm), k, nc, M)
        prof.append(dict(nc=nc, nll=best.fun, lam=lam, F_r=0.5 + Fm, kappa_mu=k, alpha0=a0, beta0=b0,
                         at_bound=bool(Fm < 2e-6 or k < 2e-5 or Fm > 9e3 or k > 9e3 or lam > 199)))
    b = min(prof, key=lambda p: p["nll"]); ok = [p for p in prof if 2*(p["nll"] - b["nll"]) <= 1.0]
    return dict(nll=b["nll"], k=4, best=b, alpha0_range=[min(p["alpha0"] for p in ok), max(p["alpha0"] for p in ok)],
                beta0_range=[min(p["beta0"] for p in ok), max(p["beta0"] for p in ok)], nc_set=[p["nc"] for p in ok])


def estimate_sr(model="LQ"):
    """ML estimate of the common relative scatter s_r, profiling each curve's parameters."""
    def total(log_sr):
        sr = np.exp(log_sr)
        if model == "LQ":
            return sum(fit_lq(row[:4], sr)["nll"] for row in T2)
        return sum(min(fit_exact(row[:4], sr, M)["nll"] for M in (1,)) for row in T2)
    r = minimize_scalar(total, bounds=(np.log(0.01), np.log(1.0)), method="bounded", options=dict(xatol=1e-3))
    return float(np.exp(r.x))


def d_from_lq(a, b, s):
    """dose giving survival s under LQ (authors' D10/D50 definition)."""
    L = -np.log(s)
    if abs(b) < 1e-9: return L/a
    disc = a*a + 4*b*L
    return (-a + np.sqrt(disc))/(2*b) if disc >= 0 else np.nan


def main():
    out = {}
    # 0. descriptive statistics and checks
    desc = {}
    for j, name in enumerate(["SF2", "SF4", "SF6", "SF8", "D10", "D50"]):
        v = T2[:, j]; desc[name] = dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()), cv=float(v.std(ddof=1)/v.mean()))
    out["descriptive"] = desc
    dup = [(i+1, j+1) for i in range(20) for j in range(i+1, 20) if np.allclose(T2[i, 1:4], T2[j, 1:4])]
    out["identical_SF4_SF8_rows"] = dup
    # 1. common scatter
    sr = estimate_sr("LQ"); out["s_r_LQ"] = sr
    print(f"s_r (ML, LQ) = {sr:.3f}", flush=True)
    rows = []
    for i, row in enumerate(T2):
        S = row[:4]
        lq = fit_lq(S, sr); lqp = fit_lq(S, sr, nonneg=True)
        ex = {M: fit_exact(S, sr, M) for M in (1, 2, 4)}
        # authors' D10/D50 vs plain LQ on the means (exclude censored point)
        m = S > 0; X = np.vstack([D[m], D[m]**2]).T; ab = np.linalg.lstsq(X, -np.log(S[m]), rcond=None)[0]
        g = -np.log(np.where(S > 0, S, np.nan))/D; dz = np.nanmin(np.diff(g))
        rec = dict(exp=i + 1, SF=S.tolist(), D10_paper=row[4], D50_paper=row[5],
                   D10_unweighted_LQ=d_from_lq(ab[0], ab[1], 0.1), D50_unweighted_LQ=d_from_lq(ab[0], ab[1], 0.5),
                   alpha_unweighted=ab[0], beta_unweighted=ab[1], min_increment_lnS_over_D=float(dz),
                   LQ=lq, LQ_beta_nonneg=lqp, **{f"exact_M{M}": ex[M] for M in (1, 2, 4)})
        rows.append(rec)
        print(f"exp {i+1:2d}: LQ a={lq['alpha']:.3f} b={lq['beta']:.4f} -2lnL={2*lq['nll']:.2f} | "
              + " ".join(f"M{M}: -2lnL={2*ex[M]['nll']:.2f} a0={ex[M]['alpha0_range'][0]:.2f}-{ex[M]['alpha0_range'][1]:.2f}" for M in (1, 2, 4)), flush=True)
    out["curves"] = rows
    # 2. pooled comparison
    tot = {"LQ": sum(2*r["LQ"]["nll"] for r in rows), "LQ beta>=0": sum(2*r["LQ_beta_nonneg"]["nll"] for r in rows)}
    for M in (1, 2, 4): tot[f"exact M={M}"] = sum(2*r[f"exact_M{M}"]["nll"] for r in rows)
    tot["exact best M per curve"] = sum(min(2*r[f"exact_M{M}"]["nll"] for M in (1, 2, 4)) for r in rows)
    out["sum_minus2lnL"] = tot
    out["n_exact_worse_than_LQ_by_gt2"] = {M: sum(2*(r[f"exact_M{M}"]["nll"] - r["LQ"]["nll"]) > 2 for r in rows) for M in (1, 2, 4)}
    out["n_exact_worse_than_LQnonneg_by_gt2"] = {M: sum(2*(r[f"exact_M{M}"]["nll"] - r["LQ_beta_nonneg"]["nll"]) > 2 for r in rows) for M in (1, 2, 4)}
    # 3. one shared curve for all 20 experiments? (LQ, s_r fixed)
    def shared(p): return sum(nll(-(p[0]*D + p[1]*D*D), row[:4], sr) for row in T2)
    r = minimize(shared, [0.15, 0.03], method="Nelder-Mead")
    out["shared_LQ"] = dict(alpha=r.x[0], beta=r.x[1], minus2lnL=2*r.fun, LR_vs_individual=2*r.fun - tot["LQ"], dof=2*20 - 2)
    # inter-experiment scatter of ln SF vs s_r
    out["between_experiment_sd_lnSF"] = {f"SF{int(d)}": float(np.std(np.log(T2[T2[:, j] > 0, j]), ddof=1)) for j, d in enumerate(D)}
    json.dump(out, open("nuryadi_inhouse_results.json", "w"), indent=1, default=float)
    with open("table_Nuryadi_inhouse.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "SF2", "SF4", "SF6", "SF8", "D10_paper", "D10_LQ_refit", "D50_paper", "D50_LQ_refit",
                    "alpha_LQ", "beta_LQ", "m2lnL_LQ", "m2lnL_LQ_beta_ge0", "m2lnL_M1", "m2lnL_M2", "m2lnL_M4",
                    "alpha0_range_M1", "alpha0_range_M2", "alpha0_range_M4", "beta0_range_M1", "nc_set_M1", "exact_M1_at_bound"])
        for r in rows:
            w.writerow([r["exp"], *r["SF"], r["D10_paper"], f"{r['D10_unweighted_LQ']:.2f}", r["D50_paper"], f"{r['D50_unweighted_LQ']:.2f}",
                        f"{r['LQ']['alpha']:.3f}", f"{r['LQ']['beta']:.4f}", f"{2*r['LQ']['nll']:.2f}", f"{2*r['LQ_beta_nonneg']['nll']:.2f}",
                        *[f"{2*r[f'exact_M{M}']['nll']:.2f}" for M in (1, 2, 4)],
                        *[f"{r[f'exact_M{M}']['alpha0_range'][0]:.3f}-{r[f'exact_M{M}']['alpha0_range'][1]:.3f}" for M in (1, 2, 4)],
                        f"{r['exact_M1']['beta0_range'][0]:.4f}-{r['exact_M1']['beta0_range'][1]:.4f}",
                        f"{min(r['exact_M1']['nc_set'])}-{max(r['exact_M1']['nc_set'])}", r["exact_M1"]["best"]["at_bound"]])
    print(json.dumps({k: v for k, v in out.items() if k not in ("curves",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
