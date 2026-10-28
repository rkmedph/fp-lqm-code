"""Summaries, CSV table and figure for the A549 literature check (reads a549_results.json)."""
import json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("a549_results.json"))["curves"]
am, bm = 0.191, 0.026
out = {}

def stats(sub, kind="abs"):
    a = np.array([r[f"LQ_{kind}"]["alpha"] for r in sub]); b = np.array([r[f"LQ_{kind}"]["beta"] for r in sub])
    zb = np.array([r[f"LQ_{kind}"]["beta"]/r[f"LQ_{kind}"]["se_beta"] for r in sub])
    return dict(n=len(sub), alpha_median=np.median(a), alpha_IQR=np.percentile(a, [25, 75]).tolist(),
                beta_median=np.median(b), beta_IQR=np.percentile(b, [25, 75]).tolist(),
                n_beta_neg=int((b < 0).sum()), n_beta_neg_2SE=int((zb < -2).sum()),
                pct_alpha_below_mono=float(100*(a <= am).mean()), pct_beta_below_mono=float(100*(b <= bm).mean()),
                ab_ratio_median=float(np.median(a/b)))

groups = {"all (>=3 points)": R,
          "4 points": [r for r in R if r["n"] == 4],
          "X-rays": [r for r in R if r["radiation"] == "X-rays"],
          "gamma-rays": [r for r in R if r["radiation"] == "gamma-rays"],
          "seeding before IR": [r for r in R if r["seeding"] == "before IR"],
          "seeding after IR": [r for r in R if r["seeding"] == "after IR"],
          "selected (X-rays, 0.64-1.64 Gy/min, before IR)": [r for r in R if r["radiation"] == "X-rays" and r["seeding"] == "before IR"
                                                           and r["dose_rate"] is not None and 0.64 <= (r["dose_rate"] or 0) <= 1.64]}
for kind in ("abs", "rel"):
    out[kind] = {g: stats(s, kind) for g, s in groups.items() if len(s) >= 3}

# convexity (model-free) and range stability
z = np.array([r["min_z_slope_increment"] for r in R])
out["convexity"] = dict(n=len(R), n_z_lt_2=int((z < -2).sum()), n_z_lt_3=int((z < -3).sum()),
                        curves_z_lt_3=[(r["PMID"], r["S"], round(r["min_z_slope_increment"], 1)) for r in R if r["min_z_slope_increment"] < -3])
F = [r for r in R if r["n"] == 4]
da = np.array([r["alpha_D<=4"] - r["LQ_abs"]["alpha"] for r in F]); db = np.array([r["beta_D<=4"] - r["LQ_abs"]["beta"] for r in F])
da6 = np.array([r["alpha_D<=6"] - r["LQ_abs"]["alpha"] for r in F])
out["range_stability"] = dict(n=len(F), dalpha_D4_median=float(np.median(da)), dalpha_D4_IQR=np.percentile(da, [25, 75]).tolist(),
                              dalpha_D4_absmedian=float(np.median(abs(da))), dbeta_D4_median=float(np.median(db)),
                              dalpha_D6_median=float(np.median(da6)), dalpha_D6_absmedian=float(np.median(abs(da6))),
                              frac_dalpha_D4_pos=float((da > 0).mean()))
# monograph prediction vs literature distribution
D = np.array([2, 4, 6, 8.]); pred = np.exp(-am*D - bm*D*D)
lit = {d: [s for r in R for dd, s in zip(r["D"], r["S"]) if dd == d] for d in D}
out["monograph_vs_literature"] = [dict(D=d, SF_pred=float(p), lit_median=float(np.median(lit[d])), lit_IQR=np.percentile(lit[d], [25, 75]).tolist(),
                                       percentile_of_pred=float(100*np.mean(np.array(lit[d]) <= p)), n=len(lit[d])) for d, p in zip(D, pred)]
out["near_monograph"] = [(r["PMID"], r["radiation"], r["S"], round(r["LQ_abs"]["alpha"], 3), round(r["LQ_abs"]["beta"], 4)) for r in R
                         if abs(r["LQ_abs"]["alpha"] - am) <= 0.02 and abs(r["LQ_abs"]["beta"] - bm) <= 0.005]
# exact model, 4-point curves
if "model" in F[0]:
    from scipy.stats import chi2 as c2d
    mod = {}
    for M in ("M1", "M2", "M4"):
        c = np.array([r["model"][M]["chi2"] for r in F]); clq = np.array([r["LQ_abs"]["chi2"] for r in F])
        a0w = np.array([r["model"][M]["alpha0_range"][1] - r["model"][M]["alpha0_range"][0] for r in F])
        bnd = np.array([r["model"][M]["best"]["at_bound"] for r in F])
        mod[M] = dict(n=len(F), n_chi2_gt_3p84=int((c > 3.84).sum()), n_chi2_gt_lq=int((c > clq + 1e-6).sum()),
                      median_chi2=float(np.median(c)), median_alpha0_spread=float(np.median(a0w)), n_at_bound=int(bnd.sum()),
                      n_Fr_boundary=int(sum(r["model"][M]["best"]["F_r"] < 0.5 + 1e-4 for r in F)))
    mod["LQ_chi2_gt_5p99_(2dof)"] = int((np.array([r["LQ_abs"]["chi2"] for r in F]) > 5.99).sum())
    out["exact_model_4pt"] = mod
json.dump(out, open("a549_summary.json", "w"), indent=1, default=float)

with open("table_A549_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["curve_id", "PMID", "radiation", "dose_rate", "seeding", "n_points", "SF2", "SF4", "SF6", "SF8",
                "alpha", "alpha_SE", "beta", "beta_SE", "LQ_chi2", "min_z_slope_increment", "alpha_D<=4", "beta_D<=4",
                "M1_chi2", "M2_chi2", "M4_chi2", "M1_alpha0_range"])
    for r in R:
        S = dict(zip(r["D"], r["S"])); L = r["LQ_abs"]; m = r.get("model")
        w.writerow([r["curve_id"], r["PMID"], r["radiation"], "" if r["dose_rate"] is None or r["dose_rate"] != r["dose_rate"] else r["dose_rate"],
                    r["seeding"], r["n"], *[S.get(d, "") for d in D], f"{L['alpha']:.3f}", f"{L['se_alpha']:.3f}", f"{L['beta']:.4f}", f"{L['se_beta']:.4f}",
                    f"{L['chi2']:.2f}", f"{r['min_z_slope_increment']:.2f}",
                    f"{r['alpha_D<=4']:.3f}" if "alpha_D<=4" in r else "", f"{r['beta_D<=4']:.4f}" if "beta_D<=4" in r else "",
                    *([f"{m[k]['chi2']:.2f}" for k in ("M1", "M2", "M4")] if m else ["", "", ""]),
                    f"{m['M1']['alpha0_range'][0]:.3f}-{m['M1']['alpha0_range'][1]:.3f}" if m else ""])

# figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
a = np.array([r["LQ_abs"]["alpha"] for r in R]); b = np.array([r["LQ_abs"]["beta"] for r in R])
xr = np.array([r["radiation"] == "X-rays" for r in R])
ax[0].scatter(a[xr], b[xr], s=14, marker="o", facecolors="none", edgecolors="k", label="X-rays")
ax[0].scatter(a[~xr], b[~xr], s=14, marker="^", facecolors="none", edgecolors="0.45", label=r"$\gamma$-rays")
ax[0].plot(am, bm, "*", ms=14, color="k", label="monograph table (A549)")
ax[0].axhline(0, color="0.7", lw=0.7); ax[0].axvline(0, color="0.7", lw=0.7)
ax[0].set_xlabel(r"$\alpha$ (Gy$^{-1}$)"); ax[0].set_ylabel(r"$\beta$ (Gy$^{-2}$)"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title(f"(a) A549, {len(R)} literature curves (LQ, 2-8 Gy)", fontsize=10)
pos = np.arange(4)
ax[1].boxplot([lit[d] for d in D], positions=D, widths=0.8, showfliers=True, flierprops=dict(marker=".", ms=3), medianprops=dict(color="k", lw=1.5))
dd = np.linspace(0, 8.5, 100); ax[1].plot(dd, np.exp(-am*dd - bm*dd*dd), "k--", label="monograph LQ")
ax[1].set_yscale("log"); ax[1].set_xlabel("dose (Gy)"); ax[1].set_ylabel("surviving fraction"); ax[1].set_xticks(D); ax[1].set_xticklabels([f"{d:g}" for d in D])
ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("(b) spread of published SF vs monograph coefficients", fontsize=10)
fig.tight_layout(); fig.savefig("fig_A549.pdf"); fig.savefig("fig_A549.png", dpi=150)
print(json.dumps({k: v for k, v in out.items() if k != "convexity"}, indent=1, default=float)[:4000]); print(out["convexity"])
