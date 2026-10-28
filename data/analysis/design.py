"""Design of a discriminating clonogenic experiment for A549 (6 MV).
Information in hand: Howard 2018/2017 A549 6 MV curve (7 doses, 1-10 Gy). Plausible parameter sets:
exact model - all profile fits with chi2 <= chi2_min + 4 (identifiability_results.json); LQ - (alpha, beta) ellipse chi2 <= min + 4.
For candidate doses we compute the range of predicted ln S under each model family and the separation between
families relative to the measurement noise of a clonogenic assay:
  var(ln SF) = 1/(N_seed * PE * SF) + cv_plate^2  (+ control normalisation, same form at D=0), per replicate.
Output: envelopes, maximal separation per dose and replicates needed for 5 % two-sided test with 80 % power."""
import sys, json
import numpy as np
from scipy.stats import norm
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc

R = json.load(open("identifiability_results.json"))
d = R["Howard2018 A549 6MV"]; D0 = np.array(d["D"]); y0 = np.array(d["y"]); s0 = np.array(d["s"])
fits = d["fits"]; cmin = min(f["chi2"] for f in fits)
ok = [f for f in fits if f["chi2"] <= cmin + 4]
DOSES = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
ex = np.array([-np.log(survival_disc(DOSES, f["lam"], 1, 2*f["F"], f["k"], f["nc"], f["M"])) for f in ok])
# LQ ellipse
X = np.vstack([D0, D0**2]).T/s0[:, None]; cov = np.linalg.inv(X.T @ X); c = cov @ X.T @ (y0/s0)
L = np.linalg.cholesky(cov); ang = np.linspace(0, 2*np.pi, 360); r = np.sqrt(4.0)
ell = np.array([c + r*L @ np.array([np.cos(t), np.sin(t)]) for t in ang] + [c])
lq = np.array([a*DOSES + b*DOSES**2 for a, b in ell])
# noise model
PE, cv = 0.7, 0.10
def sd_ln(SF, Nseed): return np.sqrt(1/(Nseed*PE*SF) + cv**2 + 1/(Nseed*PE) + cv**2)   # irradiated + control dish
out = dict(n_exact_sets=len(ok), chi2_min=cmin, alpha_LQ=c[0], beta_LQ=c[1], doses=DOSES.tolist(), rows=[])
z = norm.ppf(0.975) + norm.ppf(0.8)
print(f"{len(ok)} exact-model parameter sets within chi2_min+4; LQ alpha={c[0]:.3f} beta={c[1]:.4f}")
print(" D    lnS_exact[min,max]     lnS_LQ[min,max]     max|gap| (exact vs LQ)   min gap   n_rep(1e3 cells) n_rep(1e4)")
for i, Dd in enumerate(DOSES):
    e = ex[:, i]; q = lq[:, i]
    # separation: distance between the two prediction intervals (0 if they overlap) and the largest possible difference
    gap_min = max(0.0, max(e.min() - q.max(), q.min() - e.max()))
    gap_max = max(abs(e.max() - q.min()), abs(q.max() - e.min()))
    S = np.exp(-np.median(q))
    nrep = {N: (z*sd_ln(S, N)/gap_max)**2 if gap_max > 0 else np.inf for N in (1e3, 1e4)}
    nrep_min = {N: (z*sd_ln(S, N)/gap_min)**2 if gap_min > 0 else np.inf for N in (1e3, 1e4)}
    out["rows"].append(dict(D=Dd, ex_min=e.min(), ex_max=e.max(), lq_min=q.min(), lq_max=q.max(), gap_max=gap_max, gap_min=gap_min,
                            nrep_detect_maxgap={str(k): v for k, v in nrep.items()}, nrep_guaranteed={str(k): v for k, v in nrep_min.items()}))
    print(f"{Dd:5.2f}  [{e.min():.3f},{e.max():.3f}]  [{q.min():.3f},{q.max():.3f}]   {gap_max:.3f}   {gap_min:.3f}   {nrep[1e3]:9.1f} {nrep[1e4]:9.1f}")
# alpha0 range of the exact family vs LQ alpha
a0 = np.array([f["alpha0"] for f in ok]); b0 = np.array([f["beta0"] for f in ok])
out["alpha0_range"] = [a0.min(), a0.max()]; out["beta0_range"] = [b0.min(), b0.max()]
out["F_r_range"] = [min(f["F"] for f in ok), max(f["F"] for f in ok)]; out["nc_range"] = [min(f["nc"] for f in ok), max(f["nc"] for f in ok)]
print("exact family: alpha0", np.round(out["alpha0_range"], 3), "beta0", np.round(out["beta0_range"], 4), "F_r", np.round(out["F_r_range"], 3), "n_c", out["nc_range"])
json.dump(out, open("design_results.json", "w"), indent=1, default=float)
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
dd = np.linspace(0.02, 10, 300)
exd = np.array([-np.log(survival_disc(dd, f["lam"], 1, 2*f["F"], f["k"], f["nc"], f["M"])) for f in ok])
lqd = np.array([a*dd + b*dd**2 for a, b in ell])
for a_ in ax:
    a_.fill_between(dd, np.exp(-exd.max(0)), np.exp(-exd.min(0)), color="0.75", label=r"exact model, $\Delta\chi^2\leq4$")
    a_.plot(dd, np.exp(-lqd.max(0)), "k--", lw=1, label=r"LQ, $\Delta\chi^2\leq4$"); a_.plot(dd, np.exp(-lqd.min(0)), "k--", lw=1)
    a_.errorbar(D0, np.exp(-y0), yerr=np.exp(-y0)*s0, fmt="o", mfc="white", color="k", ms=4, capsize=2, label="A549, 6 MV (Howard 2018, digitised)")
ax[0].set_yscale("log"); ax[0].set_xlabel("dose (Gy)"); ax[0].set_ylabel("surviving fraction"); ax[0].legend(fontsize=7, frameon=False, loc="lower left")
ax[1].set_xlim(0, 2.2); ax[1].set_ylim(0.5, 1.02); ax[1].set_xlabel("dose (Gy)"); ax[1].set_title("low-dose region (linear scale)", fontsize=9)
fig.tight_layout(); fig.savefig("fig_design.pdf"); fig.savefig("fig_design.png", dpi=140)

# ---- sub-families of the exact model: strong shoulder (alpha0 < 0.06) vs LQ-like (alpha0 >= 0.15) ----
print("\nsub-family separation (min distance between bands, ln S units) and dishes per dose needed (cv 10 %, 1e4 cells)")
fam = {"shoulder (alpha0<0.06)": [i for i, f in enumerate(ok) if f["alpha0"] < 0.06], "LQ-like (alpha0>=0.15)": [i for i, f in enumerate(ok) if f["alpha0"] >= 0.15]}
out["subfamilies"] = {}
for name, idx in fam.items():
    e = ex[idx]; rows = []
    for i, Dd in enumerate(DOSES):
        q = lq[:, i]; gap = max(0.0, max(e[:, i].min() - q.max(), q.min() - e[:, i].max()))
        S = np.exp(-np.median(q)); n1 = (z*sd_ln(S, 1e4)/gap)**2 if gap > 0 else np.inf
        cv2 = 0.03; n2 = (z*np.sqrt(2*(1/(1e4*PE*S) + cv2**2))/gap)**2 if gap > 0 else np.inf
        rows.append(dict(D=Dd, gap=gap, n_dishes_cv10=n1, n_dishes_cv3=n2))
        print(f"  {name:24s} D={Dd:4.2f} n_sets={len(idx):3d} gap={gap:.3f} dishes(cv10%)={n1:8.1f} dishes(cv3%)={n2:7.1f}")
    out["subfamilies"][name] = dict(n=len(idx), rows=rows)
json.dump(out, open("design_results.json", "w"), indent=1, default=float)
