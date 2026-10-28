"""Summary, table and figure for the Polgár low-dose test (reads polgar_results.json)."""
import sys, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc

R = json.load(open("polgar_results.json"))
MODELS = ["LQ", "IR", "LQ mixture", "exact M=1", "exact M=2", "exact M=4", "exact M=1 + sensitive fraction"]
MONO = {"A549": (0.191, 0.026), "CHO-K1": (0.151, 0.037), "PC-3 control": (0.195, 0.057)}
aic = lambda f: f["chi2"] + 2*f["k"]

rows = []
for r in R:
    F = r["fits"]; A = {m: aic(F[m]) for m in MODELS}
    hom = min(A[m] for m in ("exact M=1", "exact M=2", "exact M=4"))
    st = F.get("slope_test")
    rows.append(dict(id=r["id"], cell=r["cell"], radiation=r["radiation"], n=r["n"], Dmin=r["Dmin"], Dmax=r["Dmax"],
                     slope_z=st["z"] if st else None, best=min(A, key=A.get), **{f"AIC {m}": A[m] for m in MODELS},
                     dAIC_hom_minus_IR=hom - A["IR"], dAIC_hom_minus_mixexact=hom - A["exact M=1 + sensitive fraction"],
                     dAIC_LQ_minus_hom=A["LQ"] - hom, IR_as=F["IR"]["par"][1], IR_ar=F["IR"]["par"][0], IR_Dc=F["IR"]["par"][2],
                     chi2_hom=min(F[m]["chi2"] for m in ("exact M=1", "exact M=2", "exact M=4")), dof_hom=r["n"] - 4))

z = np.array([x["slope_z"] for x in rows if x["slope_z"] is not None])
dIR = np.array([x["dAIC_hom_minus_IR"] for x in rows]); dMX = np.array([x["dAIC_hom_minus_mixexact"] for x in rows]); dLQ = np.array([x["dAIC_LQ_minus_hom"] for x in rows])
summ = dict(n_datasets=len(rows), n_slope_test=len(z), n_slope_z_lt_2=int((z < -2).sum()), n_slope_z_lt_3=int((z < -3).sum()), n_slope_z_gt_2=int((z > 2).sum()),
            best_model_counts={m: sum(x["best"] == m for x in rows) for m in MODELS},
            homogeneous_worse_than_IR_by_gt2=int((dIR > 2).sum()), homogeneous_worse_than_IR_by_gt10=int((dIR > 10).sum()),
            homogeneous_worse_than_exact_plus_fraction_by_gt2=int((dMX > 2).sum()), homogeneous_worse_than_exact_plus_fraction_by_gt10=int((dMX > 10).sum()),
            homogeneous_better_than_LQ_by_gt2=int((dLQ > 2).sum()), homogeneous_worse_than_LQ_by_gt2=int((dLQ < -2).sum()),
            exact_plus_fraction_vs_IR_median_dAIC=float(np.median([x["AIC exact M=1 + sensitive fraction"] - x["AIC IR"] for x in rows])),
            exact_plus_fraction_better_than_IR=int(sum(x["AIC exact M=1 + sensitive fraction"] < x["AIC IR"] for x in rows)),
            mixture_LQ_better_than_IR=int(sum(x["AIC LQ mixture"] < x["AIC IR"] for x in rows)))
# monograph lines present in Polgár: low-dose -lnS/D vs monograph alpha
mono = []
for r in R:
    key = r["cell"]
    if key in MONO:
        D = np.array(r["D"]); g = -np.array(r["lnS"])/D; s = np.array(r["sig"])/D; m = D <= 0.5
        w = 1/s[m]**2
        mono.append(dict(id=r["id"], cell=key, radiation=r["radiation"], doi=r["doi"], monograph_alpha=MONO[key][0],
                         mean_lnS_over_D_D_le_0p5=float(np.sum(w*g[m])/np.sum(w)) if m.any() else None,
                         se=float(1/np.sqrt(np.sum(w))) if m.any() else None, IR_alpha_s=r["fits"]["IR"]["par"][1], IR_alpha_r=r["fits"]["IR"]["par"][0],
                         slope_z=r["fits"].get("slope_test", {}).get("z")))
summ["monograph_lines"] = mono
json.dump(dict(summary=summ, rows=rows), open("polgar_summary.json", "w"), indent=1, default=float)
with open("table_Polgar.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for x in rows: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in x.items()})
print(json.dumps({k: v for k, v in summ.items() if k != "monograph_lines"}, indent=1))
for m in mono: print(m)

# figure: (a) slope z histogram, (b) example dataset with the largest homogeneous misfit among A549/HT29
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].hist(z, bins=np.arange(-12, 6.5, 0.5), color="0.6", edgecolor="k")
ax[0].axvline(-2, ls="--", color="k", lw=1); ax[0].axvline(2, ls=":", color="k", lw=1)
ax[0].set_xlabel(r"$z$ of slope of $-\ln S/D$ on $D\leq1$ Gy"); ax[0].set_ylabel("photon datasets")
ax[0].set_title(f"(a) {len(z)} photon datasets: $z<-2$ in {int((z < -2).sum())}", fontsize=10)
cand = [r for r in R if r["cell"] == "A549" and r["fits"].get("slope_test")]
ex = min(cand, key=lambda r: r["fits"]["slope_test"]["z"]) if cand else R[0]
D = np.array(ex["D"]); y = -np.array(ex["lnS"]); s = np.array(ex["sig"]); dd = np.linspace(max(0.02, D.min()*0.8), D.max()*1.02, 300)
F = ex["fits"]
ax[1].errorbar(D, y/D, s/D, fmt="o", mfc="white", color="k", capsize=2, ms=4, label="data (Polgár 2022)")
p = F["LQ"]["par"]; ax[1].plot(dd, p[0] + p[1]*dd, ":", color="0.4", label="LQ")
p = F["IR"]["par"]; ax[1].plot(dd, p[0]*(1 + (p[1]/p[0] - 1)*np.exp(-dd/p[2])) + p[3]*dd, "-.", color="0.3", label="induced repair")
best = min(("exact M=1", "exact M=2", "exact M=4"), key=lambda m: F[m]["chi2"]); q = F[best]; M = int(best[-1])
ax[1].plot(dd, -np.log(survival_disc(dd, q["par"][0], 1, 2*q["par"][1], q["par"][2], q["nc"], M))/dd, "--", color="k", label=f"exact homogeneous ({best[6:]})")
q = F["exact M=1 + sensitive fraction"]; lam, Fr, k, ah, fr = q["par"]
ax[1].plot(dd, -np.log((1 - fr)*survival_disc(dd, lam, 1, 2*Fr, k, q["nc"], 1) + fr*np.exp(-ah*dd))/dd, "-", color="k", label="exact + sensitive fraction")
ax[1].set_xscale("log"); ax[1].set_xlabel("dose (Gy)"); ax[1].set_ylabel(r"$-\ln S/D$ (Gy$^{-1}$)"); ax[1].set_ylim(bottom=0)
ax[1].legend(fontsize=7, frameon=False); ax[1].set_title(f"(b) {ex['cell']}, dataset {ex['id']} ({ex['radiation']})", fontsize=10)
fig.tight_layout(); fig.savefig("fig_Polgar.pdf"); fig.savefig("fig_Polgar.png", dpi=150)
