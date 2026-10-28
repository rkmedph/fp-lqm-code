"""Flatten validation_results.json and pooled_results_*.json into CSV tables and draw the summary figure."""
import json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc
from registry import build, sigma_lnS

R = json.load(open("validation_results.json"))
f3 = lambda x: "" if x is None else f"{x:.3f}"
f4 = lambda x: "" if x is None else f"{x:.4f}"

with open("table_fits.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["curve_id", "n_points", "dose_range_Gy",
                "source_alpha", "source_beta", "refit_source_style_alpha", "refit_source_style_beta", "source_style",
                "LQ_alpha", "LQ_alpha_SE", "LQ_beta", "LQ_beta_SE", "LQ_chi2", "LQ_dof", "LQ_p",
                "M1_chi2", "M2_chi2", "M4_chi2", "model_dof",
                "M1_alpha0_range", "M2_alpha0_range", "M4_alpha0_range", "M1_beta0_range", "M2_beta0_range", "M4_beta0_range",
                "monograph_alpha", "monograph_beta", "monograph_chi2_on_points"])
    for r in R:
        rep = r.get("reproduction"); src = r.get("source_LQ")
        style = "SEM-weighted, linear SF" if r["curve_id"].startswith("LI24") else "unweighted, linear SF"
        rr = rep[style] if rep else None
        m = r["model"]; mono = r.get("monograph") if (r["curve_id"].endswith("_6MV") or r["curve_id"].startswith("MUR04")) else None
        rng = lambda v, fmt: f"{fmt(v[0])}-{fmt(v[1])}"
        w.writerow([r["curve_id"], r["n"], f"{min(r['D']):g}-{max(r['D']):g}",
                    f3(src["alpha"]) if src else "", f3(src["beta"]) if src else "",
                    f3(rr["alpha"]) if rr else "", f4(rr["beta"]) if rr else "", style if rr else "",
                    f3(r["LQ"]["alpha"]), f3(r["LQ"]["se_alpha"]), f4(r["LQ"]["beta"]), f4(r["LQ"]["se_beta"]),
                    f"{r['LQ']['chi2']:.2f}", r["LQ"]["dof"], "" if r["LQ"]["p"] is None else f"{r['LQ']['p']:.2g}",
                    *[f"{m[k]['chi2']:.2f}" for k in ("M1", "M2", "M4")], m["M1"]["dof"],
                    *[rng(m[k]["alpha0_range"], f3) for k in ("M1", "M2", "M4")],
                    *[rng(m[k]["beta0_range"], f4) for k in ("M1", "M2", "M4")],
                    mono["alpha"] if mono else "", mono["beta"] if mono else "", f"{mono['chi2']:.1f}" if mono else ""])

with open("table_stability.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["curve_id", "subset", "n_points", "alpha", "alpha_SE", "beta", "beta_SE"])
    for r in R:
        w.writerow([r["curve_id"], "all doses", r["n"], f3(r["LQ"]["alpha"]), f3(r["LQ"]["se_alpha"]), f4(r["LQ"]["beta"]), f4(r["LQ"]["se_beta"])])
        for s in r["stability"]:
            w.writerow([r["curve_id"], s["kind"], s["n"], f3(s["alpha"]), f3(s["se_alpha"]), f4(s["beta"]), f4(s["se_beta"])])

with open("table_pooled.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell_line", "extra_relative_error", "model", "n_params", "chi2", "dof", "p", "AIC"])
    for fl in ("0", "0.1"):
        P = json.load(open(f"pooled_results_floor{fl}.json"))
        for cell, d in P.items():
            for k, v in d.items():
                w.writerow([cell, fl, k, v["k"], f"{v['chi2']:.2f}", v["dof"], f"{v['p']:.2g}", f"{v['AIC']:.2f}"])

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
es = [r for r in R if r["curve_id"].startswith("ES59")][0]
tr = [s for s in es["stability"] if s["kind"].startswith("D<=")]
dm = [float(s["kind"][3:]) for s in tr]
a = ax[0]
a.errorbar(dm, [s["alpha"] for s in tr], [s["se_alpha"] for s in tr], fmt="o-", color="k", capsize=3, label=r"$\alpha$ (Gy$^{-1}$)")
a2 = a.twinx()
a2.errorbar(dm, [s["beta"] for s in tr], [s["se_beta"] for s in tr], fmt="s--", color="0.45", mfc="white", capsize=3, label=r"$\beta$ (Gy$^{-2}$)")
a.set_xlabel(r"upper limit of fitted dose range $D_{\max}$ (Gy)"); a.set_ylabel(r"$\alpha$ (Gy$^{-1}$)"); a2.set_ylabel(r"$\beta$ (Gy$^{-2}$)")
a.set_ylim(-0.2, 0.7); a2.set_ylim(-0.15, 0.2)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1+h2, l1+l2, loc="upper left", frameon=False)
a.set_title("(a) Elkind & Sutton 1959: LQ coefficients vs fitted range", fontsize=10)

b = ax[1]
P = json.load(open("pooled_results_floor0.json"))["HeLa"]
p = P["exact model, lambda_d per beam, M=1"]["par"]; lq = P["LQ independent"]["par"]
cs = {c["beam"]: c for c in build() if c["src"] == "LI24" and c["cell"] == "HeLa"}
mk = {"6 MV": "o", "192Ir": "s", "225 kVp": "^", "50 kVp": "D"}
dd = np.linspace(0.01, 8.5, 200)
for i, beam in enumerate(["6 MV", "192Ir", "225 kVp", "50 kVp"]):
    c = cs[beam]; D = np.array(c["D"]); S = np.array(c["S"]); sl = sigma_lnS(c)
    off = 10.0**(-i)
    b.errorbar(D, S*off, yerr=S*sl*off, fmt=mk[beam], mfc="white", color="k", capsize=2, label=beam + (f" (x1e-{i})" if i else ""))
    b.plot(dd, off*np.exp(-lq[i]*dd - lq[4+i]*dd*dd), ":", color="0.4", lw=1)
    b.plot(dd, off*survival_disc(dd, p["lam"][i], 1.0, 2*p["F_r"], p["kappa_mu"], p["nc"], 1), "-", color="k", lw=1)
b.plot([], [], ":", color="0.4", label="LQ, 8 parameters"); b.plot([], [], "-", color="k", label=r"exact model, only $\lambda_d$ varies, 7 par.")
b.set_yscale("log"); b.set_xlabel("dose (Gy)"); b.set_ylabel("surviving fraction (offset)"); b.set_ylim(1e-7, 1.5)
b.legend(fontsize=7, frameon=False, loc="lower left"); b.set_title("(b) HeLa, Li et al 2024: four photon qualities", fontsize=10)
fig.tight_layout(); fig.savefig("fig_validation.pdf"); fig.savefig("fig_validation.png", dpi=150)
print("ok")
