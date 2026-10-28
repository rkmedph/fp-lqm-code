"""Figure: TrueBeam 6 MV curves (Howard 2018, digitised) with LQ and exact-model fits; monograph rows for A549 and CHO."""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis")
from discrete import survival_disc
from validation import fit_model
H = {c["cell"]: c for c in json.load(open("digitize/howard_curves.json"))}
R = {r["cell"]: r for r in json.load(open("mv_results.json")) if r["curve_id"].startswith("HOW")}
MONO = {"A549": (0.191, 0.026), "CHO": (0.151, 0.037)}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True); dd = np.linspace(0, 10.5, 200)
for ax, cell in zip(axs, ["CHO", "A549", "T98"]):
    c = H[cell]; D = np.array([p["D"] for p in c["points"]]); S = np.array([p["SF"] for p in c["points"]]); s = np.array([p["sigma_lnSF"] for p in c["points"]])
    ax.errorbar(D, S, yerr=[S*(1 - np.exp(-s)), S*(np.exp(s) - 1)], fmt="o", mfc="white", color="k", capsize=2, ms=4, label="TrueBeam 6 MV (digitised)")
    L = R[cell]["fits"]["all"]["LQ"]; ax.plot(dd, np.exp(-L["alpha"]*dd - L["beta"]*dd*dd), ":", color="0.35", label="LQ fit")
    f = R[cell]["fits"]["all"]; M = min((1, 2, 4), key=lambda m: f[f"M{m}"]["chi2"]); q = f[f"M{M}"]
    best = fit_model(D, -np.log(S), s, M)["best"]
    ax.plot(dd, survival_disc(dd, best["lam"] if "lam" in best else best["lam_d"], 1, 2*best["F_r"], best["kappa_mu"], best["nc"], M), "-", color="k", lw=1, label=f"exact model (M={M})")
    if cell in MONO:
        a, b = MONO[cell]; ax.plot(dd, np.exp(-a*dd - b*dd*dd), "--", color="0.55", label="monograph table row")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 1.3); ax.set_xlabel("dose (Gy)"); ax.set_title(f"{cell} (Howard et al 2018)", fontsize=10)
axs[0].set_ylabel("surviving fraction"); axs[1].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("fig_MV_TrueBeam.pdf"); fig.savefig("fig_MV_TrueBeam.png", dpi=140)
