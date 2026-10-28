"""Small multiples: 20 in-house A549 curves (Nuryadi 2018 Suppl. Table 2) with LQ and exact-model (best M) fits."""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc
o = json.load(open("nuryadi_inhouse_results.json")); C = o["curves"]
fig, axs = plt.subplots(4, 5, figsize=(11, 8.5), sharex=True, sharey=True); dd = np.linspace(0, 8.5, 120)
for ax, r in zip(axs.flat, C):
    S = np.array(r["SF"]); D = np.array([2, 4, 6, 8.])
    ax.plot(D[S > 0], S[S > 0], "o", mfc="white", color="k", ms=4)
    if (S == 0).any(): ax.plot(D[S == 0], [0.005]*int((S == 0).sum()), "v", color="k", ms=5)
    ax.plot(dd, np.exp(-r["LQ"]["alpha"]*dd - r["LQ"]["beta"]*dd*dd), ":", color="0.4")
    M = min((1, 2, 4), key=lambda m: r[f"exact_M{m}"]["nll"]); b = r[f"exact_M{M}"]["best"]
    ax.plot(dd, survival_disc(dd, b["lam"], 1, 2*b["F_r"], b["kappa_mu"], b["nc"], M), "-", color="k", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.text(0.3, 1.5e-3, f"#{r['exp']}  M={M}", fontsize=7)
for ax in axs[-1]: ax.set_xlabel("dose (Gy)")
for ax in axs[:, 0]: ax.set_ylabel("SF")
axs[0, 0].plot([], [], ":", color="0.4", label="LQ"); axs[0, 0].plot([], [], "-", color="k", label="exact model"); axs[0, 0].plot([], [], "v", color="k", label="SF8 < 0.005 (censored)")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle("A549, 20 in-house experiments (Nuryadi et al 2018, Suppl. Table 2; 100 kVp)", fontsize=10)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("fig_Nuryadi_inhouse.pdf"); fig.savefig("fig_Nuryadi_inhouse.png", dpi=130)
