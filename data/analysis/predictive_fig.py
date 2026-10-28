import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("predictive_summary.json"))
lab = [r["dataset"].replace("pooled 6/10MV FLAT/FFF", "pooled").replace("pooled 6FF/10FFF", "pooled").replace("pooled 4 beams", "pooled") for r in R]
y = np.arange(len(R))[::-1]
fig, ax = plt.subplots(1, 3, figsize=(11, 6.2), sharey=True)
ax[0].plot([r["dAIC_x1.0"] for r in R], y, "o", color="k", mfc="k", label=r"errors $\times1$")
ax[0].plot([r["dAIC_x2.0"] for r in R], y, "o", color="k", mfc="white", label=r"errors $\times2$")
for x in (-2, 2): ax[0].axvline(x, ls=":", color="0.5")
ax[0].axvline(0, color="0.7", lw=.6); ax[0].set_xlabel(r"$\Delta$AIC (exact $-$ LQ)"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=7); ax[0].set_title("(a) in-sample", fontsize=9)
lr = [r["loo_mz2_exact"]/r["loo_mz2_LQ"] if "loo_mz2_LQ" in r else np.nan for r in R]
ax[1].plot(lr, y, "s", color="k", mfc="white"); ax[1].axvline(1, color="0.5", ls=":"); ax[1].set_xscale("log")
ax[1].set_xlabel(r"LOO mean $z^2$: exact / LQ"); ax[1].set_title("(b) leave-one-dose-out", fontsize=9)
for t, mk, l in (("hi2lo", "v", r"fit $D\geq4$, predict $D<4$"), ("lo2hi", "^", r"fit $D\leq6.7$, predict higher")):
    v = [r[f"{t}_mz2_exact"]/r[f"{t}_mz2_LQ"] if f"{t}_mz2_LQ" in r and r[f"{t}_mz2_LQ"] > 0 else np.nan for r in R]
    ax[2].plot(v, y, mk, color="k", mfc="white" if t == "lo2hi" else "k", label=l)
ax[2].axvline(1, color="0.5", ls=":"); ax[2].set_xscale("log"); ax[2].set_xlabel(r"extrapolation mean $z^2$: exact / LQ")
ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("(c) extrapolation", fontsize=9)
for a in ax[1:]: a.text(0.02, 0.01, "exact better", transform=a.transAxes, fontsize=7, color="0.4"); a.text(0.98, 0.01, "LQ better", transform=a.transAxes, fontsize=7, color="0.4", ha="right")
fig.tight_layout(); fig.savefig("fig_model_comparison.pdf"); fig.savefig("fig_model_comparison.png", dpi=130)
