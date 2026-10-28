"""Within-lab beam-quality comparison: digitised Steenken 2015 (6/10 MV FLAT) and Verbakel 2013 (6 MV FF vs 10 MV FFF)."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from energy_data import STEENKEN, VERBAKEL
fig, axs = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
sty = {"6 MV FLAT": ("o", "white"), "10 MV FLAT": ("s", "k"), "6 MV FFF": ("^", "white"), "10 MV FFF": ("v", "0.5"), "6 MV FF": ("o", "white")}
for ax, cell in zip(axs[0], ("U251", "U251-MGMT", "U87")):
    for e in ("6 MV", "10 MV"):
        for m in ("FLAT", "FFF"):
            p = np.array(STEENKEN[(cell, e, m)]); mk, fc = sty[f"{e} {m}"]
            ax.errorbar(p[:, 0], 10**p[:, 1], yerr=[10**p[:, 1]*(1 - 10**-np.maximum(p[:, 2], .05)), 10**p[:, 1]*(10**np.maximum(p[:, 2], .05) - 1)],
                        fmt=mk, mfc=fc, color="k", ms=5, capsize=2, lw=0.7, label=f"{e} {m}")
    ax.set_title(f"{cell} - Steenken 2015 (Elekta, 5 Gy/min)", fontsize=9)
for ax, cell in zip(axs[1], ("SW1573", "D384", "T98")):
    for b, (mk, fc) in (("6 MV FF", ("o", "white")), ("10 MV FFF", ("v", "0.5"))):
        p = np.array(VERBAKEL[(cell, b)])
        ax.errorbar(p[:, 0], 10**p[:, 1], yerr=[10**p[:, 1]*(1 - 10**-p[:, 2]), 10**p[:, 1]*(10**p[:, 2] - 1)], fmt=mk, mfc=fc, color="k", ms=5, capsize=2, lw=0.7, label=b)
    ax.set_title(f"{cell} - Verbakel 2013 (TrueBeam)", fontsize=9); ax.set_xlabel("dose (Gy)")
for ax in axs.flat: ax.set_yscale("log"); ax.set_ylim(5e-4, 1.5)
for ax in axs[:, 0]: ax.set_ylabel("surviving fraction")
axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left"); axs[1, 0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("fig_energy_withinlab.pdf"); fig.savefig("fig_energy_withinlab.png", dpi=130)
