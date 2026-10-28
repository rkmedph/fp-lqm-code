"""Summaries and figure from identifiability_results.json."""
import json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("identifiability_results.json"))
rows = []
for name, d in R.items():
    F = d["fits"]; cmin = min(f["chi2"] for f in F)
    rec = dict(dataset=name, n=len(d["D"]), chi2_LQ=d["chi2_LQ"], chi2_exact_min=cmin)
    prof = {}
    for f in F:
        key = (f["M"], f["nc"]); prof[key] = min(prof.get(key, np.inf), f["chi2"])
    for lev in (1, 4):
        sel = [f for f in F if f["chi2"] <= cmin + lev]
        rec[f"nc_set_d{lev}"] = sorted(set(f["nc"] for f in sel)); rec[f"M_set_d{lev}"] = sorted(set(f["M"] for f in sel))
        for q in ("alpha0", "beta0", "lam", "F", "k"):
            v = np.array([f[q] for f in sel]); rec[f"{q}_d{lev}"] = [float(v.min()), float(v.max())]
        ab = np.array([f["alpha0"]/f["beta0"] if f["beta0"] > 0 else np.inf for f in sel]); rec[f"ab_d{lev}"] = [float(ab.min()), float(ab.max())]
    rec["dchi2_nc20"] = min(prof[(M, 20)] for M in (1, 2)) - cmin
    rec["dchi2_by_nc"] = {nc: min(prof[(M, nc)] for M in (1, 2)) - cmin for nc in range(2, 31)}
    if "truth" in d: rec["truth"] = {k: d["truth"][k] for k in ("M", "nc", "lam", "F", "k", "alpha0", "beta0")}
    rows.append(rec)
    print(f"\n{name}: n={rec['n']}  chi2 LQ {rec['chi2_LQ']:.2f}  exact min {cmin:.2f}  dchi2(nc=20) {rec['dchi2_nc20']:.2f}")
    for lev in (1, 4):
        print(f"  d<={lev}: nc {rec[f'nc_set_d{lev}'][0]}-{rec[f'nc_set_d{lev}'][-1]} ({len(rec[f'nc_set_d{lev}'])} values), M {rec[f'M_set_d{lev}']}, "
              f"alpha0 {np.round(rec[f'alpha0_d{lev}'], 3)}, beta0 {np.round(rec[f'beta0_d{lev}'], 4)}, a0/b0 {np.round(rec[f'ab_d{lev}'], 2)}, "
              f"F_r {np.round(rec[f'F_d{lev}'], 3)}, kappa/mu {np.round(rec[f'k_d{lev}'], 4)}, lam {np.round(rec[f'lam_d{lev}'], 3)}")
    if "truth" in rec: print("  truth:", {k: round(float(v), 4) for k, v in rec["truth"].items()})
json.dump(rows, open("ident_summary.json", "w"), indent=1, default=float)

# figure: 2-D profiles (kappa, nc) and (F_r, nc) for the A549 curve and for the ideal synthetic curve; nc profiles for all
names = ["Howard2018 A549 6MV", [k for k in R if k.startswith("Ideal")][0]]
fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
for row, name in enumerate(names):
    F = R[name]["fits"]; cmin = min(f["chi2"] for f in F)
    for col, (which, key, lab, sc) in enumerate((("kappa", "k", r"$\kappa/\mu$", "log"), ("F", "F", r"$F_r-1/2$", "log"))):
        ax = axs[row, col]
        grid = {}
        for f in F:
            if f["which"] != which: continue
            k2 = (f[key], f["nc"]); grid[k2] = min(grid.get(k2, np.inf), f["chi2"] - cmin)
        xs = sorted(set(k for k, _ in grid)); ncs = sorted(set(n for _, n in grid))
        Z = np.array([[grid[(x, n)] for x in xs] for n in ncs])
        xv = np.array(xs) - (0.5 if which == "F" else 0)
        cs = ax.contourf(xv, ncs, np.minimum(Z, 12), levels=[0, 1, 2.3, 4, 6, 9, 12], cmap="Greys_r")
        ax.contour(xv, ncs, Z, levels=[1, 4], colors="k", linewidths=[1.2, .8], linestyles=["-", "--"])
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel(r"$n_c$")
        ax.set_title("(%s) %s" % ("abde"[2*row + col], name.split(" (")[0]), fontsize=8)
    fig.colorbar(cs, ax=axs[row, 1], label=r"$\Delta\chi^2$ (profiled)")
    ax = axs[row, 2]
for name, mk in zip(R, ["o", "s", "^", "D", "x"]):
    r = [x for x in rows if x["dataset"] == name][0]
    axs[0, 2].plot(list(range(2, 31)), [r["dchi2_by_nc"][n] for n in range(2, 31)], marker=mk, ms=3, color="k", lw=.8, label=name.split(" (")[0])
axs[0, 2].axhline(1, ls=":", color="0.5"); axs[0, 2].axhline(4, ls="--", color="0.5"); axs[0, 2].set_yscale("symlog", linthresh=1); axs[0, 2].set_ylim(0, 600); axs[0, 2].set_xlabel(r"$n_c$")
axs[0, 2].set_ylabel(r"$\Delta\chi^2$"); axs[0, 2].legend(fontsize=6, frameon=False); axs[0, 2].set_title("(c) profile in $n_c$ (M = 1, 2)", fontsize=8)
# alpha0: all grid-optimised fits (the lower edge of each cloud approximates the profile)
for name, mk, col in [("Howard2018 A549 6MV", "o", "0.55"), ([k for k in R if k.startswith("Ideal")][0], "x", "k")]:
    F = R[name]["fits"]; cmin = min(f["chi2"] for f in F)
    a = np.array([f["alpha0"] for f in F]); c = np.array([f["chi2"] - cmin for f in F]); m = c < 12
    axs[1, 2].plot(a[m], c[m], mk, ms=3, mfc="none", color=col, lw=0, label=name.split(" (")[0])
axs[1, 2].legend(fontsize=6, frameon=False, markerscale=2)
axs[1, 2].axhline(1, ls=":", color="0.5"); axs[1, 2].axhline(4, ls="--", color="0.5"); axs[1, 2].set_ylim(-0.2, 12); axs[1, 2].set_xlabel(r"small-dose $\alpha_0$ (Gy$^{-1}$)")
axs[1, 2].set_ylabel(r"$\Delta\chi^2$ of grid-optimised fits"); axs[1, 2].set_title(r"(f) small-dose $\alpha_0$ of near-optimal fits", fontsize=8)
fig.tight_layout(); fig.savefig("fig_identifiability.pdf"); fig.savefig("fig_identifiability.png", dpi=130)
