"""Figure S1 of the extended version: all 20 survival series with LQ and exact-model fits."""
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p)); break
else:
    sys.exit("не найден paths.py ни в одном родительском каталоге этого скрипта")
from paths import ROOT, require, load_json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT))
require(ROOT / "discrete.py", what="модуль discrete.py")
from discrete import survival_disc

SRC = ROOT / "arxiv" / "analysis" / "series_fits.json"
PDF = ROOT / "arxiv" / "figS1.pdf"
PNG = ROOT / "arxiv" / "analysis" / "figS1.png"
R = load_json(SRC, require_keys=("name", "D", "y", "s", "g", "K", "EX", "LQ"), unique_by="name")
for r in R:                       # согласованность длин до начала отрисовки
    n = len(r["D"])
    if not (len(r["y"]) == len(r["s"]) == len(r["g"]) == n):
        raise SystemExit(f'{SRC}: серия {r["name"]!r}: длины D/y/s/g не совпадают')
    if max(r["g"]) >= r["K"]:
        raise SystemExit(f'{SRC}: серия {r["name"]!r}: индекс пучка {max(r["g"])} при K={r["K"]}')
order = [r["name"] for r in R]
title = {}
for r in R:
    t = r["name"].replace("Howard2018", "Howard 2018").replace("Verbakel2013", "Verbakel 2013").replace("Li2024", "Li 2024")\
        .replace("Steenken2015", "Steenken 2015").replace("Butterworth2010", "Butterworth 2010").replace("Elkind-Sutton 1959", "Elkind–Sutton 1959")
    t = t.replace(" pooled 4 beams", ", 4 beams").replace(" pooled 6/10MV FLAT/FFF", ", 4 beams").replace(" pooled 6FF/10FFF", ", 2 beams")
    t = t.replace("6MV", "6 MV").replace("10 MV FFF", "10 MV FFF").replace("6 MV FF", "6 MV")
    title[r["name"]] = t
cols = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2"]
n = len(R); nc_ = 4; nr = int(np.ceil(n/nc_))
fig, axs = plt.subplots(nr, nc_, figsize=(10, 2.15*nr), sharex=False)
for ax, r in zip(axs.flat, R):
    D = np.array(r["D"]); y = np.array(r["y"]); s = np.array(r["s"]); g = np.array(r["g"]); K = r["K"]
    E = r["EX"]; L = np.array(r["LQ"]["par"])
    for j in range(K):
        m = g == j; c = cols[j % 4]
        S = np.exp(-y[m]); ax.errorbar(D[m], S, yerr=[S - np.exp(-y[m] - s[m]), np.exp(-y[m] + s[m]) - S], fmt="o", ms=3, color=c, lw=0.8, capsize=1.5)
        Dg = np.linspace(0, D.max()*1.05, 200)
        ax.plot(Dg, np.exp(-(L[j]*Dg + L[K + j]*Dg*Dg)), "--", color=c, lw=0.9)
        ax.plot(Dg, survival_disc(Dg, E["lam"][j], 1.0, 2*E["F"], E["k"], E["nc"], E["M"]), "-", color=c, lw=0.9)
    ax.set_yscale("log"); ax.set_title(title[r["name"]], fontsize=7.5); ax.tick_params(labelsize=7)
    lo = np.exp(-(y + s)).min(); ax.set_ylim(max(lo/3, 1e-5), 1.3)
for ax in axs.flat[n:]: ax.axis("off")
for ax in axs[-1]: ax.set_xlabel("Dose (Gy)", fontsize=8)
for ax in axs[:, 0]: ax.set_ylabel("Surviving fraction", fontsize=8)
fig.tight_layout(); PDF.parent.mkdir(parents=True, exist_ok=True); PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PDF); fig.savefig(PNG, dpi=110)
print("written", PDF, "и", PNG)
