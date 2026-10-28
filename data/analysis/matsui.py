"""Matsui et al 2019 IJMS 20:4148, Supplementary Data 1: SF2-SF8 for 8 cell lines read from published figures
(same method as Nuryadi 2018, 2 decimals, no uncertainties; A549 rows are the Nuryadi set).
LQ per curve (>=3 points, sigma(SF)=0.01), sign of beta, convexity, position of the monograph rows."""
import sys, json, csv, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from validation import lq_log

SRC = "/home/claude/rawdata/matsui/s001.xlsx"
DOSES = np.array([2.0, 4.0, 6.0, 8.0])
MONO = {"HCT-116": (0.388, 0.047), "PC-3": (0.195, 0.057), "A549": (0.191, 0.026)}


def load():
    d = pd.read_excel(SRC, sheet_name="SD1", header=1).iloc[:, :13]
    d = d[d["PMID"].notna()].copy()
    for c in ("SF2", "SF4", "SF6", "SF8", "Dose Rate (Gy/min)"): d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def main():
    d = load(); rows = []
    with open("registry_points_Matsui_literature.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_id", "cell_line", "original_PMID", "figure_in_original", "radiation", "dose_rate_Gy_per_min", "cell_seeding", "dose_Gy", "SF",
                    "uncertainty", "uncertainty_type", "how_obtained", "location_in_source", "reference", "doi"])
        for i, r in d.iterrows():
            cid = f"MAT19_{r['Cell Line']}_{r['PMID']}_{i}"
            for D, c in zip(DOSES, ("SF2", "SF4", "SF6", "SF8")):
                if pd.notna(r[c]):
                    w.writerow([cid, r["Cell Line"], r["PMID"], r["Data Source"], r["Radiation Type"], "" if pd.isna(r["Dose Rate (Gy/min)"]) else r["Dose Rate (Gy/min)"],
                                r["Timing of Cell Seeding"], D, r[c], 0.01, "none reported; 0.01 absolute assumed",
                                "read from figure of original paper by Matsui et al; second-hand digitisation", "Supplementary Data 1",
                                "Matsui T et al 2019 Int. J. Mol. Sci. 20 4148", "10.3390/ijms20174148"])
            S = r[["SF2", "SF4", "SF6", "SF8"]].to_numpy(float); m = ~np.isnan(S)
            if m.sum() < 3: continue
            D, S = DOSES[m], S[m]; y = -np.log(S); s = 0.01/S
            a, b, cov, c2 = lq_log(D, y, s)
            g = y/D; sg = s/D; dz = min((g[k+1]-g[k])/np.hypot(sg[k], sg[k+1]) for k in range(len(D)-1))
            rows.append(dict(cell=r["Cell Line"], PMID=str(r["PMID"]), n=int(m.sum()), D=D.tolist(), S=S.tolist(), alpha=a, beta=b,
                             se_alpha=float(np.sqrt(cov[0, 0])), se_beta=float(np.sqrt(cov[1, 1])), chi2=c2, min_z_slope=float(dz)))
    summ = {}
    for cell in sorted(set(r["cell"] for r in rows)):
        R = [r for r in rows if r["cell"] == cell]
        a = np.array([r["alpha"] for r in R]); b = np.array([r["beta"] for r in R]); zb = b/np.array([r["se_beta"] for r in R])
        z = np.array([r["min_z_slope"] for r in R])
        s = dict(n_curves=len(R), alpha_median=float(np.median(a)), alpha_IQR=np.percentile(a, [25, 75]).tolist(),
                 beta_median=float(np.median(b)), beta_IQR=np.percentile(b, [25, 75]).tolist(),
                 n_beta_neg=int((b < 0).sum()), n_beta_neg_2SE=int((zb < -2).sum()), n_convexity_z_lt_3=int((z < -3).sum()))
        if cell in MONO:
            am, bm = MONO[cell]
            s.update(mono=MONO[cell], pct_alpha_le_mono=float(100*(a <= am).mean()), pct_beta_le_mono=float(100*(b <= bm).mean()))
            sf = {}
            for D in DOSES:
                lit = [S for r in R for dd, S in zip(r["D"], r["S"]) if dd == D]
                p = float(np.exp(-am*D - bm*D*D))
                if lit: sf[f"{D:g}"] = dict(pred=p, lit_median=float(np.median(lit)), percentile=float(100*np.mean(np.array(lit) <= p)), n=len(lit))
            s["SF_monograph_vs_literature"] = sf
            s["near_monograph"] = [(r["PMID"], r["S"], round(r["alpha"], 3), round(r["beta"], 4)) for r in R
                                   if abs(r["alpha"] - am) <= 0.03 and abs(r["beta"] - bm) <= 0.008]
        summ[cell] = s
    json.dump(dict(curves=rows, summary=summ), open("matsui_results.json", "w"), indent=1, default=float)
    for c, s in summ.items():
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("SF_monograph_vs_literature",)})
        if "SF_monograph_vs_literature" in s:
            print("    SF pred vs lit:", {k: (round(v["pred"], 3), v["lit_median"], round(v["percentile"])) for k, v in s["SF_monograph_vs_literature"].items()})


if __name__ == "__main__":
    main()
