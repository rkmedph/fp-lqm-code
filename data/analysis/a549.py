"""A549: independent check on 192 literature survival curves compiled by Nuryadi et al 2018 (Supplementary Table 1).
Values SF2, SF4, SF6, SF8 were read by Nuryadi et al from figures (or text/tables) of the original papers,
recorded with two decimals; no uncertainties are given.  Assumed uncertainty: sigma(SF) = 0.01 absolute
(read-off + rounding), alternative: 10 % relative on SF.
Outputs: registry_points_A549_literature.csv, table_A549_curves.csv, a549_results.json
"""
import sys, json, csv, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, ".")
from validation import lq_log, fit_model
from registry import MONO

SRC = "/home/claude/rawdata/nuryadi/s002.xlsx"
REF = "Nuryadi E et al 2018 Oncotarget 9 13706-12, Supplementary Table 1"
DOI = "10.18632/oncotarget.24448"
DOSES = np.array([2.0, 4.0, 6.0, 8.0])


def load():
    d = pd.read_excel(SRC, header=1).iloc[:, :13]
    d = d[pd.to_numeric(d["PMID"], errors="coerce").notna()].copy()
    d["PMID"] = d["PMID"].astype(int)
    for c in ("SF2", "SF4", "SF6", "SF8", "Dose rate*"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def sig(S, kind):
    return 0.01/S if kind == "abs" else np.full_like(S, 0.10)


def one(args):
    pmid, D, S = args
    y = -np.log(S); s = sig(S, "abs")
    return pmid, {f"M{M}": {k: v for k, v in fit_model(D, y, s, M).items() if k != "profile"} for M in (1, 2, 4)}


def main(run_model=True):
    d = load()
    # ---- registry
    with open("registry_points_A549_literature.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_id", "cell_line", "original_PMID", "year", "journal", "figure_in_original", "radiation", "dose_rate_Gy_per_min",
                    "cell_seeding", "dose_Gy", "SF", "uncertainty", "uncertainty_type", "how_obtained", "location_in_source", "reference", "doi"])
        for _, r in d.iterrows():
            for D, c in zip(DOSES, ("SF2", "SF4", "SF6", "SF8")):
                if pd.notna(r[c]):
                    w.writerow([f"NUR18_A549_{r['PMID']}", "A549", r["PMID"], r["Year"], r["Journal"], r["Data source"], r["Radiation type"],
                                "" if pd.isna(r["Dose rate*"]) else r["Dose rate*"], r["Cell seeding"], D, r[c], 0.01,
                                "none reported; 0.01 absolute assumed (2-decimal read-off)",
                                "read from figure of original paper (or text/table) by Nuryadi et al; second-hand digitisation",
                                "Supplementary Table 1", REF, DOI])
    # ---- per-curve LQ
    rows = []
    for _, r in d.iterrows():
        S = r[["SF2", "SF4", "SF6", "SF8"]].to_numpy(float); m = ~np.isnan(S)
        if m.sum() < 3: continue
        D, S = DOSES[m], S[m]; y = -np.log(S)
        rec = dict(curve_id=f"NUR18_A549_{r['PMID']}", PMID=int(r["PMID"]), radiation=r["Radiation type"], dose_rate=r["Dose rate*"],
                   seeding=r["Cell seeding"], n=int(m.sum()), D=D.tolist(), S=S.tolist())
        for kind in ("abs", "rel"):
            a, b, cov, c2 = lq_log(D, y, sig(S, kind))
            rec[f"LQ_{kind}"] = dict(alpha=a, beta=b, se_alpha=float(np.sqrt(cov[0, 0])), se_beta=float(np.sqrt(cov[1, 1])), chi2=c2)
        # model-free: -lnS/D must be non-decreasing for a homogeneous population (convexity of -lnS with -lnS(0)=0)
        g = y/D; sg = sig(S, "abs")/D
        dz = [(g[i+1]-g[i])/np.hypot(sg[i], sg[i+1]) for i in range(len(D)-1)]
        rec["min_z_slope_increment"] = float(min(dz))
        if m.all():  # range stability
            a24, b24, _, _ = lq_log(D[:2], y[:2], sig(S, "abs")[:2]); a26, b26, _, _ = lq_log(D[:3], y[:3], sig(S, "abs")[:3])
            rec["alpha_D<=4"], rec["beta_D<=4"], rec["alpha_D<=6"], rec["beta_D<=6"] = a24, b24, a26, b26
        rows.append(rec)
    res = dict(n_rows=len(d), curves=rows)
    # ---- exact model on the 4-point curves (0 dof: chi2_min > 0 means the model shape cannot pass through the points)
    if run_model:
        jobs = [(r["PMID"], np.array(r["D"]), np.array(r["S"])) for r in rows if r["n"] == 4]
        with Pool() as p:
            out = dict(p.map(one, jobs))
        for r in rows:
            if r["PMID"] in out: r["model"] = out[r["PMID"]]
    json.dump(res, open("a549_results.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    main(run_model=("--no-model" not in sys.argv))
