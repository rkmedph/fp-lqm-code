"""Provenance-tracked registry of raw clonogenic survival curves used for independent model checks.

Every point keeps: source, location in source, cell line, radiation and energy, dose, SF,
reported uncertainty and its type, how the point was obtained, and notes.
Writes registry_points.csv (one row per dose point) and registry_curves.csv (one row per curve).
"""
import csv, numpy as np

REF = {
 "LI24": dict(ref="Li J et al 2024 Phys. Med. Biol. 69 205008", doi="10.1088/1361-6560/ad7d5a",
              location="Appendix A.2 (SF table); LQ fit in Table 2",
              method="table (transcribed from PDF text layer; values printed with 3 significant figures)",
              unc_type="SEM (error bars = standard error of the mean; >=3 technical replicates per dose)",
              assay="clonogenic, seeded immediately after irradiation, 7-14 d, colonies >50 cells, SF normalised by PE",
              fit_in_source="LQ, SEM-weighted, linear SF scale, Levenberg-Marquardt (OriginPro 2022)"),
 "MUR04": dict(ref="Murakami D et al 2004 Radiat. Environ. Biophys. 43 91-9", doi="10.1007/s00411-004-0239-6",
               location="Table 1 (colony counts, PE, SF); LQ fit in Results text",
               method="table (transcribed from PDF text layer; SF printed with 2 decimals)",
               unc_type="SD over 3 independent assays (each in triplicate); SEM = SD/sqrt(3) used for fitting",
               assay="clonogenic, 300 cells/plate at every dose, 10-14 d, colonies >=50 cells, SF normalised by PE",
               fit_in_source="LQ, method not stated (GraphPad Prism 2.0)"),
 "ES59": dict(ref="Elkind M M and Sutton H 1959 Nature 184 1293-5", doi="10.1038/1841293a0 (verified); same data (Exp. 87, clone A) = Fig. 7 of Elkind & Sutton 1960 Radiat Res 13:556-93",
              location="single-dose survival curve (figure), digitised from a reproduced image supplied by the author of this study",
              method="digitised from figure (not original data); x,y read error not quantified",
              unc_type="none reported; 10% relative error on SF assumed",
              assay="clonogenic (original 1959 protocol)", fit_in_source="none (multi-target description in original)"),
}

D4 = [2.0, 4.0, 6.0, 8.0]
LI24 = {  # (cell, beam): (SF, SEM)   -- Appendix A.2
 ("HCT116","6 MV"):   ([3.52e-1,1.05e-1,1.60e-2,3.00e-3],[2.90e-2,4.00e-3,1.00e-3,1.00e-3]),
 ("HCT116","192Ir"):  ([3.54e-1,8.10e-2,1.50e-2,2.00e-3],[2.00e-2,5.00e-3,1.00e-3,1.00e-3]),
 ("HCT116","225 kVp"):([3.67e-1,6.60e-2,7.00e-3,1.00e-3],[2.30e-2,6.00e-3,1.00e-3,1.00e-3]),
 ("HCT116","50 kVp"): ([1.98e-1,3.40e-2,3.00e-3,1.00e-3],[2.00e-2,2.00e-3,1.00e-3,1.00e-3]),
 ("HeLa","6 MV"):     ([5.15e-1,1.21e-1,3.40e-2,7.00e-3],[4.30e-2,3.50e-2,6.00e-3,1.00e-3]),
 ("HeLa","192Ir"):    ([5.32e-1,1.57e-1,2.40e-2,8.00e-3],[6.10e-2,2.20e-2,5.00e-3,2.00e-3]),
 ("HeLa","225 kVp"):  ([5.29e-1,1.23e-1,2.70e-2,4.00e-3],[2.60e-2,8.00e-3,3.00e-3,1.00e-3]),
 ("HeLa","50 kVp"):   ([4.13e-1,1.14e-1,1.70e-2,2.00e-3],[3.70e-2,1.70e-2,2.00e-3,1.00e-3]),
 ("PC3","6 MV"):      ([5.37e-1,1.94e-1,3.90e-2,5.00e-3],[5.90e-2,7.00e-2,2.40e-2,3.00e-3]),
 ("PC3","192Ir"):     ([5.29e-1,1.87e-1,3.80e-2,5.00e-3],[3.70e-2,2.10e-2,3.00e-3,1.00e-3]),
 ("PC3","225 kVp"):   ([4.04e-1,1.08e-1,1.80e-2,2.00e-3],[1.01e-1,2.60e-2,8.00e-3,1.00e-3]),
 ("PC3","50 kVp"):    ([3.89e-1,6.90e-2,1.10e-2,1.00e-3],[8.70e-2,1.00e-2,4.00e-3,1.00e-3]),
}
LI24_LQ = {  # Table 2 of Li et al: alpha (SE), beta (SE)
 ("HCT116","6 MV"):(0.388,0.038,0.047,0.007), ("HCT116","192Ir"):(0.468,0.017,0.038,0.003),
 ("HCT116","225 kVp"):(0.433,0.049,0.060,0.008), ("HCT116","50 kVp"):(0.667,0.022,0.047,0.004),
 ("HeLa","6 MV"):(0.300,0.052,0.040,0.007), ("HeLa","192Ir"):(0.238,0.098,0.058,0.019),
 ("HeLa","225 kVp"):(0.271,0.053,0.055,0.010), ("HeLa","50 kVp"):(0.359,0.031,0.050,0.005),
 ("PC3","6 MV"):(0.195,0.006,0.057,0.001), ("PC3","192Ir"):(0.183,0.013,0.061,0.002),
 ("PC3","225 kVp"):(0.348,0.008,0.053,0.001), ("PC3","50 kVp"):(0.452,0.032,0.051,0.005),
}
BEAM = {"6 MV":("photon","6 MV x-rays (linac)"), "192Ir":("photon","192Ir HDR brachytherapy gamma"),
        "225 kVp":("photon","225 kVp x-rays"), "50 kVp":("photon","50 kVp x-rays")}

# Monograph (dissertation) table, origin of CHO and A549 rows not yet documented
MONO = {"CHO":(0.151,0.037), "A549":(0.191,0.026), "HCT116":(0.388,0.047), "HeLa":(0.300,0.040), "PC3":(0.195,0.057)}


def build():
    curves = []
    for (cell, beam), (S, e) in LI24.items():
        a, sa, b, sb = LI24_LQ[(cell, beam)]
        curves.append(dict(curve_id=f"LI24_{cell}_{beam.replace(' ','')}", src="LI24", cell=cell, beam=beam,
            radiation=BEAM[beam][1], dose_rate="see source", D=list(D4), S=S, unc=e, unc_kind="SEM", n_rep="3 technical",
            counts=None, src_alpha=a, src_alpha_se=sa, src_beta=b, src_beta_se=sb))
    mur = {"60Co": ([0.80,0.55,0.31,0.05],[0.15,0.14,0.16,0.01],[211.6,146.0,81.2,14.0],"60Co gamma, 0.34 Gy/min", (0.195,0.061,0.014,0.014)),
           "90Sr/90Y": ([0.75,0.56,0.34,0.07],[0.05,0.10,0.02,0.01],[201.8,152.9,92.7,18.0],"90Sr/90Y beta, 0.23 Gy/min", (0.215,0.036,0.008,0.008))}
    for beam, (S, sd, cnt, rad, lq) in mur.items():
        curves.append(dict(curve_id=f"MUR04_CHO-K1_{beam.replace('/','')}", src="MUR04", cell="CHO-K1", beam=beam,
            radiation=rad, dose_rate=rad.split(", ")[1], D=[1.0,2.5,5.0,7.5], S=S, unc=sd, unc_kind="SD (n=3 assays)", n_rep="3 independent x triplicate",
            counts=cnt, src_alpha=lq[0], src_alpha_se=lq[1], src_beta=lq[2], src_beta_se=lq[3]))
    D = [0.90,1.82,3.22,5.00,6.62,8.34,9.91,11.44]; S = [0.77,0.60,0.275,0.085,0.0193,0.0061,0.00182,0.00051]
    curves.append(dict(curve_id="ES59_ChineseHamster_X", src="ES59", cell="Chinese hamster (Elkind-Sutton line)", beam="X-rays",
        radiation="X-rays (energy not recovered from the digitised image)", dose_rate="see source", D=D, S=S,
        unc=[0.1*s for s in S], unc_kind="assumed 10% relative", n_rep="unknown", counts=None,
        src_alpha=None, src_alpha_se=None, src_beta=None, src_beta_se=None))
    return curves


def sigma_lnS(c):
    """Uncertainty of ln SF used in fits: SEM-equivalent relative error plus rounding of the printed SF."""
    S = np.asarray(c["S"], float); u = np.asarray(c["unc"], float)
    if c["unc_kind"].startswith("SD"):
        u = u / np.sqrt(3.0)
    rnd = {"LI24": 0.005*S, "MUR04": np.full_like(S, 0.005), "ES59": 0*S}[c["src"]] / np.sqrt(3.0)  # half-unit of last digit, uniform
    return np.sqrt(u**2 + rnd**2) / S


if __name__ == "__main__":
    curves = build()
    with open("registry_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_id","cell_line","radiation","dose_Gy","SF","uncertainty","uncertainty_type","colonies_counted_mean",
                    "how_obtained","location_in_source","reference","doi"])
        for c in curves:
            R = REF[c["src"]]
            for i, (d, s, u) in enumerate(zip(c["D"], c["S"], c["unc"])):
                w.writerow([c["curve_id"], c["cell"], c["radiation"], d, s, round(u, 6), c["unc_kind"],
                            "" if c["counts"] is None else c["counts"][i], R["method"], R["location"], R["ref"], R["doi"]])
    with open("registry_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_id","cell_line","radiation","dose_rate","n_points","dose_range_Gy","replicates","uncertainty_type",
                    "assay","how_obtained","LQ_alpha_in_source","LQ_alpha_SE","LQ_beta_in_source","LQ_beta_SE","fit_method_in_source",
                    "monograph_table_alpha","monograph_table_beta","matches_monograph_row","reference","doi"])
        for c in curves:
            R = REF[c["src"]]; key = "CHO" if c["cell"].startswith("CHO") else c["cell"]
            mono = MONO.get(key)
            match = ""
            if mono and c["src_alpha"] is not None:
                match = "yes (identical to 3 decimals)" if (abs(mono[0]-c["src_alpha"]) < 5e-4 and abs(mono[1]-c["src_beta"]) < 5e-4) else "no"
            w.writerow([c["curve_id"], c["cell"], c["radiation"], c["dose_rate"], len(c["D"]), f"{min(c['D'])}-{max(c['D'])}",
                        c["n_rep"], c["unc_kind"], R["assay"], R["method"], c["src_alpha"], c["src_alpha_se"], c["src_beta"], c["src_beta_se"],
                        R["fit_in_source"], mono[0] if mono else "", mono[1] if mono else "", match, R["ref"], R["doi"]])
    print(len(curves), "curves,", sum(len(c["D"]) for c in curves), "points")
