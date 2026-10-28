"""Registry of megavoltage (6/10/15 MV, FF/FFF) photon survival data with suitability tiers A/B/C, plus point table."""
import csv, json

ROWS = [
 ("A", "Howard M et al 2017 J Radiat Res 58:608-13", "CHO; A549; U251; T98; U2OS", "Varian TrueBeam", "6 MV FF (vs 137Cs)", "600 cGy/min nominal (360 cGy/min at 10 cm)",
  "see Fig. 2", "figure", "SD; 3 independent runs x triplicate", "Table 1: LQ alpha, beta (± SD) for 6 MV and 137Cs; Fig. 2 curves",
  "SOURCE of monograph rows CHO (0.151/0.037) and A549 (0.191/0.026) - identical to the 6 MV column of Table 1", "10.1093/jrr/rrx018"),
 # tier, study, cell lines, linac, energy/mode, dose rate, doses, n points, uncertainties, data form, status, doi
 ("A", "Li J et al 2024 Phys Med Biol 69:205008", "HCT116; HeLa; PC3", "Varian TrueBeam", "6 MV FF", "see source", "2,4,6,8 Gy", "4", "SEM (>=3 technical replicates)",
  "table (Appendix A.2)", "in use; monograph rows HCT116/HeLa/PC3 = these LQ fits", "10.1088/1361-6560/ad7d5a"),
 ("A", "Howard ME et al 2018 Int J Part Ther 4(3):12-22", "CHO; A549; T98", "Varian TrueBeam", "6 MV FF", "600 cGy/min", "1,2,4,5,6,8,10 Gy", "7",
  "error bars (type not stated); 3 independent runs x triplicate", "figure 3 (digitised here); LQ fits in Supplementary Table 1 (not in hand)",
  "digitised; same group, linac and protocol as Howard 2017 (likely the same 6 MV control data); refit A549 0.191/0.0235, CHO 0.134/0.038 consistent with Howard 2017 Table 1", "10.14338/IJPT-17-00031.1"),
 ("A*", "Kalash R et al 2014 In Vivo 28:1-12", "3LL (murine Lewis lung)", "Varian TrueBeam STx", "6 MV (300, 1400 MU/min); 10 MV (300, 2400 MU/min)", "300-2400 MU/min",
  "1-8 Gy (integer)", "8 per series", "error bars; triplicate", "scanned colour figure (Fig. 1) + Table I (D0, n)",
  "automatic digitisation failed (hollow, overlapping, washed-out markers); usable now only as tier C (D0, n); cells irradiated in suspension", "PMC author manuscript nihms581957"),
 ("A (parameters)", "Dini V et al 2023 Int J Mol Sci 24:1107", "V79; GSC #1, #83", "Varian TrueBeam", "10 MV FF and FFF", "4 and 24 Gy/min", "V79 1-10 Gy; GSC 5-40 Gy",
  "figure", "3 independent experiments (V79)", "Table 1: LQ alpha, beta (± SE) early survival, V79, 10 MV FF 4 Gy/min 0.084/0.025, FFF 4 Gy/min 0.124/0.019, FFF 24 Gy/min 0.166/0.017; Fig. 1 points not digitised (overlapping)", "parameters in use; GSC two-population fit (f = 0.56-0.93) relevant to heterogeneity", "10.3390/ijms24021107"),
 ("B", "Butterworth KT et al 2010 Phys Med Biol 55:1607-18", "DU-145; T98G; AGO-1522B", "Varian 600CD", "6 MV", "see source",
  "DU-145, T98G: 0-8 Gy (6 doses); AGO: 0-6.4 Gy (9 doses)", "3 tabulated (+ figure)", "SEM; triplicate x >=3 experiments",
  "Table 3 (SF at 2,4,8 Gy) + Fig. 4 (digitised for AGO-1522B)", "in use; figure digitisation checked against table (<=8 %, AGO 4 Gy 13 %)", "10.1088/0031-9155/55/6/005"),
 ("B", "Polgár 2022 database: Radiat Oncol 2008 3:19 (PC-3, LNCaP); HFIB fibroblasts (IJROBP 2014, DNA Repair 2016/2021); HeLa (JCDR 2015)",
  "PC-3; LNCaP; HFIB; HeLa", "not stated in database", "6 MV ('6 MeV' in database)", "2.4 Gy/min; 300 MU/min; 200 MU/min", "0.05-4 Gy", "7-14",
  "whiskers (SD/SEM unspecified)", "digitised by Polgár et al", "in use (low-dose test); <=4 Gy", "10.1038/s41597-022-01653-3"),
 ("C", "Lasio G et al 2014 Radiat Environ Biophys 53:671-6", "T98G; V79-4; U87-MG", "Varian TrueBeam", "6X, 6X FFF, 10X FFF", "4, 14, 24 Gy/min",
  "5 and 10 Gy", "2", "mean ± SD, 5 independent experiments", "bar charts (Figs 2-3)", "not yet digitised; tests dose-per-pulse / dose-rate only", "10.1007/s00411-014-0565-2"),
 ("C", "Kalash R et al 2014 (Table I)", "3LL", "Varian TrueBeam STx", "6/10 MV", "300-2400 MU/min", "0-8 Gy", "-", "SEM of D0, n", "table (single-hit multi-target D0, n)",
  "D0 1.7-2.0 Gy, n 6.8-13.3; no energy/dose-rate effect", "see above"),
 ("B", "Steenken C et al 2015 Strahlenther Onkol 191:590-6", "U251; U251-MGMT; U87", "Elekta Versa HD (Agility)", "6 and 10 MV, FLAT and FFF", "5 Gy/min (Fig. 2)",
  "measured doses ~1.9, 5.3, 7.6, 11.5 Gy (Table 1)", "4 per curve", "SEM (n = 3)", "Fig. 2 read manually (~54 px/decade, JPEG; ~0.03-0.05 dex)",
  "digitised; within-lab 6 vs 10 MV and FLAT vs FFF test; RBE FFF/FLAT 1.027 (6 MV), 1.063 (10 MV) reported", "10.1007/s00066-015-0823-5"),
 ("A", "Verbakel WFAR et al 2013 Acta Oncol 52:652-7", "SW1573; D384; T98", "Varian TrueBeam", "6 MV FF vs 10 MV FFF", "600 vs 2400 MU/min (5.86 vs 24 Gy/min)",
  "2,4,6,8,10,12 Gy", "6", "SEM (n = 3)", "Fig. 3 read manually from gridded zoom (0.05-dex grid)", "digitised; within-lab 6 MV FF vs 10 MV FFF test", "10.3109/0284186X.2012.737021"),
 ("C", "Lohse I et al 2011 Radiother Oncol 101:226-32", "2 glioblastoma lines", "not named", "FFF vs flat", "6-24 Gy/min", "5, 10 Gy", "2", "-", "figures", "PDF still needed (uploaded RO 101(1) archive held other articles)", "10.1016/j.radonc.2011.05.072"),
 ("C", "Ceylan C et al 2026 Life 16:903", "A549; MCF-7", "Varian Trilogy", "6 MV FF, 6 MV FFF, 10 MV FFF (VMAT)", "600-2400 MU/min", "5, 10, 20 Gy", "3",
  "SD stated in methods but not given", "SF in text only (PDF in hand)", "low priority", "10.3390/life16060903"),
 ("C", "Gruda M et al 2009 Rep Pract Oncol Radiother 14:85-8", "K-562", "not named", "6 MV vs 20 MV", "two rates", "2 Gy", "1", "-", "text", "low priority", "10.1016/S1507-1367(10)60099-5"),
 ("C", "PIDE photon references (GSI fits only)", "A-549 (kamlah11, 6 MV); 8 uveal melanoma lines (calipel15, 6 MV); HFIB (slonina14, 6 MV); V79 (boehrnsen02, 6 MV); ONS76, MOLT4 (gerelchuluun11, 10 MV); CGL1 (bettega09, 15 MV)",
  "various", "6/10/15 MV", "-", "-", "-", "-", "alpha, beta only", "raw points need GSI registration or digitisation", "Friedrich et al 2013, 2021"),
]

if __name__ == "__main__":
    with open("registry_MV_studies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "study", "cell_lines", "linac", "energy_mode", "dose_rate", "doses", "n_points", "uncertainty", "data_form", "status", "doi"])
        w.writerows(ROWS)
    H = json.load(open("digitize/howard_curves.json"))
    with open("registry_points_MV.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve_id", "tier", "cell_line", "linac", "energy_mode", "dose_rate", "dose_Gy", "SF", "sigma_lnSF_used", "how_obtained", "reference", "doi"])
        for c in H:
            for p in c["points"]:
                w.writerow([c["curve_id"], "A", c["cell"], c["linac"], c["energy"], c["dose_rate"], p["D"], round(p["SF"], 4), round(p["sigma_lnSF"], 3),
                            "digitised from Fig. 3 (" + p["method"] + "); error = plotted bar (+) 2 % digitisation" + (" (+) 5 % occlusion" if "hidden" in p["method"] else ""),
                            c["source"], c["doi"]])
        but = {"DU-145": [(2, .79, .04), (4, .49, .03), (8, .11, .01)], "T98G": [(2, .86, .03), (4, .58, .03), (8, .18, .01)], "AGO-1522B": [(2, .36, .02), (4, .10, .01)]}
        for cell, rows in but.items():
            for d, s, e in rows:
                w.writerow([f"BUT10_{cell}_6MV", "B", cell, "Varian 600CD", "6 MV", "", d, s, round(e/s, 3), "Table 3 (mean ± SEM)", "Butterworth KT et al 2010 Phys Med Biol 55:1607", "10.1088/0031-9155/55/6/005"])
        for d, s in [(0.33, 0.793), (0.5, 0.694), (4.67, 0.059), (6.3, 0.016)]:
            w.writerow(["BUT10_AGO-1522B_6MV", "B", "AGO-1522B", "Varian 600CD", "6 MV", "", d, s, 0.104, "digitised Fig. 4 (filled squares; 10 % assumed (+) 3 % digitisation)",
                        "Butterworth KT et al 2010 Phys Med Biol 55:1607", "10.1088/0031-9155/55/6/005"])
    print("ok")
