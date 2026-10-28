"""MV photon curves (TrueBeam / clinical linac): LQ vs exact threshold model (M = 1, 2, 4) with one likelihood:
ln SF ~ Normal(model, sigma^2), sigma = reported relative error (+) digitisation error.  Fits on all points and on D >= 2 Gy.
Inputs: digitize/howard_curves.json, Butterworth et al 2010 PMB 55:1607 (Table 3 + Fig. 4 digitised)."""
import sys, json, csv
import numpy as np
from scipy.stats import chi2 as c2d
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis")
from validation import lq_log, fit_model

MONO = {"A549": (0.191, 0.026), "CHO": (0.151, 0.037)}


def curves():
    C = json.load(open("/home/claude/rawdata/analysis/digitize/howard_curves.json"))
    tier = "A"
    for c in C: c["tier"] = tier
    B = dict(source="Butterworth KT et al 2010 Phys Med Biol 55:1607-18, Table 3 (SF +/- SEM) and Fig. 4 (digitised)", doi="10.1088/0031-9155/55/6/005",
             linac="Varian 600CD", energy="6 MV", dose_rate="see source", replicates="triplicate, >=3 independent experiments", tier="B")
    tab = {"DU-145": [(2, .79, .04), (4, .49, .03), (8, .11, .01)], "T98G": [(2, .86, .03), (4, .58, .03), (8, .18, .01)],
           "AGO-1522B": [(2, .36, .02), (4, .10, .01)]}
    dig = {"AGO-1522B": [(0.33, 0.793), (0.5, 0.694), (4.67, 0.059), (6.3, 0.016)]}
    for cell, rows in tab.items():
        pts = [dict(D=float(d), SF=s, sigma_lnSF=float(np.hypot(e/s, 0.0)), method="Table 3 (SEM)") for d, s, e in rows]
        for d, s in dig.get(cell, []):
            pts.append(dict(D=d, SF=s, sigma_lnSF=float(np.hypot(0.10, 0.03)), method="digitised Fig. 4 (error bar not resolved; 10 % assumed)"))
        pts.sort(key=lambda p: p["D"])
        C.append(dict(curve_id=f"BUT10_{cell}_6MV", cell=cell, points=pts, **B))
    return C


def fit(D, y, s):
    a, b, cov, c2 = lq_log(D, y, s); r = dict(LQ=dict(alpha=a, beta=b, se_alpha=float(np.sqrt(cov[0, 0])), se_beta=float(np.sqrt(cov[1, 1])),
                                                       chi2=c2, dof=len(D) - 2, AIC=c2 + 4))
    if len(D) >= 4:
        for M in (1, 2, 4):
            f = fit_model(D, y, s, M)
            r[f"M{M}"] = dict(chi2=f["chi2"], dof=len(D) - 4, AIC=f["chi2"] + 8, nc=f["best"]["nc"], F_r=f["best"]["F_r"], kappa_mu=f["best"]["kappa_mu"],
                              alpha0_range=f["alpha0_range"], beta0_range=f["beta0_range"], at_bound=f["best"]["at_bound"])
    return r


def main():
    res = []
    for c in curves():
        D = np.array([p["D"] for p in c["points"]]); S = np.array([p["SF"] for p in c["points"]]); s = np.array([p["sigma_lnSF"] for p in c["points"]])
        y = -np.log(S); rec = dict(curve_id=c["curve_id"], cell=c["cell"], tier=c["tier"], linac=c["linac"], n=len(D), fits={})
        rec["fits"]["all"] = fit(D, y, s)
        m = D >= 2
        if m.sum() >= 3: rec["fits"]["D>=2"] = fit(D[m], y[m], s[m])
        if c["cell"] in MONO:
            am, bm = MONO[c["cell"]]; z = (y[m] - (am*D[m] + bm*D[m]**2))/s[m]
            rec["monograph"] = dict(alpha=am, beta=bm, chi2_D_ge_2=float(z @ z), n=int(m.sum()), p=float(c2d.sf(z @ z, int(m.sum()))), z=z.round(2).tolist())
        res.append(rec)
        for k, f in rec["fits"].items():
            L = f["LQ"]
            line = f"{c['curve_id']:24s} {k:5s} n={int((D >= 2).sum() if k != 'all' else len(D))} LQ a={L['alpha']:.3f}±{L['se_alpha']:.3f} b={L['beta']:.4f}±{L['se_beta']:.4f} chi2={L['chi2']:.2f}/{L['dof']}"
            for M in (1, 2, 4):
                if f"M{M}" in f: line += f" | M{M} chi2={f[f'M{M}']['chi2']:.2f}/{f[f'M{M}']['dof']} a0={f[f'M{M}']['alpha0_range'][0]:.2f}-{f[f'M{M}']['alpha0_range'][1]:.2f}"
            print(line)
        if "monograph" in rec: print("    monograph", rec["monograph"])
    json.dump(res, open("mv_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
