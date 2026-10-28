#!/usr/bin/env python3
"""
Analysis of PIDE raw photon survival data for the lethal-threshold model.

Input
  A whitespace/comma/semicolon-separated text file with one row per data point.
  Columns are given by name (--id-col, --dose-col, --sf-col) or by zero-based index.
  Lines starting with '#' are ignored. Example: PIDE3.4_PhotonRawData.dat.
  The exact column layout of the PIDE file is described in its Information_sheet.pdf;
  check it and pass the right column names/indices.

Per survival curve (photon experiment) the script reports
  1. Full-range LQ fit of -ln S (unweighted least squares, as in PIDE): alpha, beta,
     standard errors, and the z-score of beta (negative beta = sign test of convexity).
  2. Low-dose LQ fit (D <= --dlow, at least --min-low points): small-dose coefficients
     and the bound ratio R = alpha^2/(2 beta) with a delta-method standard error.
     Only these coefficients may be compared with Corollary 3 (beta >= alpha^2/2M).
  3. Optional (--model-fit): best fit of the exact discrete model for M in {1,2,4,10}
     and comparison with LQ by AIC.
Output: <out>_curves.csv and a short summary on stdout.
"""
import argparse, sys, numpy as np, pandas as pd

def read_table(path, sep=None):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(line)
    from io import StringIO
    txt = "".join(rows)
    if sep is None:
        sep = ";" if txt.count(";") > txt.count(",") and ";" in txt else ("," if "," in txt.split("\n")[0] else r"\s+")
    try:
        df = pd.read_csv(StringIO(txt), sep=sep, engine="python")
        float(df.iloc[0, -1])  # header present if this works on data row
    except Exception:
        df = pd.read_csv(StringIO(txt), sep=sep, engine="python", header=None)
    return df

def col(df, spec):
    if spec is None: return None
    try: return df.iloc[:, int(spec)]
    except ValueError: return df[spec]

def lq_fit(D, y):
    X = np.vstack([D, D**2]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(D) - 2
    if dof <= 0: return coef, np.full((2, 2), np.nan), np.nan
    res = y - X @ coef; s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov, s2

def bound_ratio(a, b, cov):
    if not (b > 0): return np.inf, np.nan
    R = a*a/(2*b)
    g = np.array([a/b, -a*a/(2*b*b)])
    return R, float(np.sqrt(max(g @ cov @ g, 0)))

def model_fit(D, y, Ms=(1, 2, 4, 10), ncs=range(2, 21)):
    from scipy.optimize import least_squares
    from discrete import survival_disc
    best = {}
    for M in Ms:
        bm = None
        for nc in ncs:
            for p0 in ([np.log(0.5/M), 0.0, np.log(0.3)], [np.log(2.0/M), np.log(3), np.log(1.0)]):
                f = lambda p: -np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1.0, 2*(0.5+np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y
                r = least_squares(f, p0, bounds=(np.log([1e-3, 1e-3, 1e-4]), np.log([100, 100, 100])))
                rss = float(r.fun @ r.fun)
                if bm is None or rss < bm[0]: bm = (rss, nc)
        best[M] = bm
    return best

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw"); ap.add_argument("--id-col", required=True); ap.add_argument("--dose-col", required=True)
    ap.add_argument("--sf-col", required=True); ap.add_argument("--sep", default=None)
    ap.add_argument("--dlow", type=float, default=3.0); ap.add_argument("--min-low", type=int, default=4)
    ap.add_argument("--min-points", type=int, default=4); ap.add_argument("--model-fit", action="store_true")
    ap.add_argument("--out", default="pide_raw")
    a = ap.parse_args()
    df = read_table(a.raw, a.sep)
    ids, Dall, Sall = col(df, a.id_col), col(df, a.dose_col).astype(float), col(df, a.sf_col).astype(float)
    out = []
    for cid, g in pd.DataFrame({"id": ids, "D": Dall, "S": Sall}).groupby("id"):
        g = g[(g.D > 0) & (g.S > 0) & (g.S <= 1.5)].sort_values("D")
        if len(g) < a.min_points: continue
        D, y = g.D.values, -np.log(g.S.values)
        (al, be), cov, s2 = lq_fit(D, y)
        rec = dict(id=cid, n=len(D), Dmax=D.max(), alpha=al, beta=be,
                   se_alpha=np.sqrt(cov[0, 0]), se_beta=np.sqrt(cov[1, 1]), z_beta=be/np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan)
        lo = D <= a.dlow
        if lo.sum() >= a.min_low:
            (al0, be0), cov0, _ = lq_fit(D[lo], y[lo])
            R, sR = bound_ratio(al0, be0, cov0)
            rec.update(alpha_low=al0, beta_low=be0, se_beta_low=np.sqrt(cov0[1, 1]), R_low=R, se_R_low=sR)
        if a.model_fit:
            n = len(D); rss_lq = float(((y - (al*D + be*D*D))**2).sum())
            rec["aic_lq"] = n*np.log(rss_lq/n) + 2*2
            for M, (rss, nc) in model_fit(D, y).items():
                rec[f"aic_M{M}"] = n*np.log(rss/n) + 2*4; rec[f"nc_M{M}"] = nc
        out.append(rec)
    res = pd.DataFrame(out); res.to_csv(f"{a.out}_curves.csv", index=False)
    print(f"curves analysed: {len(res)}")
    print(f"full-range LQ: beta<0 in {(res.beta<0).sum()} ({(res.beta<0).mean():.1%}); "
          f"significantly negative (z<-2) in {(res.z_beta<-2).sum()}")
    if "R_low" in res:
        r = res.dropna(subset=["R_low"])
        sig = (r.R_low - 2*r.se_R_low) > 1
        print(f"low-dose fits (D<={a.dlow} Gy, >= {a.min_low} points): {len(r)}; "
              f"R_low>1: {(r.R_low>1).sum()}; R_low-2se>1 (evidence for M>=2): {sig.sum()}")
    if a.model_fit:
        best = res[[c for c in res if c.startswith("aic_")]].idxmin(axis=1).value_counts()
        print("best model by AIC:", best.to_dict())

if __name__ == "__main__":
    main()
