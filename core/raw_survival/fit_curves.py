import numpy as np, json, sys
from scipy.optimize import least_squares
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc
from curves import CURVES

def lq_w(D, y, w):
    X = np.vstack([D, D*D]).T; W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X); c = cov @ X.T @ W @ y
    r = y - X @ c; chi2 = float(r @ W @ r); dof = len(D) - 2
    if dof > 0: cov = cov * max(chi2/dof, 1.0)       # inflate if underdispersed? keep conservative
    return c, cov, chi2

def model_fit(D, y, w, M):
    best = None
    for nc in range(2, 26):
        for p0 in ([np.log(0.5/M), 0.0, np.log(0.3)], [np.log(2.0/M), np.log(3), np.log(1.0)], [np.log(1.0/M), np.log(0.3), np.log(0.01)]):
            f = lambda p: (-np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1.0, 2*(0.5+np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y)*np.sqrt(w)
            r = least_squares(f, p0, bounds=(np.log([1e-3, 1e-3, 1e-4]), np.log([100, 100, 100])))
            chi2 = float(r.fun @ r.fun)
            if best is None or chi2 < best[0]:
                lam, Fm, k = np.exp(r.x); best = (chi2, nc, lam, 0.5+Fm, k)
    return best

out = []
for c in CURVES:
    D, S, sS = c["D"], c["S"], c["sS"]; y = -np.log(S); sy = sS/S; w = 1/sy**2
    (a, b), cov, chi2 = lq_w(D, y, w)
    rec = dict(cell=c["cell"], beam=c["beam"], n=len(D), alpha=a, beta=b, se_a=np.sqrt(cov[0,0]), se_b=np.sqrt(cov[1,1]),
               z_beta=b/np.sqrt(cov[1,1]), chi2_lq=chi2, source=c["source"])
    # -lnS/D monotonic?
    rec["lnS_over_D_nondecreasing"] = bool(np.all(np.diff(y/D) >= -2*np.sqrt((sy/D)[1:]**2+(sy/D)[:-1]**2)))
    if len(D) >= 6:
        for M in (1, 4, 10):
            chi2m, nc, lam, F, k = model_fit(D, y, w, M)
            rec[f"chi2_M{M}"] = chi2m; rec[f"par_M{M}"] = dict(nc=nc, lam_d=lam, F_r=F, kappa_mu=k)
        n = len(D)
        rec["AIC_lq"] = chi2 + 2*2
        for M in (1, 4, 10): rec[f"AIC_M{M}"] = rec[f"chi2_M{M}"] + 2*4
    out.append(rec)
    print(f"{c['cell']:15s} {c['beam']:14s} a={a:.3f}±{rec['se_a']:.3f} b={b:.4f}±{rec['se_b']:.4f} z_b={rec['z_beta']:5.1f} chi2_LQ={chi2:6.2f} (dof {len(D)-2})"
          + ("" if len(D) < 6 else "  chi2 model M1/M4/M10 = %.2f/%.2f/%.2f" % (rec['chi2_M1'], rec['chi2_M4'], rec['chi2_M10'])))
json.dump(out, open("fits.json", "w"), indent=1, default=float)
