"""Within-laboratory tests of beam quality (6 vs 10 MV, FLAT vs FFF) on manually digitised data.
Models per cell line (all conditions jointly): LQ independent; LQ common curve; LQ dose-rescaling (one factor per condition);
exact model with shared F_r, kappa/mu, n_c and lambda_d per condition (= dose rescaling inside the model); exact model common curve."""
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as c2d
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis/digitize")
from discrete import survival_disc, ab_disc
from energy_data import STEENKEN, VERBAKEL, DIG

LN10 = np.log(10)


def stack(conds):
    D, y, s, g = [], [], [], []
    for j, pts in enumerate(conds):
        for d, l, b in pts:
            D.append(d); y.append(-LN10*l); s.append(LN10*np.hypot(max(b, 0.05), DIG)); g.append(j)
    return np.array(D), np.array(y), np.array(s), np.array(g)


def best(f, starts, lb, ub):
    r = None
    for p0 in starts:
        q = least_squares(f, p0, bounds=(lb, ub))
        if r is None or q.cost < r.cost: r = q
    return r


def run(conds, labels):
    D, y, s, g = stack(conds); K = len(conds); N = len(D); out = {}
    f = lambda p: (p[g]*D + p[K+g]*D*D - y)/s
    r = best(f, [np.r_[np.full(K, .2), np.full(K, .03)]], -np.inf, np.inf); out["LQ independent"] = (2*r.cost, 2*K)
    f = lambda p: (p[0]*D + p[1]*D*D - y)/s
    r = best(f, [[.2, .03]], -np.inf, np.inf); out["LQ common curve"] = (2*r.cost, 2)
    def fr(p):
        rr = np.r_[1, p[2:]][g]; x = rr*D
        return (p[0]*x + p[1]*x*x - y)/s
    r = best(fr, [np.r_[.2, .03, np.ones(K-1)]], np.r_[-5, -1, np.full(K-1, .5)], np.r_[5, 1, np.full(K-1, 2)])
    out["LQ dose-rescaling"] = (2*r.cost, 2 + K - 1); out["_rescale_factors"] = np.r_[1, r.x[2:]].tolist()
    for M in (1, 2):
        bb = None; bc = None
        for nc in range(2, 21):
            def fx(p):
                lam = np.exp(p[:K]); F = .5 + np.exp(p[K]); k = np.exp(p[K+1]); pred = np.empty(N)
                for j in range(K):
                    m = g == j; pred[m] = -np.log(np.maximum(survival_disc(D[m], lam[j], 1, 2*F, k, nc, M), 1e-300))
                return (pred - y)/s
            st = [np.r_[np.full(K, np.log(x/M)), np.log(F0), np.log(k0)] for x, F0, k0 in [(.5, .5, .3), (2, 3, 1), (1, .05, .01)]]
            lb = np.r_[np.full(K, np.log(1e-3)), np.log(1e-6), np.log(1e-5)]; ub = np.r_[np.full(K, np.log(200)), np.log(1e4), np.log(1e4)]
            r = best(fx, st, lb, ub)
            if bb is None or r.cost < bb[0].cost: bb = (r, nc)
            def fc(p):
                return (-np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1, 2*(.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y)/s
            r = best(fc, [[np.log(.5/M), np.log(.5), np.log(.3)], [np.log(2/M), np.log(3), np.log(1)], [np.log(1/M), np.log(.05), np.log(.01)]],
                     np.log([1e-3, 1e-6, 1e-5]), np.log([200, 1e4, 1e4]))
            if bc is None or r.cost < bc[0].cost: bc = (r, nc)
        out[f"exact M={M}, lambda_d per condition"] = (2*bb[0].cost, K + 2 + 1)
        out[f"_lam_M{M}"] = (np.exp(bb[0].x[:K])/np.exp(bb[0].x[0])).tolist()
        out[f"exact M={M}, common curve"] = (2*bc[0].cost, 4)
    res = {}
    for k, v in out.items():
        if k.startswith("_"): res[k] = v; continue
        c2, kk = v; dof = N - kk
        res[k] = dict(chi2=c2, k=kk, dof=dof, p=float(c2d.sf(c2, dof)) if dof > 0 else None, AIC=c2 + 2*kk)
    res["_conditions"] = labels; res["_N"] = N
    return res


def main():
    allr = {}
    for cell in ("U251", "U251-MGMT", "U87"):
        keys = [(cell, e, m) for e in ("6 MV", "10 MV") for m in ("FLAT", "FFF")]
        allr[f"Steenken {cell} (6/10 MV x FLAT/FFF)"] = run([STEENKEN[k] for k in keys], [f"{k[1]} {k[2]}" for k in keys])
        keys = [(cell, e, "FLAT") for e in ("6 MV", "10 MV")]
        allr[f"Steenken {cell} (6 vs 10 MV, FLAT)"] = run([STEENKEN[k] for k in keys], [f"{k[1]} {k[2]}" for k in keys])
    for cell in ("SW1573", "D384", "T98"):
        keys = [(cell, "6 MV FF"), (cell, "10 MV FFF")]
        allr[f"Verbakel {cell} (6 MV FF vs 10 MV FFF)"] = run([VERBAKEL[k] for k in keys], [k[1] for k in keys])
    json.dump(allr, open("energy_results.json", "w"), indent=1, default=float)
    with open("table_energy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "model", "n_params", "chi2", "dof", "p", "AIC"])
        for ds, r in allr.items():
            print("==", ds, "N =", r["_N"], "rescale(LQ):", np.round(r["_rescale_factors"], 3), "lambda ratio M1:", np.round(r["_lam_M1"], 3))
            for k, v in r.items():
                if k.startswith("_"): continue
                print(f"   {k:38s} chi2={v['chi2']:7.2f} k={v['k']:2d} dof={v['dof']:2d} p={'' if v['p'] is None else f'{v[chr(112)]:.2g}'} AIC={v['AIC']:.2f}")
                w.writerow([ds, k, v["k"], f"{v['chi2']:.2f}", v["dof"], "" if v["p"] is None else f"{v['p']:.3g}", f"{v['AIC']:.2f}"])


if __name__ == "__main__":
    main()
