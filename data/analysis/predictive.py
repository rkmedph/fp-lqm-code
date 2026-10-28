"""Predictive comparison of LQ and the exact model (M in {1,2,4} chosen in-sample, n_c in 2..30).
(a) leave-one-dose-out (LOO) on single curves with >= 6 points and on pooled multi-condition sets
    (pooled: LQ alpha, beta per condition vs exact model with shared F_r, kappa/mu, n_c, M and lambda_d per condition);
(b) extrapolation: fit low doses -> predict high doses, and fit high doses -> predict low doses;
(c) error-scale sensitivity (x1, x1.5, x2) for in-sample criteria (chi2, AIC, BIC).
Scores: standardized prediction error z = (y_obs - y_pred)/sigma on y = -ln S; mean z^2; mean |delta ln S|."""
import sys, json
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis/digitize"); sys.path.insert(0, "/home/claude/rawdata/analysis")
from discrete import survival_disc
from registry import build, sigma_lnS
from energy_data import STEENKEN, VERBAKEL, DIG

LN10 = np.log(10)
NC = range(2, 31)


def g(D, lam, F, k, nc, M):
    return -np.log(np.maximum(survival_disc(D, lam, 1.0, 2*F, k, nc, M), 1e-300))


def best(f, starts, lb, ub):
    r = None
    for p0 in starts:
        q = least_squares(f, p0, bounds=(lb, ub))
        if r is None or q.cost < r.cost: r = q
    return r


# ---------- grouped data: D, y, s, group index ----------
def fit_lq(D, y, s, gidx, K):
    X = np.zeros((len(D), 2*K))
    for j in range(K):
        m = gidx == j; X[m, j] = D[m]; X[m, K + j] = D[m]**2
    Xw = X/s[:, None]; c, *_ = np.linalg.lstsq(Xw, y/s, rcond=None); r = y/s - Xw @ c
    return dict(par=c, chi2=float(r @ r), k=2*K)


def pred_lq(fit, D, gidx, K):
    c = fit["par"]; return c[gidx]*D + c[K + gidx]*D*D


def fit_exact(D, y, s, gidx, K, Ms=(1, 2, 4)):
    bestfit = None
    for M in Ms:
        for nc in NC:
            def f(p):
                lam = np.exp(p[:K]); F = .5 + np.exp(p[K]); k = np.exp(p[K + 1]); out = np.empty(len(D))
                for j in range(K):
                    m = gidx == j
                    if m.any(): out[m] = g(D[m], lam[j], F, k, nc, M)
                return (out - y)/s
            st = [np.r_[np.full(K, np.log(x/M)), np.log(F0), np.log(k0)] for x, F0, k0 in [(.5, .5, .3), (2, 3, 1), (1, .05, .01)]]
            r = best(f, st, np.r_[np.full(K, np.log(1e-3)), np.log(1e-6), np.log(1e-5)], np.r_[np.full(K, np.log(300)), np.log(1e4), np.log(1e4)])
            if bestfit is None or r.cost < bestfit[0].cost: bestfit = (r, nc, M)
    r, nc, M = bestfit
    return dict(lam=np.exp(r.x[:K]), F=.5 + np.exp(r.x[K]), k=np.exp(r.x[K + 1]), nc=nc, M=M, chi2=2*r.cost, k_par=K + 3)


def pred_exact(fit, D, gidx, K):
    out = np.empty(len(D))
    for j in range(K):
        m = gidx == j
        if m.any(): out[m] = g(D[m], fit["lam"][j], fit["F"], fit["k"], fit["nc"], fit["M"])
    return out


# ---------- datasets ----------
def datasets():
    DS = {}
    H = json.load(open("/home/claude/rawdata/analysis/digitize/howard_curves.json"))
    for c in H:
        D = np.array([p["D"] for p in c["points"]]); DS[f"Howard2018 {c['cell']} 6MV"] = (D, -np.log([p["SF"] for p in c["points"]]),
            np.array([p["sigma_lnSF"] for p in c["points"]]), np.zeros(len(D), int), 1)
    for (cell, b), pts in VERBAKEL.items():
        p = np.array(pts); DS[f"Verbakel2013 {cell} {b}"] = (p[:, 0], -LN10*p[:, 1], LN10*np.hypot(np.maximum(p[:, 2], .05), DIG), np.zeros(len(p), int), 1)
    D = np.array([0.90, 1.82, 3.22, 5.00, 6.62, 8.34, 9.91, 11.44]); S = np.array([0.77, 0.60, 0.275, 0.085, 0.0193, 0.0061, 0.00182, 0.00051])
    DS["Elkind-Sutton 1959"] = (D, -np.log(S), np.full(8, 0.10), np.zeros(8, int), 1)
    D = np.array([0.33, 0.5, 2, 4, 4.67, 6.3]); S = np.array([0.793, 0.694, .36, .10, .059, .016]); s = np.array([.104, .104, .02/.36, .01/.10, .104, .104])
    DS["Butterworth2010 AGO-1522B 6MV"] = (D, -np.log(S), s, np.zeros(6, int), 1)
    # pooled sets
    for cell in ("HCT116", "HeLa", "PC3"):
        cs = [c for c in build() if c["src"] == "LI24" and c["cell"] == cell]
        DS[f"Li2024 {cell} pooled 4 beams"] = (np.concatenate([c["D"] for c in cs]), np.concatenate([-np.log(c["S"]) for c in cs]),
            np.concatenate([sigma_lnS(c) for c in cs]), np.repeat(np.arange(4), 4), 4)
    for cell in ("U251", "U251-MGMT", "U87"):
        keys = [(cell, e, m) for e in ("6 MV", "10 MV") for m in ("FLAT", "FFF")]
        P = [np.array(STEENKEN[k]) for k in keys]
        DS[f"Steenken2015 {cell} pooled 6/10MV FLAT/FFF"] = (np.concatenate([p[:, 0] for p in P]), np.concatenate([-LN10*p[:, 1] for p in P]),
            np.concatenate([LN10*np.hypot(np.maximum(p[:, 2], .05), DIG) for p in P]), np.repeat(np.arange(4), 4), 4)
    for cell in ("SW1573", "D384", "T98"):
        P = [np.array(VERBAKEL[(cell, b)]) for b in ("6 MV FF", "10 MV FFF")]
        DS[f"Verbakel2013 {cell} pooled 6FF/10FFF"] = (np.concatenate([p[:, 0] for p in P]), np.concatenate([-LN10*p[:, 1] for p in P]),
            np.concatenate([LN10*np.hypot(np.maximum(p[:, 2], .05), DIG) for p in P]), np.repeat(np.arange(2), 6), 2)
    return DS


def job(args):
    name, (D, y, s, gi, K), task = args
    out = dict(name=name, task=task)
    if task == "full":
        L = fit_lq(D, y, s, gi, K); E = fit_exact(D, y, s, gi, K); n = len(D)
        out.update(n=n, LQ=dict(chi2=L["chi2"], k=L["k"]), EX=dict(chi2=E["chi2"], k=E["k_par"], nc=E["nc"], M=E["M"], F=E["F"], kappa=E["k"]))
        return out
    if task.startswith("loo"):
        i = int(task[3:]); m = np.ones(len(D), bool); m[i] = False
        # need each condition to keep >= 3 points for LQ
        if np.sum(gi[m] == gi[i]) < 3: return None
        L = fit_lq(D[m], y[m], s[m], gi[m], K); E = fit_exact(D[m], y[m], s[m], gi[m], K)
        yl = pred_lq(L, D[i:i+1], gi[i:i+1], K)[0]; ye = pred_exact(E, D[i:i+1], gi[i:i+1], K)[0]
        out.update(D=float(D[i]), y=float(y[i]), s=float(s[i]), yLQ=float(yl), yEX=float(ye))
        return out
    if task in ("lo2hi", "hi2lo"):
        if task == "lo2hi": cut = 6.7; fitm = D <= cut
        else: cut = 3.9; fitm = D >= cut
        if K == 1 and fitm.sum() < 5: return None
        if any(np.sum((gi == j) & fitm) < 3 for j in range(K)) or (~fitm).sum() == 0: return None
        L = fit_lq(D[fitm], y[fitm], s[fitm], gi[fitm], K); E = fit_exact(D[fitm], y[fitm], s[fitm], gi[fitm], K)
        pm = ~fitm
        out.update(D=D[pm].tolist(), y=y[pm].tolist(), s=s[pm].tolist(), yLQ=pred_lq(L, D[pm], gi[pm], K).tolist(), yEX=pred_exact(E, D[pm], gi[pm], K).tolist())
        return out


def main():
    DS = datasets(); jobs = []
    for name, d in DS.items():
        jobs.append((name, d, "full"))
        if d[4] == 1 and len(d[0]) < 6: continue
        jobs += [(name, d, f"loo{i}") for i in range(len(d[0]))] + [(name, d, "lo2hi"), (name, d, "hi2lo")]
    with Pool() as pool:
        res = [r for r in pool.map(job, jobs, chunksize=1) if r is not None]
    json.dump(res, open("predictive_results.json", "w"), indent=1, default=float)
    print(len(res), "results")


if __name__ == "__main__":
    main()
