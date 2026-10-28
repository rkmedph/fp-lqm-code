"""Transferability of model parameters between beams (one cell line, one laboratory, several beam qualities).
Joint fits per cell line:
  LQ independent (2B params) | LQ common curve with dose rescaling (2 + B-1) | LQ common curve (2)
  exact model with shared F_r, kappa/mu, n_c, M and lambda_d per beam (3 + 1 + B-1; M in {1,2} chosen) | exact common (4).
Superposability (condition R3) is tested model-independently by the likelihood ratio LQ rescaling vs LQ independent (df = B-1).
Robustness: second (blind) digitisation; exclusion of ambiguous points; digitisation error x2, x3; for Li 2024 (tabulated SEM)
exclusion of the 8 Gy points (SEM at the rounding floor) and extra relative error 10 % / 20 %.
Datasets are analysed per laboratory and cell line and never pooled across laboratories."""
import sys, json, csv
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.stats import chi2 as c2d
sys.path.insert(0, "/home/claude/pmb"); sys.path.insert(0, "/home/claude/rawdata/analysis/digitize"); sys.path.insert(0, "/home/claude/rawdata/analysis")
from discrete import survival_disc
from energy_data import STEENKEN, VERBAKEL
from registry import build, sigma_lnS

LN10 = np.log(10)


def best(f, starts, lb, ub):
    r = None
    for p0 in starts:
        q = least_squares(f, p0, bounds=(lb, ub))
        if r is None or q.cost < r.cost: r = q
    return r


def fit_all(D, y, s, g, K):
    N = len(D); out = {}
    f = lambda p: (p[g]*D + p[K+g]*D*D - y)/s
    r = best(f, [np.r_[np.full(K, .2), np.full(K, .03)]], -np.inf, np.inf); out["LQ independent"] = (2*r.cost, 2*K, None)
    f = lambda p: (p[0]*D + p[1]*D*D - y)/s
    r = best(f, [[.2, .03]], -np.inf, np.inf); out["LQ common"] = (2*r.cost, 2, None)
    def fr(p):
        x = np.r_[1, p[2:]][g]*D
        return (p[0]*x + p[1]*x*x - y)/s
    r = best(fr, [np.r_[.2, .03, np.ones(K-1)]], np.r_[-5, -1, np.full(K-1, .3)], np.r_[5, 1, np.full(K-1, 3)])
    J = r.jac; se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)))
    out["LQ rescaling"] = (2*r.cost, K + 1, dict(factors=np.r_[1, r.x[2:]].tolist(), se=np.r_[0, se[2:]].tolist()))
    bl = bc = None
    for M in (1, 2):
        for nc in range(2, 21):
            def fx(p):
                lam = np.exp(p[:K]); F = .5 + np.exp(p[K]); k = np.exp(p[K+1]); pred = np.empty(N)
                for j in range(K):
                    m = g == j; pred[m] = -np.log(np.maximum(survival_disc(D[m], lam[j], 1, 2*F, k, nc, M), 1e-300))
                return (pred - y)/s
            st = [np.r_[np.full(K, np.log(x/M)), np.log(F0), np.log(k0)] for x, F0, k0 in [(.5, .5, .3), (2, 3, 1)]]
            r = best(fx, st, np.r_[np.full(K, np.log(1e-3)), np.log(1e-6), np.log(1e-5)], np.r_[np.full(K, np.log(200)), np.log(1e4), np.log(1e4)])
            if bl is None or r.cost < bl[0].cost: bl = (r, nc, M)
            def fc(p):
                return (-np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1, 2*(.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y)/s
            r = best(fc, [[np.log(.5/M), np.log(.5), np.log(.3)], [np.log(2/M), np.log(3), np.log(1)]], np.log([1e-3, 1e-6, 1e-5]), np.log([200, 1e4, 1e4]))
            if bc is None or r.cost < bc[0].cost: bc = (r, nc, M)
    r, nc, M = bl; lam = np.exp(r.x[:K])
    out["exact lambda_d per beam"] = (2*r.cost, K + 3, dict(lambda_ratio=(lam/lam[0]).tolist(), nc=nc, M=M, F_r=float(.5 + np.exp(r.x[K])), kappa_mu=float(np.exp(r.x[K+1]))))
    out["exact common"] = (2*bc[0].cost, 4, dict(nc=bc[1], M=bc[2]))
    res = {}
    for k, (c2, kk, extra) in out.items():
        dof = N - kk; res[k] = dict(chi2=float(c2), k=kk, dof=dof, p=float(c2d.sf(c2, dof)) if dof > 0 else None, AIC=float(c2 + 2*kk), extra=extra)
    lr = res["LQ rescaling"]["chi2"] - res["LQ independent"]["chi2"]
    res["superposition_test"] = dict(LR=float(lr), df=K - 1, p=float(c2d.sf(max(lr, 0), K - 1)))
    res["N"] = N
    return res


# ---------------- datasets and variants ----------------
def auto_values():
    S = json.load(open("/home/claude/rawdata/analysis/digitize/steenken_raw.json"))
    st = {(c["cell"], c["energy"], c["mode"]): [np.log10(p["SF"]) for p in c["points"]] for c in S}
    V = json.load(open("/home/claude/rawdata/analysis/digitize/verbakel_curves.json"))
    ve = {(c["cell"], c["beam"]): [np.log10(p["SF"]) for p in c["points"]] for c in V["curves"]}
    hidden = {(c["cell"], c["beam"]): [("hidden" in p["how"]) for p in c["points"]] for c in V["curves"]}
    return st, ve, hidden


def digitised_job(lab, cell, variant):
    st_auto, ve_auto, hidden = auto_values()
    if lab == "Steenken":
        keys = [(cell, e, m) for e in ("6 MV", "10 MV") for m in ("FLAT", "FFF")]; man = STEENKEN; auto = st_auto; labels = [f"{k[1]} {k[2]}" for k in keys]
    else:
        keys = [(cell, "6 MV FF"), (cell, "10 MV FFF")]; man = VERBAKEL; auto = ve_auto; labels = [k[1] for k in keys]
    dig = {"main": 0.03, "auto": 0.03, "exclude": 0.03, "dig x2": 0.06, "dig x3": 0.09}[variant]
    D, y, s, g = [], [], [], []
    for j, k in enumerate(keys):
        for i, (d, l, b) in enumerate(man[k]):
            la = auto[k][i]
            if variant == "exclude":
                amb = abs(la - l) > 0.1 or (lab == "Verbakel" and hidden[k][i])
                if amb: continue
            val = la if variant == "auto" else l
            D.append(d); y.append(-LN10*val); s.append(LN10*np.hypot(max(b, 0.05), dig)); g.append(j)
    return np.array(D), np.array(y), np.array(s), np.array(g), len(keys), labels


def li_job(cell, variant):
    BEAMS = ["6 MV", "192Ir", "225 kVp", "50 kVp"]
    cs = {c["beam"]: c for c in build() if c["src"] == "LI24" and c["cell"] == cell}
    floor = {"main": 0.0, "exclude": 0.0, "err+10%": 0.10, "err+20%": 0.20}[variant]
    D, y, s, g = [], [], [], []
    for j, b in enumerate(BEAMS):
        c = cs[b]; sig = sigma_lnS(c)
        for d, S, e in zip(c["D"], c["S"], sig):
            if variant == "exclude" and d == 8: continue
            D.append(d); y.append(-np.log(S)); s.append(np.hypot(e, floor)); g.append(j)
    return np.array(D), np.array(y), np.array(s), np.array(g), 4, BEAMS


def work(job):
    lab, cell, variant = job
    if lab == "Li": D, y, s, g, K, labels = li_job(cell, variant)
    else: D, y, s, g, K, labels = digitised_job(lab, cell, variant)
    r = fit_all(D, y, s, g, K); r["beams"] = labels
    return (lab, cell, variant), r


if __name__ == "__main__":
    jobs = [("Steenken", c, v) for c in ("U251", "U251-MGMT", "U87") for v in ("main", "auto", "exclude", "dig x2", "dig x3")]
    jobs += [("Verbakel", c, v) for c in ("SW1573", "D384", "T98") for v in ("main", "auto", "exclude", "dig x2", "dig x3")]
    jobs += [("Li", c, v) for c in ("HCT116", "HeLa", "PC3") for v in ("main", "exclude", "err+10%", "err+20%")]
    with Pool() as p:
        res = dict(p.map(work, jobs))
    out = {f"{a}|{b}|{c}": v for (a, b, c), v in res.items()}
    json.dump(out, open("transfer_results.json", "w"), indent=1, default=float)
    with open("table_transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["laboratory", "cell", "variant", "N", "beams", "p_superposition(LQ rescaling vs independent)", "dAIC exact-lambda minus LQ independent",
                    "dAIC exact-lambda minus LQ rescaling", "dAIC LQ rescaling minus LQ independent", "p_GOF exact-lambda", "p_GOF LQ independent",
                    "dose factors (LQ)", "lambda_d ratios (exact)", "nc, M"])
        for (lab, cell, var), r in res.items():
            e = r["exact lambda_d per beam"]; L = r["LQ independent"]; R = r["LQ rescaling"]
            w.writerow([lab, cell, var, r["N"], "; ".join(r["beams"]), f"{r['superposition_test']['p']:.3g}", f"{e['AIC'] - L['AIC']:.2f}", f"{e['AIC'] - R['AIC']:.2f}",
                        f"{R['AIC'] - L['AIC']:.2f}", "" if e["p"] is None else f"{e['p']:.3g}", "" if L["p"] is None else f"{L['p']:.3g}",
                        " ".join(f"{x:.3f}" for x in R["extra"]["factors"]), " ".join(f"{x:.3f}" for x in e["extra"]["lambda_ratio"]), f"{e['extra']['nc']}, {e['extra']['M']}"])
            print(f"{lab:8s} {cell:9s} {var:8s} N={r['N']:2d} p_sup={r['superposition_test']['p']:.3g} dAIC(ex-indep)={e['AIC']-L['AIC']:6.2f} dAIC(ex-resc)={e['AIC']-R['AIC']:6.2f} "
                  f"pGOF ex={e['p'] if e['p'] is None else round(e['p'],3)} factors={np.round(R['extra']['factors'],3)} lam={np.round(e['extra']['lambda_ratio'],3)} nc,M={e['extra']['nc']},{e['extra']['M']}", flush=True)
