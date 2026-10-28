"""Within-laboratory test of beam energy / FF-FFF effects (Verbakel 2013: TrueBeam; Steenken 2015: Elekta Versa HD).
Per cell line, all beams together; ln SF ~ Normal(model, s^2) with s common to a laboratory,
estimated from the residual variance of independent LQ fits (one LQ per beam).
Models: LQ common; LQ dose-rescaling (one RBE factor per non-reference beam); LQ independent;
exact model common (M = 1, 2, 4, n_c scanned); exact model with lambda_d per beam (= dose rescaling).
In the exact model the only way a photon spectrum can enter is lambda_d (lesion yield per Gy), so
'common' vs 'lambda_d per beam' is the model's energy test; LQ independent is the unconstrained benchmark."""
import sys, json, csv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as c2d
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc


def datasets():
    V = json.load(open("digitize/verbakel_curves.json"))
    ver = {}
    for c in V["curves"]: ver.setdefault(c["cell"], []).append((c["beam"], np.array([p["D"] for p in c["points"]]), np.array([p["SF"] for p in c["points"]])))
    S = json.load(open("digitize/steenken_raw.json"))
    st = {}
    for c in S: st.setdefault(c["cell"], []).append((f"{c['energy']} {c['mode']}", np.array([p["D"] for p in c["points"]]), np.array([p["SF"] for p in c["points"]])))
    for v in st.values(): v.sort(key=lambda t: (t[0] != "6 MV FLAT", t[0]))   # reference = 6 MV FLAT
    for v in ver.values(): v.sort(key=lambda t: t[0] != "6 MV FF")             # reference = 6 MV FF
    return {"Verbakel 2013 (TrueBeam; 6 MV FF vs 10 MV FFF)": ver, "Steenken 2015 (6/10 MV, FLAT vs FFF, 5 Gy/min)": st}


def lq_indep(curves):
    rss = 0; n = 0; k = 0
    for _, D, S in curves:
        X = np.vstack([D, D*D]).T; y = -np.log(S); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
        rss += r @ r; n += len(D); k += 2
    return rss, n, k


def fit_lq_resc(curves, s, rescale):
    B = len(curves); D = np.concatenate([c[1] for c in curves]); y = np.concatenate([-np.log(c[2]) for c in curves]); g = np.concatenate([[i]*len(c[1]) for i, c in enumerate(curves)])
    def f(p):
        r = np.r_[1.0, p[2:]][g] if rescale else np.ones_like(D); Dd = r*D
        return (p[0]*Dd + p[1]*Dd*Dd - y)/s
    p0 = [0.2, 0.03] + ([1.0]*(B - 1) if rescale else [])
    res = least_squares(f, p0)
    J = res.jac; cov = np.linalg.inv(J.T @ J)
    return 2*res.cost, len(p0), res.x, np.sqrt(np.diag(cov))


def fit_exact(curves, s, rescale):
    B = len(curves); D = np.concatenate([c[1] for c in curves]); y = np.concatenate([-np.log(c[2]) for c in curves]); g = np.concatenate([[i]*len(c[1]) for i, c in enumerate(curves)])
    best = None
    for M in (1, 2, 4):
        for nc in range(2, 26):
            def f(p):
                lam = np.exp(p[0])*(np.r_[1.0, np.exp(p[3:])][g] if rescale else 1.0)
                pred = np.empty_like(y)
                for i in range(B):
                    m = g == i; l = lam[m][0] if rescale else lam
                    pred[m] = -np.log(np.maximum(survival_disc(D[m], l, 1, 2*(0.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300))
                return (pred - y)/s
            for p0 in ([np.log(0.5/M), np.log(0.5), np.log(0.3)], [np.log(2/M), np.log(3), np.log(1)], [np.log(1/M), np.log(.05), np.log(.01)]):
                q = p0 + ([0.0]*(B - 1) if rescale else [])
                lb = [np.log(1e-3), np.log(1e-6), np.log(1e-5)] + [-1.0]*(B - 1 if rescale else 0)
                ub = [np.log(200), np.log(1e4), np.log(1e4)] + [1.0]*(B - 1 if rescale else 0)
                r = least_squares(f, q, bounds=(lb, ub))
                if best is None or r.cost < best[0].cost: best = (r, M, nc)
    r, M, nc = best
    return 2*r.cost, 3 + 1 + (B - 1 if rescale else 0), dict(M=M, nc=nc, lam=float(np.exp(r.x[0])), F_r=float(0.5 + np.exp(r.x[1])), kappa_mu=float(np.exp(r.x[2])),
                                                             rbe=[1.0] + (np.exp(r.x[3:]).tolist() if rescale else []))


def main():
    out = {}
    for lab, cells in datasets().items():
        rss = n = k = 0
        for cv in cells.values():
            a, b, c = lq_indep(cv); rss += a; n += b; k += c
        s = float(np.sqrt(rss/(n - k)))
        out[lab] = dict(sigma_lnSF=s, n_points=n, cells={})
        print(f"== {lab}: s(ln SF) = {s:.3f} from independent LQ ({n} points, {k} parameters)")
        for cell, cv in cells.items():
            beams = [c[0] for c in cv]; npts = sum(len(c[1]) for c in cv)
            R = {}
            rss_i, _, k_i = lq_indep(cv); R["LQ independent"] = (rss_i/s**2, k_i, None)
            c2, kk, x, se = fit_lq_resc(cv, s, False); R["LQ common"] = (c2, kk, dict(alpha=x[0], beta=x[1]))
            c2, kk, x, se = fit_lq_resc(cv, s, True); R["LQ dose-rescaling"] = (c2, kk, dict(alpha=x[0], beta=x[1], rbe=[1.0] + x[2:].tolist(), rbe_se=[0.0] + se[2:].tolist()))
            c2, kk, par = fit_exact(cv, s, False); R["exact common"] = (c2, kk, par)
            c2, kk, par = fit_exact(cv, s, True); R["exact lambda_d per beam"] = (c2, kk, par)
            rec = {}
            for m, (c2, kk, par) in R.items():
                rec[m] = dict(chi2=float(c2), k=kk, dof=npts - kk, AIC=float(c2 + 2*kk), par=par)
            lr = rec["LQ common"]["chi2"] - rec["LQ dose-rescaling"]["chi2"]; rec["LR_beam_effect_LQ"] = dict(stat=lr, df=len(beams) - 1, p=float(c2d.sf(lr, len(beams) - 1)))
            lr = rec["exact common"]["chi2"] - rec["exact lambda_d per beam"]["chi2"]; rec["LR_beam_effect_exact"] = dict(stat=lr, df=len(beams) - 1, p=float(c2d.sf(max(lr, 0), len(beams) - 1)))
            rec["beams"] = beams
            out[lab]["cells"][cell] = rec
            print(f"  {cell}: beams {beams}")
            for m in R: print(f"     {m:26s} chi2={rec[m]['chi2']:6.2f} k={rec[m]['k']} AIC={rec[m]['AIC']:6.2f}")
            print("     RBE (LQ rescaling):", [f"{b}: {r:.3f}±{e:.3f}" for b, r, e in zip(beams, rec['LQ dose-rescaling']['par']['rbe'], rec['LQ dose-rescaling']['par']['rbe_se'])])
            print("     RBE (exact, lambda_d ratio):", [f"{b}: {r:.3f}" for b, r in zip(beams, rec['exact lambda_d per beam']['par']['rbe'])],
                  "| LR beam effect: LQ p=%.3g, exact p=%.3g" % (rec["LR_beam_effect_LQ"]["p"], rec["LR_beam_effect_exact"]["p"]))
    json.dump(out, open("mv_energy_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
