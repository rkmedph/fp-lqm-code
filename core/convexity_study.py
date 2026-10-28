"""Convexity of -ln S for the homogeneous threshold model, sign of range-fitted beta,
range-fitted alpha^2/(2 beta), and a heterogeneous counterexample."""
import numpy as np, json
from scipy.stats import poisson
from discrete import h_disc

def neglogS(D, lam, mu, s2, kap, nc, M=1):
    h = h_disc(nc, mu, s2, kap); nc = int(nc); out = []
    for d in np.atleast_1d(D):
        x = lam*d; k = np.arange(nc)
        p = np.sum(poisson.pmf(k, x)*h[:nc]) + poisson.sf(nc-1, x)
        out.append(-M*np.log1p(-min(p, 1-1e-300)))
    return np.array(out)

def draw(rng):
    return dict(lam=10**rng.uniform(-1.5,1), F=10**rng.uniform(np.log10(0.51),1.5),
                k=10**rng.uniform(-3,1.3), nc=int(rng.integers(2,40)), M=int(rng.choice([1,2,4,10,50])))

def lqfit(D, y):
    X = np.vstack([D, D**2]).T; return np.linalg.lstsq(X, y, rcond=None)[0]

if __name__ == "__main__":
    rng = np.random.default_rng(2026); Dg = np.linspace(0.05, 15, 300)
    n_conv = n_bad = 0
    for _ in range(3000):
        q = draw(rng); y = neglogS(Dg, q["lam"], 1.0, 2*q["F"], q["k"], q["nc"], q["M"])
        ok = (y > 1e-8) & (y < 27.6)            # 1e-12 < S < 1 - 1e-8
        if ok.sum() < 20: continue
        d2 = np.diff(y[ok], 2); n_conv += 1
        if d2.min() < -1e-6*np.abs(d2).max(): n_bad += 1
    print("convexity: evaluated", n_conv, "non-convex", n_bad)
    # range fits on a typical experimental design
    res = {"convexity": {"n": n_conv, "nonconvex": n_bad}}; fits = {}
    for M in [1, 2, 4, 10]:
        rng2 = np.random.default_rng(M); A = []; B = []
        tried = 0
        while len(A) < 1500 and tried < 20000:
            tried += 1; q = draw(rng2); Dmax = rng2.choice([6., 8., 10.])
            D = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10.]); D = D[D <= Dmax]
            y = neglogS(D, q["lam"], 1.0, 2*q["F"], q["k"], q["nc"], M); ok = y < np.log(1e4)
            if ok.sum() < 5 or y[ok][-1] < 0.7: continue      # at least ~50% kill at the top usable dose
            a, b = lqfit(D[ok], y[ok]); A.append(a); B.append(b)
        A = np.array(A); B = np.array(B); R = np.where(B > 0, A**2/(2*B), np.inf)
        fits[M] = {"alpha": A.tolist(), "beta": B.tolist()}
        res[f"M{M}"] = dict(n=len(A), frac_beta_le0=float((B <= 0).mean()), frac_R_gt_M=float((R > M).mean()),
                           R_q50_q90_max=[float(np.median(R)), float(np.percentile(R, 90)), float(R[np.isfinite(R)].max())])
        print(M, res[f"M{M}"], flush=True)
    # heterogeneous counterexample (M=1): sensitive + resistant subpopulation
    D = np.array([0.5, 1, 2, 3, 4, 5, 6, 8.])
    def S_mix(f, D):
        return f*np.exp(-neglogS(D, 1.5, 1.0, 3.0, 0.2, 3)) + (1-f)*np.exp(-neglogS(D, 0.25, 1.0, 3.0, 0.2, 3))
    best = None
    for f in np.linspace(0.5, 0.99, 50):
        y = -np.log(S_mix(f, D)); a, b = lqfit(D, y)
        if best is None or b < best[2]: best = (f, a, b)
    f, a, b = best; print("mixture: sensitive fraction %.2f -> LQ fit alpha=%.3f beta=%.4f" % (f, a, b))
    ya = neglogS(D, 1.5, 1.0, 3.0, 0.2, 3); yb = neglogS(D, 0.25, 1.0, 3.0, 0.2, 3)
    print("components LQ fits:", lqfit(D, ya), lqfit(D, yb))
    res["mixture"] = dict(f=f, alpha=a, beta=b, comp_a=lqfit(D, ya).tolist(), comp_b=lqfit(D, yb).tolist())
    json.dump(res, open("convexity_study.json", "w"), indent=1)
    json.dump(fits, open("rangefit_ab.json", "w"))
