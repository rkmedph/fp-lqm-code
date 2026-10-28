"""Least-favourable-configuration (LFC) calibration of the monotonicity test, and its size on explicit convex curves.

H0: G(D) = -ln S(D)/D non-decreasing. Observations G_i = G(D_i) + (l + e_i)/D_i (sign absorbed), e_i ~ N(0, s_i^2)
independent, l ~ N(0, sc^2) common to all doses (normalisation to the control). In the G-space the covariance is
Sigma = diag(s_i^2/D_i^2) + sc^2 v v^T with v_i = 1/D_i; in the -ln S space it is diag(s_i^2) + sc^2 1 1^T.
T = min over non-decreasing f and l of sum w_i (G_i - f_i - l v_i)^2 + (l/sc)^2, w_i = (D_i/s_i)^2, which by
the Woodbury identity is the squared Sigma^{-1}-distance from G to the cone C of non-decreasing vectors.
For any mean mu in C, C - mu contains C, so dist(mu + eps, C) <= dist(eps, C): T is stochastically largest when G is
constant (mu in the lineality space of C), i.e. at -ln S linear in D (g'' = 0). Simulating T at a constant G gives a
critical value valid over the whole of H0 (size <= alpha) under the stated Gaussian error model.

The earlier p-values came from a parametric bootstrap at the fitted constrained curve f_hat, which is in general
not the LFC; here both are compared on explicit convex curves including the boundary."""
import json, sys, zlib, numpy as np
from multiprocessing import Pool
from isotonic_test import stat, test as boot_fixed
from isotonic_norm import T_of, test as boot_norm

R = json.load(open("polgar_results.json"))
KS = [1/np.sqrt(3), 1.0, 1.5, 2.0]; SC = [0.0, 0.03, 0.05, 0.10, np.inf]
BNULL = 2000

def design(r):
    D = np.array(r["D"]); o = np.argsort(D)
    return D[o], np.array(r["lnS"])[o], np.array(r["sig"])[o]

def T(D, y, s, sc):
    return stat(-y/D, (D/s)**2)[0] if sc == 0 else T_of(D, y, s, sc)[0]

def null(D, s, sc, B, rng):
    """T at a constant G (LFC); T is invariant to the value of the constant."""
    return np.sort([T(D, (rng.normal(0, sc) if 0 < sc < np.inf else 0.0) + rng.normal(0, s), s, sc) for _ in range(B)])

def pval(t, nl):
    return (1 + len(nl) - np.searchsorted(nl, t, side="left"))/(len(nl) + 1)

def job_null(args):
    i, k, sc = args; D, _, s = design(R[i]); rng = np.random.default_rng(100000*i + int(1000*k) + int(10*min(sc, 99)))
    return (i, k, sc), null(D, k*s, sc, BNULL, rng)

def bh(ps, q=0.05):
    o = np.argsort(ps); m = len(ps); t = 0
    for j, i in enumerate(o, 1):
        if ps[i] <= q*j/m: t = j
    return t

# ---------- size on explicit convex curves ----------
def curve(name, D, rng):
    a = rng.uniform(0.05, 0.4)
    if name == "linear (g''=0)": return a*D
    if name == "LQ, beta = 0.005": return a*D + 0.005*D*D
    if name == "LQ, beta ~ U(0, 0.06)": return a*D + rng.uniform(0, 0.06)*D*D
    if name == "linear to 1 Gy, then convex": b = rng.uniform(0.02, 0.06); return a*D + b*np.clip(D - 1, 0, None)**2
SCEN = ["linear (g''=0)", "LQ, beta = 0.005", "LQ, beta ~ U(0, 0.06)", "linear to 1 Gy, then convex"]
REPS = 4

def job_size(args):
    scen, sc, nulls = args; rng = np.random.default_rng(zlib.crc32(f"{scen}|{sc}".encode()))
    rb = rl = n = 0
    for rep in range(REPS):
        for i, r in enumerate(R):
            D, _, s = design(r)
            y = -curve(scen, D, rng) + (rng.normal(0, sc) if 0 < sc < np.inf else (rng.normal(0, 0.05) if sc == np.inf else 0)) + rng.normal(0, s)
            pb = boot_fixed(D, y, s, B=200, rng=rng)[1] if sc == 0 else boot_norm(D, y, s, sc, 200, rng)[2]
            pl = pval(T(D, y, s, sc), nulls[i])
            rb += pb < 0.05; rl += pl < 0.05; n += 1
    return (scen, sc), dict(n=n, boot=rb/n, lfc=rl/n)

if __name__ == "__main__":
    jobs = [(i, k, sc) for i in range(len(R)) for k in KS for sc in SC if not (sc in (0.0, np.inf) and k != 1.0)]
    with Pool(2) as p: NL = dict(p.map(job_null, jobs, chunksize=8))
    def nl(i, k, sc):   # sigma_c = 0 or free: T scales as 1/k^2 and its null distribution is invariant
        return NL[(i, k, sc)] if (i, k, sc) in NL else NL[(i, 1.0, sc)]*1.0
    # real data: LFC p-values over the 20 error models
    grid = {}
    for k in KS:
        for sc in SC:
            ps = []
            for i, r in enumerate(R):
                D, y, s = design(r)
                if sc in (0.0, np.inf): ps.append(pval(T(D, y, s, sc)/k**2, nl(i, 1.0, sc)))
                else: ps.append(pval(T(D, y, k*s, sc), nl(i, k, sc)))
            ps = np.array(ps); grid[f"{k:.3f}|{sc}"] = dict(p05=int((ps < .05).sum()), p01=int((ps < .01).sum()), BH=bh(ps), p=ps.tolist())
    print("LFC-calibrated, p<0.05 (BH):")
    for k in KS: print(f"{k:5.2f}  " + "  ".join(f"{grid[f'{k:.3f}|{sc}']['p05']:>3} ({grid[f'{k:.3f}|{sc}']['BH']:>2})" for sc in SC), flush=True)
    # size
    sjobs = [(scen, sc, [NL[(i, 1.0, sc)] for i in range(len(R))]) for scen in SCEN for sc in (0.0, 0.03, 0.05, np.inf)]
    with Pool(2) as p: SZ = dict(p.map(job_size, sjobs))
    print("\nrejection rate at nominal 5 % (true errors = plotted bars, offset model as assumed by the test):")
    for scen in SCEN:
        print(f"{scen:30s} " + "  ".join(f"sc={sc}: boot {SZ[(scen, sc)]['boot']:.3f} LFC {SZ[(scen, sc)]['lfc']:.3f}" for sc in (0.0, 0.03, 0.05, np.inf)), flush=True)
    json.dump(dict(grid=grid, size={f"{a}|{b}": v for (a, b), v in SZ.items()}, n_null=BNULL, reps=REPS),
              open("isotonic_lfc_results.json", "w"), indent=1)
