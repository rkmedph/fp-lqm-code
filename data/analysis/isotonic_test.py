"""Falsification criterion for the homogeneous model class (theorem: -ln S convex => G(D) = -ln S/D non-decreasing).
H0: G non-decreasing. Statistic T = min over non-decreasing G of sum w_i (G_i - Gobs_i)^2 (weighted isotonic regression, PAVA).
p-value by parametric bootstrap under the H0 fit (Gaussian errors on ln S). Applied to the 91 photon datasets of Polgár 2022."""
import json, numpy as np

def pava(y, w):
    y = list(map(float, y)); w = list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i+1][0] + 1e-15:
            a, b = blocks[i], blocks[i+1]; W = a[1] + b[1]
            blocks[i] = [(a[0]*a[1] + b[0]*b[1])/W, W, a[2] + b[2]]; del blocks[i+1]
            i = max(i - 1, 0)
        else: i += 1
    return np.concatenate([[v]*n for v, _, n in blocks])

def stat(G, w):
    f = pava(G, w); return float(np.sum(w*(G - f)**2)), f

def test(D, lnS, sig, B=4000, rng=np.random.default_rng(0)):
    o = np.argsort(D); D, lnS, sig = D[o], lnS[o], sig[o]
    G = -lnS/D; w = (D/sig)**2
    T, f = stat(G, w)
    sims = np.array([stat(f + rng.normal(0, sig/D), w)[0] for _ in range(B)])
    return T, float((1 + np.sum(sims >= T))/(B + 1))

if __name__ == "__main__":
    R = json.load(open("polgar_results.json")); out = []
    for r in R:
        D = np.array(r["D"]); lnS = np.array(r["lnS"]); s = np.array(r["sig"])
        T, p = test(D, lnS, s)
        out.append(dict(id=r["id"], cell=r["cell"], radiation=r["radiation"], n=len(D), T=T, p=p, slope_z=r["fits"].get("slope_test", {}).get("z")))
    p = np.array([o["p"] for o in out])
    from statistics import median
    print(f"{len(out)} photon datasets: p<0.05: {(p<0.05).sum()}, p<0.01: {(p<0.01).sum()}, p<0.001: {(p<0.001).sum()}")
    # Benjamini-Hochberg at 5 %
    o = np.argsort(p); m = len(p); thr = 0
    for k, i in enumerate(o, 1):
        if p[i] <= 0.05*k/m: thr = k
    print("rejected after Benjamini-Hochberg (FDR 5 %):", thr)
    for sc in (1.5, 2.0):
        pp = [test(np.array(r["D"]), np.array(r["lnS"]), sc*np.array(r["sig"]), B=2000)[1] for r in R]
        pp = np.array(pp); print(f"errors x{sc}: p<0.05 {(pp<0.05).sum()}, p<0.01 {(pp<0.01).sum()}")
        for x, v in zip(out, pp): x[f"p_x{sc}"] = float(v)
    json.dump(out, open("isotonic_results.json", "w"), indent=1)
