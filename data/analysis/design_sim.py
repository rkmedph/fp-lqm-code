"""Simulated experiments: can a new clonogenic dataset discriminate LQ from the exact model for A549?
Truths: A = exact model, best fit to the A549 6 MV curve (strong shoulder, alpha0 ~ 0.01);
        B = LQ best fit (alpha 0.189, beta 0.024);  C = exact model with alpha0 in 0.10-0.14 (intermediate).
Designs: 'low' 0.1, 0.25, 0.5, 1, 2, 4, 6, 8 Gy; 'standard' 2, 4, 6, 8 Gy; 'extended' 1, 2, 4, 5, 6, 8, 10 Gy.
Noise per dish: ln SF = ln S + N(0, cv^2 + 1/C), C = 200 expected colonies; r = 6 dishes per dose; common control
normalisation offset N(0, (cv^2 + 1/C)/r). Both models fitted (exact: M in {1,2}, n_c 2..20); dAIC = AIC_exact - AIC_LQ."""
import sys, json
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, "/home/claude/pmb")
from discrete import survival_disc

R = json.load(open("identifiability_results.json"))["Howard2018 A549 6MV"]["fits"]
cmin = min(f["chi2"] for f in R); ok = [f for f in R if f["chi2"] <= cmin + 4]
A = min(ok, key=lambda f: f["chi2"])
C = min([f for f in ok if 0.10 <= f["alpha0"] <= 0.14], key=lambda f: f["chi2"])
def gex(D, f): return -np.log(survival_disc(D, f["lam"], 1, 2*f["F"], f["k"], f["nc"], f["M"]))
TRUTH = {"A exact shoulder": lambda D: gex(D, A), "B LQ": lambda D: 0.189*D + 0.0237*D*D, "C exact intermediate": lambda D: gex(D, C)}
DESIGN = {"low (0.1-8 Gy, 8 doses)": np.array([0.1, 0.25, 0.5, 1, 2, 4, 6, 8.]), "standard (2-8 Gy, 4 doses)": np.array([2, 4, 6, 8.]),
          "extended (1-10 Gy, 7 doses)": np.array([1, 2, 4, 5, 6, 8, 10.])}
r, COL = 6, 200

def fit_both(D, y, s):
    X = np.vstack([D, D*D]).T/s[:, None]; c, *_ = np.linalg.lstsq(X, y/s, rcond=None); chiLQ = float(np.sum((y/s - X @ c)**2))
    best = np.inf
    for M in (1, 2):
        for nc in range(2, 21):
            f = lambda p: (-np.log(np.maximum(survival_disc(D, np.exp(p[0]), 1, 2*(.5 + np.exp(p[1])), np.exp(p[2]), nc, M), 1e-300)) - y)/s
            for p0 in ([np.log(.5/M), np.log(.5), np.log(.3)], [np.log(2/M), np.log(3), np.log(1)], [np.log(1/M), np.log(.01), np.log(.01)]):
                q = least_squares(f, p0, bounds=(np.log([1e-3, 1e-6, 1e-5]), np.log([300, 1e4, 1e4])))
                best = min(best, 2*q.cost)
    return chiLQ, best, c

def sim(args):
    tname, dname, cv, seed = args
    rng = np.random.default_rng(seed); D = DESIGN[dname]; mu = TRUTH[tname](D)
    sd_dish = np.sqrt(cv**2 + 1/COL); s = np.full(len(D), sd_dish/np.sqrt(r))
    y = mu + rng.normal(0, s) + rng.normal(0, sd_dish/np.sqrt(r))
    chiLQ, chiEX, c = fit_both(D, y, s)
    return dict(truth=tname, design=dname, cv=cv, dAIC=(chiEX + 8) - (chiLQ + 4), chiLQ=chiLQ, chiEX=chiEX, alphaLQ=c[0])

if __name__ == "__main__":
    print("truth A:", {k: round(float(A[k]), 4) for k in ("M", "nc", "lam", "F", "k", "alpha0", "beta0")})
    print("truth C:", {k: round(float(C[k]), 4) for k in ("M", "nc", "lam", "F", "k", "alpha0", "beta0")})
    jobs = [(t, d, cv, 1000*i + j) for i, t in enumerate(TRUTH) for d in DESIGN for cv in (0.10, 0.03) for j in range(40)]
    with Pool() as p: res = p.map(sim, jobs, chunksize=4)
    json.dump(dict(A=A, C=C, results=res), open("design_sim_results.json", "w"), default=float)
    import itertools
    for t in TRUTH:
        for d in DESIGN:
            for cv in (0.10, 0.03):
                v = np.array([x["dAIC"] for x in res if x["truth"] == t and x["design"] == d and x["cv"] == cv])
                print(f"{t:22s} {d:28s} cv={cv:.2f}: P(dAIC<-2, exact preferred)={np.mean(v < -2):.2f}  P(dAIC>2, LQ preferred)={np.mean(v > 2):.2f}  median dAIC={np.median(v):.2f}")
