"""Extra designs for the simulated experiment (same truths, noise model and fits as design_sim.py)."""
import json, numpy as np
from multiprocessing import Pool
import design_sim as d
d.DESIGN.clear()
d.DESIGN.update({"combined (0.25-12 Gy, 9 doses)": np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12.]),
                 "high (2-12 Gy, 6 doses)": np.array([2, 4, 6, 8, 10, 12.])})
if __name__ == "__main__":
    jobs = [(t, k, cv, 5000 + 1000*i + j) for i, t in enumerate(["A exact shoulder", "B LQ", "C exact intermediate"]) for k in d.DESIGN for cv in (0.10, 0.03) for j in range(40)]
    with Pool() as p: res = p.map(d.sim, jobs, chunksize=4)
    json.dump(res, open("design_sim2_results.json", "w"), default=float)
    for t in ["A exact shoulder", "B LQ", "C exact intermediate"]:
        for k in d.DESIGN:
            for cv in (0.10, 0.03):
                v = np.array([x["dAIC"] for x in res if x["truth"] == t and x["design"] == k and x["cv"] == cv])
                print(f"{t:22s} {k:32s} cv={cv:.2f}: P(exact pref)={np.mean(v < -2):.2f} P(LQ pref)={np.mean(v > 2):.2f} median dAIC={np.median(v):.2f}")
