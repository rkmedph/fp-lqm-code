"""Additional 160 simulations (total 200 with the 40 already run) for the four design entries that carry the
conclusions of section 5.11: truth A with the low and extended designs, truth C with the extended and combined
designs, cv = 10 %."""
import json, numpy as np
from multiprocessing import Pool
import design_sim as d
d.DESIGN["combined (0.25-12 Gy, 9 doses)"] = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12.])
CELLS = [("A exact shoulder", "low (0.1-8 Gy, 8 doses)"), ("A exact shoulder", "extended (1-10 Gy, 7 doses)"),
         ("C exact intermediate", "extended (1-10 Gy, 7 doses)"), ("C exact intermediate", "combined (0.25-12 Gy, 9 doses)")]
if __name__ == "__main__":
    jobs = [(t, k, 0.10, 90000 + 1000*i + j) for i, (t, k) in enumerate(CELLS) for j in range(160)]
    with Pool(2) as p: res = p.map(d.sim, jobs, chunksize=4)
    old = json.load(open("design_sim_results.json"))["results"] + json.load(open("design_sim2_results.json"))
    out = {}
    for t, k in CELLS:
        v0 = [x["dAIC"] for x in old if x["truth"] == t and x["design"] == k and abs(x["cv"] - 0.10) < 1e-9]
        v1 = [x["dAIC"] for x in res if x["truth"] == t and x["design"] == k]
        v = np.array(v0 + v1); p = float(np.mean(v < -2)); n = len(v)
        out[f"{t}|{k}"] = dict(n=n, p_first40=float(np.mean(np.array(v0) < -2)), p_new160=float(np.mean(np.array(v1) < -2)), p_all=p, se=float(np.sqrt(p*(1-p)/n)))
        print(f"{t:22s} {k:32s} n={n}: P(true exact preferred) first40={out[f'{t}|{k}']['p_first40']:.2f} new160={out[f'{t}|{k}']['p_new160']:.2f} all={p:.2f} +- {out[f'{t}|{k}']['se']:.2f}", flush=True)
    json.dump(dict(summary=out, raw=res), open("design_sim_extra_results.json", "w"), default=float)
