"""Summarise predictive_results.json: in-sample criteria (with error scale x1, x1.5, x2), LOO and extrapolation scores."""
import json, csv
import numpy as np
R = json.load(open("predictive_results.json"))
names = sorted(set(r["name"] for r in R), key=lambda n: [r["name"] for r in R].index(n))
rows = []
for nm in names:
    full = [r for r in R if r["name"] == nm and r["task"] == "full"][0]
    n = full["n"]; kL, kE = full["LQ"]["k"], full["EX"]["k"]
    rec = dict(dataset=nm, n_points=n, k_LQ=kL, k_exact=kE)
    for sc in (1.0, 1.5, 2.0):
        cL, cE = full["LQ"]["chi2"]/sc**2, full["EX"]["chi2"]/sc**2
        rec[f"dchi2_x{sc}"] = cE - cL; rec[f"dAIC_x{sc}"] = (cE + 2*kE) - (cL + 2*kL); rec[f"dBIC_x{sc}"] = (cE + kE*np.log(n)) - (cL + kL*np.log(n))
    rec["chi2_LQ"] = full["LQ"]["chi2"]; rec["chi2_exact"] = full["EX"]["chi2"]; rec["exact_M"] = full["EX"]["M"]; rec["exact_nc"] = full["EX"]["nc"]
    rec["exact_F_r"] = full["EX"]["F"]; rec["exact_kappa_mu"] = full["EX"]["kappa"]
    loo = [r for r in R if r["name"] == nm and r["task"].startswith("loo")]
    if loo:
        zL = np.array([(r["y"] - r["yLQ"])/r["s"] for r in loo]); zE = np.array([(r["y"] - r["yEX"])/r["s"] for r in loo])
        rec.update(n_loo=len(loo), loo_mz2_LQ=float(np.mean(zL**2)), loo_mz2_exact=float(np.mean(zE**2)),
                   loo_mabs_LQ=float(np.mean([abs(r["y"] - r["yLQ"]) for r in loo])), loo_mabs_exact=float(np.mean([abs(r["y"] - r["yEX"]) for r in loo])),
                   loo_exact_better=int(np.sum(zE**2 < zL**2)))
    for t in ("lo2hi", "hi2lo"):
        e = [r for r in R if r["name"] == nm and r["task"] == t]
        if e:
            e = e[0]; y = np.array(e["y"]); s = np.array(e["s"]); yl = np.array(e["yLQ"]); ye = np.array(e["yEX"])
            rec[f"{t}_D"] = e["D"]; rec[f"{t}_mz2_LQ"] = float(np.mean(((y - yl)/s)**2)); rec[f"{t}_mz2_exact"] = float(np.mean(((y - ye)/s)**2))
            rec[f"{t}_mabs_LQ"] = float(np.mean(abs(y - yl))); rec[f"{t}_mabs_exact"] = float(np.mean(abs(y - ye)))
    rows.append(rec)
json.dump(rows, open("predictive_summary.json", "w"), indent=1, default=float)
keys = ["dataset", "n_points", "k_LQ", "k_exact", "dchi2_x1.0", "dAIC_x1.0", "dBIC_x1.0", "dAIC_x1.5", "dAIC_x2.0", "loo_mz2_LQ", "loo_mz2_exact", "loo_exact_better", "n_loo",
        "lo2hi_mz2_LQ", "lo2hi_mz2_exact", "hi2lo_mz2_LQ", "hi2lo_mz2_exact", "exact_M", "exact_nc", "exact_F_r"]
with open("table_model_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(keys)
    for r in rows: w.writerow([("%.3g" % r[k]) if isinstance(r.get(k), float) else r.get(k, "") for k in keys])
for r in rows:
    print(f"{r['dataset'][:44]:44s} n={r['n_points']:2d} k={r['k_LQ']}/{r['k_exact']} dchi2={r['dchi2_x1.0']:7.2f} dAIC={r['dAIC_x1.0']:6.2f} dBIC={r['dBIC_x1.0']:6.2f} "
          f"dAIC(x1.5)={r['dAIC_x1.5']:5.2f} dAIC(x2)={r['dAIC_x2.0']:5.2f} | LOO z2 LQ {r.get('loo_mz2_LQ', np.nan):6.2f} ex {r.get('loo_mz2_exact', np.nan):6.2f} "
          f"({r.get('loo_exact_better', '-')}/{r.get('n_loo', '-')}) | lo->hi {r.get('lo2hi_mz2_LQ', np.nan):6.2f}/{r.get('lo2hi_mz2_exact', np.nan):6.2f} hi->lo {r.get('hi2lo_mz2_LQ', np.nan):6.2f}/{r.get('hi2lo_mz2_exact', np.nan):6.2f}")
# aggregate
L = [r for r in rows if "loo_mz2_LQ" in r]
print("\nLOO aggregate: datasets where exact has lower mean z^2:", sum(r["loo_mz2_exact"] < r["loo_mz2_LQ"] for r in L), "of", len(L))
print("median ratio exact/LQ LOO mean z^2:", np.median([r["loo_mz2_exact"]/r["loo_mz2_LQ"] for r in L]))
for t in ("lo2hi", "hi2lo"):
    T = [r for r in rows if f"{t}_mz2_LQ" in r]
    print(t, "exact better in", sum(r[f"{t}_mz2_exact"] < r[f"{t}_mz2_LQ"] for r in T), "of", len(T), "median ratio", np.median([r[f"{t}_mz2_exact"]/r[f"{t}_mz2_LQ"] for r in T]))
print("dAIC<-2 (exact preferred):", sum(r["dAIC_x1.0"] < -2 for r in rows), " dAIC>2 (LQ preferred):", sum(r["dAIC_x1.0"] > 2 for r in rows), "of", len(rows))
print("at x2 errors: dAIC<-2:", sum(r["dAIC_x2.0"] < -2 for r in rows), " >2:", sum(r["dAIC_x2.0"] > 2 for r in rows))
print("dBIC<-2:", sum(r["dBIC_x1.0"] < -2 for r in rows), " dBIC>2:", sum(r["dBIC_x1.0"] > 2 for r in rows))
