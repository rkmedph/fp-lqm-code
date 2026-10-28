"""Parameter recovery from cell-level lesion kinetics N(t)=sum_j n_j(t)."""
import numpy as np, json, sys
from scipy.optimize import least_squares
from discrete import rates

def moments(t, m0, mu, s2, kap, M):
    m = m0*np.exp(-mu*t)
    C = m0 - s2*m0/(mu-kap) + m0**2/M
    v = s2*m0*np.exp(-mu*t)/(mu-kap) - (m0**2/M)*np.exp(-2*mu*t) + C*np.exp(-(2*mu-kap)*t)
    return m, v

def simulate_pool(ncell, M, lam_d, D, mu, s2, kap, times, dt=5e-3, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam_d*D, size=(ncell, M))
    out = {}; t = 0.0; times = sorted(times); k = 0
    while k < len(times):
        if t >= times[k]-1e-9:
            out[times[k]] = n.sum(axis=1).copy(); k += 1; continue
        wp, wm = rates(n, mu, s2, kap); u = rng.random(n.shape)
        n = n + (u < wp*dt) - ((u >= wp*dt) & (u < (wp+wm)*dt)); t += dt
    return out

def fit(times, mobs, vobs, ncells, p0s=None):
    times = np.asarray(times)
    def res(p):
        m0, mu, s2, kap, M = p
        if abs(mu-kap) < 1e-3: return np.full(2*len(times), 1e3)
        m, v = moments(times, m0, mu, s2, kap, M)
        sm = np.sqrt(np.maximum(v, 1e-6)/ncells); sv = np.sqrt(2/(ncells-1))*np.maximum(v, 1e-6)
        return np.concatenate([(mobs-m)/sm, (vobs-v)/sv])
    lb = [0.1, 0.01, 0.01, 0.0, 0.2]; ub = [100, 5, 50, 2.0, 200]
    best = None
    starts = p0s or [[mobs[0]*1.3, 0.4, 1.0, 0.1, 2], [mobs[0]*1.3, 0.6, 2.0, 0.4, 8], [mobs[0]*1.3, 0.5, 1.5, 0.8, 30]]
    for p0 in starts:
        p0 = np.clip(p0, np.array(lb)+1e-6, np.array(ub)-1e-6)
        try: r = least_squares(res, p0, bounds=(lb, ub))
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    return best.x

if __name__ == "__main__":
    D = 2.0; lam_d = 1.0; M = 4
    designs = {"K3": [0.5, 2.0, 6.0], "K8": [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]}
    alltimes = sorted(set(designs["K3"] + designs["K8"]))
    scen = {"base": (0.5, 1.5, 0.2), "near": (0.5, 1.5, 0.45)}
    POOL = 60000; results = {}
    for sname, (mu, s2, kap) in scen.items():
        from gsim import gillespie_pool
        pool = gillespie_pool(POOL, M, lam_d, D, mu, s2, kap, alltimes, seed=1 if sname=="base" else 2)
        ab_true = (s2-mu+kap)/(mu*lam_d)
        for dname, ts in designs.items():
            for nc in ([100, 1000] if sname == "base" else [1000]):
                if sname == "near" and dname == "K3": continue
                rng = np.random.default_rng(7); est = []
                for rep in range(200):
                    mo, vo = [], []
                    for t in ts:
                        x = rng.choice(pool[t], nc, replace=False); mo.append(x.mean()); vo.append(x.var(ddof=1))
                    m0, mu_h, s2_h, k_h, M_h = fit(ts, np.array(mo), np.array(vo), nc)
                    lam_h = m0/(D*M_h); ab_h = (s2_h-mu_h+k_h)/(mu_h*lam_h)
                    est.append([m0/D, mu_h, s2_h, k_h, M_h, ab_h])
                est = np.array(est); key = f"{sname}_{dname}_n{nc}"
                truth = np.array([M*lam_d, mu, s2, kap, M, ab_true])
                results[key] = {"est": est.tolist(), "truth": truth.tolist()}
                rel = (est-truth)/truth
                print(key, "median rel.err", np.round(np.median(rel, 0), 3), "IQR", np.round(np.subtract(*np.percentile(rel, [75, 25], 0)), 3), flush=True)
    json.dump(results, open("recovery.json", "w"))
