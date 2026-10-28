"""Synthetic test data in a PIDE-like long format: id dose sf (homogeneous curves + mixtures)."""
import numpy as np
from discrete import survival_disc
rng = np.random.default_rng(0); rows = []
for i in range(40):
    D = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8.])
    if i < 30:
        S = survival_disc(D, 10**rng.uniform(-0.5, 0.4), 1.0, 2*10**rng.uniform(0, 0.6), 10**rng.uniform(-2, 0), int(rng.integers(2, 6)), int(rng.choice([1, 4])))
    else:
        f = rng.uniform(0.85, 0.97)
        S = f*survival_disc(D, 1.5, 1.0, 3.0, 0.2, 3) + (1-f)*survival_disc(D, 0.25, 1.0, 3.0, 0.2, 3)
    S = S*np.exp(rng.normal(0, 0.05, D.size))
    rows += [f"{i+1} {d:g} {s:.5g}" for d, s in zip(D, S) if s > 1e-5]
open("synthetic_raw.dat", "w").write("# id dose sf\n" + "\n".join(rows) + "\n")
