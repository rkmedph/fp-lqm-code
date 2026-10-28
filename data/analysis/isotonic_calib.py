"""Size of the monotonicity test under H0: convex (LQ, beta >= 0) curves on the 91 Polgár designs, independent
point errors from the digitised whiskers and a common normalisation offset l ~ N(0, sc_true^2)."""
import json, numpy as np
from isotonic_norm import test as test_norm
from isotonic_test import test as test_fixed
R = json.load(open("polgar_results.json")); rng = np.random.default_rng(11)
for sc_true, sc_test in [(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.03, 0.03)]:
    rej = 0; n = 0
    for rep in range(2):
        for r in R:
            D = np.array(r["D"]); s = np.array(r["sig"]); a, b = rng.uniform(0.05, 0.4), rng.uniform(0.0, 0.06)
            y = -(a*D + b*D*D) + (rng.normal(0, sc_true) if sc_true > 0 else 0) + rng.normal(0, s)
            p = test_fixed(D, y, s, B=200, rng=rng)[1] if sc_test == 0 else test_norm(D, y, s, sc_test, 200, rng)[2]
            rej += p < 0.05; n += 1
    print(f"true offset SD {sc_true:.2f}, test assumes {sc_test:.2f}: rejection rate {rej/n:.3f} ({n} simulated datasets)", flush=True)
