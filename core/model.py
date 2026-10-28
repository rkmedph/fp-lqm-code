"""Exact results for the Fokker-Planck DSB model with A=-mu n, B=s2 n + kap n^2 (repair phase)."""
import numpy as np
from scipy.stats import poisson

def h(n, mu, s2, kap, nc):
    """Probability that a trajectory started at n reaches nc before 0 (scale function)."""
    n = np.minimum(np.asarray(n, float), nc)
    if kap == 0:
        c = 2*mu/s2
        return np.expm1(c*n)/np.expm1(c*nc)
    q, b1 = kap/s2, 2*mu/kap + 1
    return np.expm1(b1*np.log1p(q*n))/np.expm1(b1*np.log1p(q*nc))

def survival_acute(D, lam, mu, s2, kap, nc, M=1):
    """Exact S(D): Poisson(lam_d D) initial lesions in each of M independent domains."""
    D = np.atleast_1d(D).astype(float)
    kmax = int(nc) + 1
    out = []
    for d in D:
        m = lam*d
        ks = np.arange(0, kmax + 1)
        pk = poisson.pmf(ks, m)
        hk = h(ks, mu, s2, kap, nc); hk[ks >= nc] = 1.0
        p = np.sum(pk[:-1]*hk[:-1]) + poisson.sf(kmax - 1, m)  # k>=kmax dies
        out.append((1 - p)**M)
    return np.array(out)

def alpha_beta(lam, mu, s2, kap, nc, M=1):
    """LQ coefficients of -ln S at D->0 (lam = per-domain yield)."""
    h1, h2 = h(1, mu, s2, kap, nc), h(2, mu, s2, kap, nc)
    a = M*lam*h1
    b = M*lam**2*(h2/2 - h1) + M*(lam*h1)**2/2
    return a, b
