"""Discrete birth-death master equation whose first two Kramers-Moyal moments are
A(n) = lam*Ddot - mu*n,  B(n) = lam*Ddot + s2*n + kap*n^2."""
import numpy as np
from scipy.stats import poisson

def rates(n, mu, s2, kap, src=0.0):
    n = np.asarray(n, float)
    wp = src + 0.5*((s2-mu)*n + kap*n*n)
    wm = 0.5*((s2+mu)*n + kap*n*n)
    return wp, wm

def h_disc(nc, mu, s2, kap):
    """Exact hitting probabilities h_k, k=0..nc (absorbing at 0 and nc), repair phase."""
    nc = int(nc)
    i = np.arange(1, nc)
    wp, wm = rates(i, mu, s2, kap)
    rho = np.concatenate([[1.0], np.cumprod(wm/wp)])      # rho_0..rho_{nc-1}
    S = np.concatenate([[0.0], np.cumsum(rho)])             # S_0..S_nc
    return S/S[-1]

def survival_disc(D, lam, mu, s2, kap, nc, M=1):
    h = h_disc(nc, mu, s2, kap); nc = int(nc)
    D = np.atleast_1d(D).astype(float); out = []
    for d in D:
        m = lam*d; k = np.arange(nc)
        p = np.sum(poisson.pmf(k, m)*h[:nc]) + poisson.sf(nc-1, m)
        out.append((1-p)**M)
    return np.array(out)

def ab_disc(lam, mu, s2, kap, nc, M=1):
    h = h_disc(nc, mu, s2, kap)
    a = M*lam*h[1]; b = M*lam**2*(h[2]/2 - h[1]) + a*a/(2*M)
    return a, b

def ab_ratio_leading(lam, mu, s2, kap):
    return (s2 - mu + kap)/(mu*lam)

def stationary_disc(nmax, lamD, mu, s2, kap):
    n = np.arange(nmax+1)
    wp_prev, _ = rates(n[:-1], mu, s2, kap, src=lamD)
    _, wm = rates(n[1:], mu, s2, kap)
    logp = np.concatenate([[0.0], np.cumsum(np.log(wp_prev) - np.log(wm))])
    p = np.exp(logp - logp.max()); return n, p/p.sum()
