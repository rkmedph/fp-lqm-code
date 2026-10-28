import numpy as np
from discrete import rates
def gillespie_pool(ncell, M, lam_d, D, mu, s2, kap, times, seed=0):
    """Exact SSA for ncell*M independent domains (repair phase); returns cell sums at times."""
    rng=np.random.default_rng(seed); n=rng.poisson(lam_d*D,size=ncell*M).astype(np.int64)
    t=np.zeros(n.size); times=sorted(times); out={}; state=n.copy()
    # event-driven: advance each chain until it passes each observation time
    for T in times:
        active=np.nonzero(t<T)[0]
        while active.size:
            wp,wm=rates(state[active],mu,s2,kap); W=wp+wm
            dead=W<=0
            t[active[dead]]=np.inf  # n=0: absorbing
            a=active[~dead]; wp=wp[~dead]; W=W[~dead]
            tau=rng.exponential(1/W); tn=t[a]+tau
            jump=tn<=T
            aj=a[jump]; up=rng.random(aj.size)<wp[jump]/W[jump]
            state[aj]+=np.where(up,1,-1); t[aj]=tn[jump]
            # chains whose next event is after T: memoryless -> set t=T and resample later
            t[a[~jump]]=T
            active=aj[t[aj]<T]
        out[T]=state.reshape(ncell,M).sum(1).copy()
    return out
if __name__=="__main__":
    from recovery import moments
    ts=[0.5,2,6]; pool=gillespie_pool(60000,4,1.0,2.0,.5,1.5,.2,ts,seed=1)
    for T in ts:
        x=pool[T]; m,v=moments(T,8.,.5,1.5,.2,4); print(T,x.mean(),m,x.var(),v, 'max',x.max())
