import numpy as np
from discrete import rates
def hit_ssa(n0, mu, s2, kap, nc, N=100000, seed=0):
    """Embedded jump chain is enough for hitting probabilities."""
    rng=np.random.default_rng(seed); n=np.full(N,n0); act=np.ones(N,bool); hit=np.zeros(N,bool)
    while act.any():
        i=np.nonzero(act)[0]; wp,wm=rates(n[i],mu,s2,kap)
        up=rng.random(i.size)<wp/(wp+wm); n[i]+=np.where(up,1,-1)
        hit[i[n[i]>=nc]]=True; act[i[(n[i]>=nc)|(n[i]<=0)]]=False
    return hit.mean()
if __name__=="__main__":
    from discrete import h_disc
    for kap in [0.0,0.2,0.6]:
        hd=h_disc(3,.5,1.5,kap)
        for n0 in [1,2]:
            p=hit_ssa(n0,.5,1.5,kap,3,seed=n0); print(kap,n0,round(p,4),'+-',round(np.sqrt(p*(1-p)/1e5),4),'exact',round(hd[n0],4))
