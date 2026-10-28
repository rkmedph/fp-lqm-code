import numpy as np, json
from discrete import rates, survival_disc, ab_disc
lam,mu,s2,kap,nc=1.0,0.5,1.5,0.2,3
def surv(D, Ddot, N=40000, seed=0):
    rng=np.random.default_rng(seed); n=np.zeros(N,int); alive=np.ones(N,bool); T=D/Ddot; t=0.0
    while True:
        irr=t<T; dt=min(2e-3 if irr else 5e-3, 0.02/max(lam*Ddot,1e-9)) if irr else 5e-3
        if irr: dt=min(dt,T-t)
        i=np.nonzero(alive&((n>0)|irr))[0]
        if i.size==0: break
        wp,wm=rates(n[i],mu,s2,kap,src=lam*Ddot if irr else 0.0)
        # exact per-step: at most one event per step, probabilities w*dt (dt small)
        u=rng.random(i.size); pp=wp*dt; pm=wm*dt
        n[i]+= (u<pp).astype(int) - ((u>=pp)&(u<pp+pm)).astype(int)
        alive[i[n[i]>=nc]]=False; t+=dt
        if t>T+80: break
    return alive.mean()
if __name__=="__main__":
    doses=[1,2,4,6,8]; res={}
    for Ddot in [60.0,6.0,1.0,0.3]:
        res[str(Ddot)]=[surv(D,Ddot,seed=int(10*D+Ddot)) for D in doses]; print(Ddot,res[str(Ddot)],flush=True)
    res["acute_exact"]=list(survival_disc(np.array(doses,float),lam,mu,s2,kap,nc))
    json.dump(res,open("doserate_disc.json","w"),indent=1)
