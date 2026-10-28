import numpy as np, json
from scipy.optimize import least_squares
from recovery import moments
from gsim import gillespie_pool

def stats(x):
    n=x.size; m=x.mean(); v=x.var(ddof=1); m4=np.mean((x-m)**4)
    se_m=np.sqrt(max(v,1.0/n)/n); se_v=np.sqrt(max(m4-v*v*(n-3)/(n-1),1.0/n)/n)
    return m,v,se_m,se_v

def fit(ts,mo,vo,sm,sv,Mfix=None):
    ts=np.asarray(ts)
    def unpack(p): return (*p,Mfix) if Mfix else tuple(p)
    def res(p):
        m0,mu,s2,k,M=unpack(p)
        if abs(mu-k)<1e-3: return np.full(2*len(ts),1e3)
        m,v=moments(ts,m0,mu,s2,k,M); return np.concatenate([(mo-m)/sm,(vo-v)/sv])
    lb=[0.1,0.01,0.01,0.0]+([] if Mfix else [0.2]); ub=[100,5,50,2.0]+([] if Mfix else [200])
    starts=[[mo[0]*1.2,0.4,1.0,0.1,2],[mo[0]*1.2,0.6,2.0,0.4,8],[mo[0]*1.2,0.5,1.5,0.8,30]]
    best=None
    for p0 in starts:
        p0=np.clip(p0[:len(lb)],np.array(lb)+1e-6,np.array(ub)-1e-6)
        r=least_squares(res,p0,bounds=(lb,ub))
        if best is None or r.cost<best.cost: best=r
    return unpack(best.x)

if __name__=='__main__':
    D,lam_d,M=2.0,1.0,4
    designs={"K3":[0.5,2.0,6.0],"K8":[0.25,0.5,1,2,3,4,6,8.0],"K16":list(np.round(np.linspace(0.25,8,16),3))}
    allt=sorted(set(sum(designs.values(),[])))
    scen={"base":(0.5,1.5,0.2),"near":(0.5,1.5,0.45)}
    out={}
    for sn,(mu,s2,k) in scen.items():
        pool=gillespie_pool(300000,M,lam_d,D,mu,s2,k,allt,seed=11 if sn=="base" else 12)
        abt=(s2-mu+k)/(mu*lam_d)
        for dn,ts in designs.items():
            for nc in [1000,10000]:
                for Mfix in [None,M]:
                    if sn=="near" and (dn!="K8" or Mfix is None): continue
                    if Mfix is None and dn!="K8": continue
                    rng=np.random.default_rng(5); est=[]
                    for rep in range(150):
                        S=[stats(rng.choice(pool[t],nc,replace=False)) for t in ts]; S=np.array(S)
                        m0,mu_h,s2_h,k_h,M_h=fit(ts,S[:,0],S[:,1],S[:,2],S[:,3],Mfix)
                        lamd=m0/(D*M_h); est.append([mu_h,s2_h,k_h,M_h,(s2_h-mu_h+k_h)/(mu_h*lamd)])
                    est=np.array(est); tr=np.array([mu,s2,k,M,abt]); key=f"{sn}|{dn}|n{nc}|{'Mfixed' if Mfix else 'Mfree'}"
                    q=np.percentile(est,[10,50,90],axis=0)
                    out[key]={"q":q.tolist(),"truth":tr.tolist(),"est":est.tolist()}
                    print(key,"alpha/beta 10-50-90%:",np.round(q[:,4],2),"| kappa:",np.round(q[:,2],3),"| M:",np.round(q[:,3],1),flush=True)
    json.dump(out,open("recovery2.json","w"))
