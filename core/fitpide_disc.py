import numpy as np, json
from scipy.optimize import least_squares
from discrete import survival_disc, ab_disc
LINES={"CHO":(0.151,0.037),"A549":(0.191,0.026),"HCT116":(0.388,0.047),"HeLa":(0.300,0.040),"PC3":(0.195,0.057)}
D=np.linspace(0.25,8,32)
def lnS(p,nc,M):
    lam,Fm,k=np.exp(p); F=0.5+Fm
    return np.log(np.maximum(survival_disc(D,lam,1.0,2*F,k,nc,M),1e-300))
out={}
for name,(a,b) in LINES.items():
    y=-(a*D+b*D*D); out[name]={}
    for M in [1,2,4]:
        best=None
        for nc in range(2,31):
            for p0 in ([np.log(0.5/M),0.0,np.log(0.3)],[np.log(2/M),np.log(3),np.log(1.0)]):
                r=least_squares(lambda p:lnS(p,nc,M)-y,p0,bounds=(np.log([1e-3,1e-3,1e-4]),np.log([100,100,100])))
                rms=float(np.sqrt(np.mean(r.fun**2)))
                if best is None or rms<best[0]:
                    lam,Fm,k=np.exp(r.x); best=(rms,nc,float(lam),float(0.5+Fm),float(k))
        out[name][M]=best; print(name,M,[round(x,4) if isinstance(x,float) else x for x in best],flush=True)
json.dump(out,open("fitpide_disc.json","w"),indent=1)
