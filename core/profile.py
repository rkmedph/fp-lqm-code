import numpy as np, json
from scipy.optimize import least_squares
from discrete import survival_disc
D=np.linspace(0.25,8,32)
LINES={"CHO":(0.151,0.037),"HCT116":(0.388,0.047)}
Fs=[0.6,0.8,1.0,1.5,2,3,5,8,13,20,35,60]
out={}
for name,(a,b) in LINES.items():
    y=-(a*D+b*D*D); prof=[]
    for F in Fs:
        best=9
        for nc in range(2,21):
            for p0 in ([np.log(0.5),np.log(0.3)],[np.log(1.5),np.log(3.0)]):
                f=lambda p: np.log(np.maximum(survival_disc(D,np.exp(p[0]),1.0,2*F,np.exp(p[1]),nc,2),1e-300))-y
                r=least_squares(f,p0,bounds=(np.log([1e-3,1e-4]),np.log([100,100])))
                best=min(best,float(np.sqrt(np.mean(r.fun**2))))
        prof.append(best); print(name,F,round(best,4),flush=True)
    out[name]=prof
out["F"]=Fs; json.dump(out,open("profile.json","w"))
