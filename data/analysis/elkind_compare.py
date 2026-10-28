import numpy as np, json
from scipy.optimize import least_squares
from curves import CURVES
c=[c for c in CURVES if c["cell"]=="Chinese hamster"][0]
D,S=c["D"],c["S"]; y=-np.log(S)
r=json.load(open("fits.json"))[-1]
print("exact model M=1 params:",r["par_M1"])
def fit(f,p0,lb,ub,sig):
    res=least_squares(lambda p:(f(p,D)-y)/sig,p0,bounds=(lb,ub)); return res.x, float(res.fun@res.fun)
models={
 "LQ": (lambda p,D:p[0]*D+p[1]*D*D,[0.2,0.03],[-5,-1],[5,1]),
 "LQ-L (Astrahan)": (lambda p,D:np.where(D<p[2],p[0]*D+p[1]*D*D,p[0]*p[2]+p[1]*p[2]**2+(p[0]+2*p[1]*p[2])*(D-p[2])),[0.2,0.05,5],[0,0,0.5],[5,1,15]),
 "single-hit multi-target": (lambda p,D:-np.log(np.exp(-p[0]*D)*(1-(1-np.exp(-D/p[1]))**p[2])),[0.1,1.5,5],[0,0.1,1],[5,10,100]),
}
for sigrel in [0.10,0.20]:
    sig=sigrel*np.ones_like(y)  # sigma of ln S
    print(f"--- assumed relative error {sigrel:.0%}")
    for name,(f,p0,lb,ub) in models.items():
        p,chi2=fit(f,p0,lb,ub,sig); k=len(p0); print(f"{name:24s} chi2={chi2:6.2f} k={k} AIC={chi2+2*k:6.2f} params={np.round(p,3)}")
    chi2m=r["chi2_M1"]*(0.10/sigrel)**2; print(f"{'exact model, M=1':24s} chi2={chi2m:6.2f} k=4 AIC={chi2m+8:6.2f}")
