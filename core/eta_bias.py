import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from recovery import moments
from recovery2 import fit
ts=np.array([0.25,0.5,1,2,3,4,6,8.]); D,lam_d,M,mu,s2,k=2.0,1.0,4,0.5,1.5,0.2
ab=lambda p:(p[2]-p[1]+p[3])/(p[1]*(p[0]/(D*p[4])))
out={"eta":[],"ab_eta":[],"M_eta":[],"bg":[],"ab_bg":[],"M_bg":[]}
m,v=moments(ts,M*lam_d*D,mu,s2,k,M)
for eta in [0.5,0.7,0.9,1.0]:
    mY=eta*m; vY=eta**2*v+eta*(1-eta)*m; p=fit(ts,mY,vY,0.01*mY,0.01*vY)
    out["eta"].append(eta); out["ab_eta"].append(ab(p)); out["M_eta"].append(p[4])
    print("eta",eta,"M=%.3f ab=%.3f"%(p[4],ab(p)))
for bg in [0.0,0.25,0.5,1.0,2.0]:
    mY=m+bg; vY=v+bg; pf=fit(ts,mY,vY,0.01*mY,0.01*vY,Mfix=M); p=fit(ts,mY,vY,0.01*mY,0.01*vY)
    out["bg"].append(bg); out["ab_bg"].append(ab(pf)); out["M_bg"].append(p[4])
    print("bg",bg,"Mfixed ab=%.3f | Mfree M=%.2f ab=%.2f"%(ab(pf),p[4],ab(p)))
json.dump(out,open("eta_bias.json","w"))
