import sys, numpy as np, pandas as pd, json
sys.path.insert(0,'.')
from load import load, photons
d=load(); p=photons(d)
out={}
def summarize(a,b,label):
    m=a.notna()&b.notna(); a=a[m]; b=b[m]
    n=len(a); neg=(b<=0).sum(); pos=b>0
    R=(a[pos]**2)/(2*b[pos])
    res=dict(n=int(n), beta_le0=int(neg), frac_beta_le0=float(neg/n),
             n_pos=int(pos.sum()), R_median=float(R.median()), R_q=[float(x) for x in R.quantile([.1,.25,.75,.9])],
             frac_R_gt1=float((R>1).mean()), frac_R_gt2=float((R>2).mean()), frac_R_gt5=float((R>5).mean()),
             ab_median=float((a[pos]/b[pos]).median()), ab_IQR=[float(x) for x in (a[pos]/b[pos]).quantile([.25,.75])],
             alpha_median=float(a.median()), beta_median=float(b.median()))
    out[label]=res; print(label, json.dumps(res))
    return R
R_fit=summarize(p.ax_fit,p.bx_fit,"GSI fit")
R_pap=summarize(p.ax_paper,p.bx_paper,"paper")
# agreement between two estimates
m=p[["ax_fit","bx_fit","ax_paper","bx_paper"]].notna().all(1)&(p.bx_fit>0)&(p.bx_paper>0)
q=p[m]; Rf=q.ax_fit**2/(2*q.bx_fit); Rp=q.ax_paper**2/(2*q.bx_paper)
agree=((Rf>1)==(Rp>1)).mean()
print("both available",len(q),"class agreement R>1:",round(agree,3),"Spearman R",round(pd.Series(Rf.values).corr(pd.Series(Rp.values),method='spearman'),3))
print("among R_fit>1: R_paper>1 fraction",round(((Rp>1)&(Rf>1)).sum()/max((Rf>1).sum(),1),3), "n",int((Rf>1).sum()))
out["agreement"]=dict(n=int(len(q)),agree=float(agree))
# subgroups (GSI fit)
p2=p[p.ax_fit.notna()&p.bx_fit.notna()].copy(); p2["R"]=np.where(p2.bx_fit>0,p2.ax_fit**2/(2*p2.bx_fit),np.inf)
p2["viol"]=p2.R>1
for c in ["CellClass","CellOrigin"]:
    g=p2.groupby(c).agg(n=("R","size"),frac_viol=("viol","mean"),alpha_med=("ax_fit","median"),beta_med=("bx_fit","median"))
    print(g.to_string()); out[c]=g.reset_index().to_dict("records")
p2["sync"]=np.where(p2.CellCycle=="a","async","synchronized")
print(p2.groupby("sync").agg(n=("R","size"),frac_viol=("viol","mean")).to_string())
# alpha dependence: violation vs alpha quantile
p2["aq"]=pd.qcut(p2.ax_fit,4,labels=["Q1","Q2","Q3","Q4"])
g=p2.groupby("aq",observed=True).agg(n=("R","size"),alpha_range_lo=("ax_fit","min"),alpha_range_hi=("ax_fit","max"),frac_viol=("viol","mean"),beta_med=("bx_fit","median"))
print(g.to_string())
# top violators
print(p2.sort_values("R",ascending=False)[["Cells","PublicationName","ax_fit","bx_fit","R","CellClass","CellOrigin"]].head(15).to_string())
p2.to_csv("photon_fit_table.csv",index=False)
json.dump(out,open("summary.json","w"),indent=1,default=str)
