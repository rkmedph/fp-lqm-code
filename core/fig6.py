import numpy as np, json, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from convexity_study import neglogS
C=["#2a78d6","#eb6834","#1baf7a","#eda100"]; INK="#333333"
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":"#888","axes.labelcolor":INK,"xtick.color":"#555","ytick.color":"#555",
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"font.family":"DejaVu Sans"})
fits=json.load(open("rangefit_ab.json")); p=pd.read_csv("photon_fit_table.csv")
fig,ax=plt.subplots(1,2,figsize=(7.4,3.1))
A=np.array(fits["1"]["alpha"]); B=np.array(fits["1"]["beta"])
ax[0].scatter(A,B,s=4,c="#bbbbbb",marker=".",label="model, $M$=1, LQ fit to 0.5–10 Gy",rasterized=True)
pos=p[p.bx_fit>0]; ax[0].scatter(pos.ax_fit,pos.bx_fit,s=13,facecolors="white",edgecolors=C[0],linewidths=0.9,label=r"PIDE photons, $\beta>0$")
z=p[p.bx_fit==0]; ax[0].scatter(z.ax_fit,np.full(len(z),1.25e-3),s=14,marker="s",facecolors="white",edgecolors=C[3],linewidths=1,label=r"PIDE, $\beta=0$")
ng=p[p.bx_fit<0]; ax[0].scatter(ng.ax_fit,np.full(len(ng),1.0e-3),s=16,marker="v",facecolors=C[1],edgecolors=C[1],label=r"PIDE, $\beta<0$")
ax[0].set(xscale="log",yscale="log",xlim=(5e-3,4),ylim=(8e-4,1.5),xlabel=r"$\alpha$ (Gy$^{-1}$)",ylabel=r"$\beta$ (Gy$^{-2}$)")
ax[0].legend(fontsize=6.5,loc="upper left",markerscale=1.3)
D=np.linspace(0.2,10,120)
for i,(lam,F,k,nc,lab,ls) in enumerate([(1.5,1.5,0.2,3,"homogeneous, sensitive","-"),(0.25,1.5,0.2,3,"homogeneous, resistant","--"),(1.0,4.0,1.0,8,"homogeneous, $n_c$=8","-.")]):
    y=neglogS(D,lam,1.0,2*F,k,nc); ax[1].plot(D,y/D,ls,color=C[i],label=lab)
f=0.92; Sm=f*np.exp(-neglogS(D,1.5,1.0,3.0,0.2,3))+(1-f)*np.exp(-neglogS(D,0.25,1.0,3.0,0.2,3))
ax[1].plot(D,-np.log(Sm)/D,":",color=INK,lw=2,label="92 : 8 mixture")
ax[1].set(xlabel="dose $D$ (Gy)",ylabel=r"$-\ln S/D$ (Gy$^{-1}$)",ylim=(0,1.05)); ax[1].legend(fontsize=6.5,loc="upper left")
for a_,t in zip(ax,["(a)","(b)"]): a_.text(0.93,0.98,t,transform=a_.transAxes,va="top",fontweight="bold")
fig.tight_layout(); fig.savefig("fig6.pdf"); fig.savefig("fig6.png",dpi=150)
