import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import betaprime
from model import h as h_fp
from discrete import *
C=["#2a78d6","#eb6834","#1baf7a","#eda100"]; INK="#333333"
plt.rcParams.update({"font.size":9,"axes.edgecolor":"#888","axes.labelcolor":INK,"xtick.color":"#555","ytick.color":"#555",
 "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.6,"legend.frameon":False,"font.family":"DejaVu Sans"})
mu,s2,nc,lam,kap0=0.5,1.5,3,1.0,0.2
def lab(ax,t,x=0.02,y=0.97): ax.text(x,y,t,transform=ax.transAxes,va="top",fontweight="bold")

# Fig 1 ---------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); LD=2.0
for i,k in enumerate([0.05,0.2,0.5]):
    a=2*LD/s2; b=2*mu/k; x=np.logspace(-1,2.3,400)
    ax[0].plot(x,betaprime.pdf(k*x/s2,a,b+1)*k/s2,color=C[i],ls=["-","--","-."][i],label=fr"$\kappa={k}$ h$^{{-1}}$ ($b={b:g}$)")
    n,p=stationary_disc(3000,LD,mu,s2,k); sel=(n>=1)&(p>1e-9)&((n<12)|(n%np.maximum(1,(n//8))==0))
    ax[0].plot(n[sel],p[sel],["o","^","s"][i],ms=3.2,mfc="white",mec=C[i],mew=1,ls="none")
    ax[1].plot(x,betaprime.sf(k*x/s2,a,b+1),color=C[i],ls=["-","--","-."][i])
    tail=np.cumsum(p[::-1])[::-1]; ax[1].plot(n[1:120][::3],tail[2:121][::3],["o","^","s"][i],ms=3,mfc="white",mec=C[i],mew=1,ls="none")
from scipy.stats import norm
gx=np.linspace(1,60,300); m=LD/mu
ax[1].plot(gx,norm.sf(gx,m,np.sqrt((LD+s2*m)/(2*mu))),"--",color="#777",lw=1.2,label=r"Gaussian, $\kappa=0$")
ax[0].set(xscale="log",yscale="log",ylim=(1e-8,1),xlabel=r"number of lesions $n$",ylabel=r"stationary probability"); ax[0].legend(fontsize=7,loc="lower left")
ax[1].set(xscale="log",yscale="log",ylim=(1e-8,1.2),xlim=(1,150),xlabel=r"threshold $n_c$",ylabel=r"$\Pr(n>n_c)$"); ax[1].legend(fontsize=7,loc="lower left")
lab(ax[0],"(a)",0.9); lab(ax[1],"(b)",0.9)
fig.tight_layout(); fig.savefig("fig1.pdf"); fig.savefig("fig1.png",dpi=150)

# Fig 2 ---------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); n0=np.linspace(0,nc,300)
for i,k in enumerate([0.0,0.2,0.6]):
    ax[0].plot(n0,h_fp(n0,mu,s2,k,nc),color=C[i],ls=["-","--","-."][i],label=fr"$\kappa={k}$ h$^{{-1}}$")
    hd=h_disc(nc,mu,s2,k); ax[0].plot(np.arange(nc+1),hd,["o","^","s"][i],ms=5,mfc="white",mec=C[i],mew=1.2,ls="none")
ax[0].plot([],[],"o",ms=5,mfc="white",mec="#555",ls="none",label="discrete (exact)"); ax[0].plot([],[],"-",color="#555",label="diffusion limit")
ax[0].set(xlabel=r"initial lesions $n_0$",ylabel=r"$h(n_0)$",xlim=(0,nc),ylim=(0,1.02)); ax[0].legend(fontsize=7,loc="upper left")
Dd=np.linspace(0,10,200); S=survival_disc(Dd,lam,mu,s2,kap0,nc); al,be=ab_disc(lam,mu,s2,kap0,nc)
ax[1].semilogy(Dd,S,color=C[1],label="exact (discrete)")
ax[1].semilogy(Dd,np.exp(-al*Dd-be*Dd**2),"--",color=C[0],label=fr"LQ limit: $\alpha$={al:.3f}, $\beta$={be:.3f}")
ax[1].semilogy(Dd,np.exp(-al*Dd),":",color="#777",label=r"$e^{-\alpha D}$")
ax[1].set(xlabel="dose $D$ (Gy)",ylabel="surviving fraction $S$",ylim=(1e-4,1.1),xlim=(0,10)); ax[1].legend(fontsize=7,loc="lower left")
lab(ax[0],"(a)",0.9,0.1); lab(ax[1],"(b)",0.9)
fig.tight_layout(); fig.savefig("fig2.pdf"); fig.savefig("fig2.png",dpi=150)

# Fig 3 ---------------------------------------------------------------------
r=json.load(open("doserate_disc.json")); D5=np.array([1,2,4,6,8.])
fit=lambda y: np.sum((-np.log(np.array(y))-al*D5)*D5**2)/np.sum(D5**4); ba=fit(r["acute_exact"])
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); Dc=np.linspace(0,8.5,200)
ax[0].semilogy(Dc,survival_disc(Dc,lam,mu,s2,kap0,nc),color="#444",label="acute (exact)")
mk=["o","s","^","D"]; ratio=[]
for i,k in enumerate(["60.0","6.0","1.0","0.3"]):
    ax[0].semilogy(D5,r[k],mk[i],ms=4.5,mfc="white",mec=C[i],mew=1.2,label=fr"$\dot D$ = {float(k):g} Gy h$^{{-1}}$"); ratio.append(fit(r[k])/ba)
ax[0].semilogy(Dc,np.exp(-al*Dc),":",color="#777",label=r"LDR limit $e^{-\alpha D}$")
ax[0].set(xlabel="dose $D$ (Gy)",ylabel="surviving fraction $S$",ylim=(5e-3,1.1)); ax[0].legend(fontsize=6.5,loc="lower left")
R=np.array([60,6,1,.3]); G=lambda y:2*(y-1+np.exp(-y))/y**2; Rc=np.logspace(-1.2,2.2,200)
ax[1].semilogx(R,ratio,"o",ms=5,mfc="white",mec=C[0],mew=1.3,label="stochastic simulation")
ax[1].semilogx(Rc,G(mu*4/Rc),"--",color=C[1],label=r"$G(\mu T)$, $\mu$ = 0.5 h$^{-1}$")
ax[1].semilogx(Rc,G(1.64*4/Rc),"-",color=C[2],label=r"$G(\mu_{\rm eff}T)$, $\mu_{\rm eff}$ = 1.6 h$^{-1}$")
ax[1].set(xlabel=r"dose rate $\dot D$ (Gy h$^{-1}$)",ylabel=r"$\beta_{\rm eff}(\dot D)/\beta_{\rm eff}(\infty)$",ylim=(0,1.05)); ax[1].legend(fontsize=7,loc="upper left")
lab(ax[0],"(a)",0.9); lab(ax[1],"(b)",0.9,0.12)
fig.tight_layout(); fig.savefig("fig3.pdf"); fig.savefig("fig3.png",dpi=150)
json.dump({"ratio":ratio},open("fig3_vals_disc.json","w")); print("ratios",np.round(ratio,3))

# Fig 4 ---------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); F=np.linspace(0.5,6,300)
L={"CHO":(0.151,0.037),"A549":(0.191,0.026),"HCT116":(0.388,0.047),"HeLa":(0.300,0.040),"PC3":(0.195,0.057)}
ab=[a/b for a,b in L.values()]
for i,ld in enumerate([0.5,1.0,2.0]):
    ax[0].plot(F,(2*F-1)/ld,color=C[i],ls=["-","--","-."][i],label=fr"$\lambda_d$ = {ld:g} Gy$^{{-1}}$")
ax[0].plot(F,2/np.expm1(1/F),":",color="#555",lw=1.2,label=r"diffusion limit, $\lambda_d$ = 1")
ax[0].axhspan(min(ab),max(ab),color="#dddddd",zorder=0); ax[0].text(0.55,max(ab)+0.3,r"range of $\alpha/\beta$, 5 cell lines",fontsize=7,color=INK)
ax[0].set(xlabel=r"$F_r=\sigma^2/2\mu$",ylabel=r"$\alpha/\beta$ (Gy), $\kappa=0$",ylim=(0,16),xlim=(0.5,6)); ax[0].legend(fontsize=7,loc="upper left")
pr=json.load(open("profile.json"))
for i,nm in enumerate(["CHO","HCT116"]):
    ax[1].semilogx(pr["F"],pr[nm],["o-","s--"][i],color=C[i],mfc="white",mew=1.2,ms=4,label=nm)
ax[1].axhline(0.05,color="#999",ls=":",lw=1); ax[1].text(3,0.052,r"5% error in $S$",fontsize=7,color=INK)
ax[1].set(xlabel=r"$F_r$ (fixed)",ylabel=r"best-fit rms residual in $\ln S$",ylim=(0,0.06)); ax[1].legend(fontsize=7,loc="center left")
lab(ax[0],"(a)",0.92); lab(ax[1],"(b)",0.92)
fig.tight_layout(); fig.savefig("fig4.pdf"); fig.savefig("fig4.png",dpi=150)
