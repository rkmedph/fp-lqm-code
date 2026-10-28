import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C=["#2a78d6","#eb6834","#1baf7a","#eda100"]; INK="#333333"
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":"#888","axes.labelcolor":INK,"xtick.color":"#555","ytick.color":"#555",
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"font.family":"DejaVu Sans"})
r=json.load(open("recovery2.json"))
keys=[("base|K3|n1000|Mfixed","3\n$10^3$"),("base|K3|n10000|Mfixed","3\n$10^4$"),("base|K8|n1000|Mfixed","8\n$10^3$"),
      ("base|K8|n10000|Mfixed","8\n$10^4$"),("base|K16|n1000|Mfixed","16\n$10^3$"),("base|K16|n10000|Mfixed","16\n$10^4$"),
      ("near|K8|n10000|Mfixed","8\n$10^4$\n$\\kappa\\approx\\mu$")]
fig,ax=plt.subplots(1,3,figsize=(7.4,3.0),gridspec_kw={"width_ratios":[1.25,1.25,0.8]})
for j,(col,lab,tr) in enumerate([(4,r"estimated $\alpha/\beta$ (Gy)",None),(2,r"estimated $\kappa$ (h$^{-1}$)",None)]):
    for i,(k,l) in enumerate(keys):
        e=np.array(r[k]["est"])[:,col]; t=r[k]["truth"][col]; q=np.percentile(e,[10,25,50,75,90])
        c=C[1] if k.startswith("near") else C[0]
        ax[j].plot([i,i],[q[0],q[4]],color=c,lw=1); ax[j].add_patch(plt.Rectangle((i-0.25,q[1]),0.5,q[3]-q[1],fc="white",ec=c,lw=1.3))
        ax[j].plot([i-0.25,i+0.25],[q[2],q[2]],color=c,lw=2); ax[j].plot([i-0.35,i+0.35],[t,t],":",color=INK,lw=1)
    ax[j].set_xticks(range(len(keys))); ax[j].set_xticklabels([l for _,l in keys],fontsize=7); ax[j].set_ylabel(lab)
    ax[j].set_xlabel("time points (top) / cells per point (bottom)",fontsize=7)
ax[0].set_ylim(0,6); ax[1].set_ylim(-0.02,0.6)
for i,(k,l) in enumerate([("base|K8|n1000|Mfree","10$^3$"),("base|K8|n10000|Mfree","10$^4$")]):
    e=np.array(r[k]["est"])[:,3]; ax[2].hist(np.log10(e),bins=np.linspace(-0.8,2.4,33),histtype="step",lw=1.5,color=C[i],ls=["-","--"][i],label=f"{l} cells/point")
ax[2].axvline(np.log10(4),color=INK,ls=":",lw=1); ax[2].set_xlabel(r"$\log_{10}\hat M$ (true $M$=4)"); ax[2].set_ylabel("replicates"); ax[2].legend(fontsize=6.5,loc="center left",bbox_to_anchor=(0.0,0.62))
for a_,t in zip(ax,["(a)","(b)","(c)"]): a_.text(0.02,0.98,t,transform=a_.transAxes,va="top",fontweight="bold")
fig.tight_layout(); fig.savefig("fig5.pdf"); fig.savefig("fig5.png",dpi=150)
for k,_ in keys+[("base|K8|n1000|Mfree",""),("base|K8|n10000|Mfree","")]:
    e=np.array(r[k]["est"]); print(k,"ab",np.round(np.percentile(e[:,4],[10,50,90]),2),"k",np.round(np.percentile(e[:,2],[10,50,90]),3),"M",np.round(np.percentile(e[:,3],[10,50,90]),1),"frac M at bounds",np.mean((e[:,3]<0.21)|(e[:,3]>199)))
