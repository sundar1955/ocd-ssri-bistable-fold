"""Figure 5 (R3): SSRI-monotherapy drug-attributable improvement trajectories, by baseline severity.
4 panels (Y0=20,24,28,32); u=0.8/0.9. Solid = drug-attributable (Y0-Y_drug); dashed = total incl.
illustrative placebo eps(t)=4(1-e^{-t/5.5}), floored so total Y>=0. Dotted = 25%/35% response lines."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gcs_bistable_calibrate as C, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json'))
THM=9.15;HMAX=2.0;BB=bf['b'];KD=bf['kappa_des'];TG=bf['tau_G'];PS=GP.PHI_SAT;eC_h=M.eC_h
GP.KAPPA_DES=KD;C.KAPPA_DES=KD;M.GAMMA_OBS=2.0;M.b_5HT1B=BB;M.setup(KD,HMAX)
DT=0.1; TMAX=52.
def traj(Y0,u):
    al,G0,ok=GP.alpha_for(Y0,THM)
    tg=np.arange(0,TMAX+DT/2,DT); ec=C.ec_series(lambda t:(u*10./(1-u)),10.,tg)
    G=G0; Y=np.empty(len(tg))
    for i in range(len(tg)): Y[i]=M.Yread(G); G+=DT*(al*B.Sshape(G,THM,PS)-(1.0+BB*(ec[i]-eC_h))*G)/TG; G=min(max(G,0.02),3.4)
    return tg,Y,(Y0-Y)
def eps(t): return 4.0*(1-np.exp(-t/5.5))
UC={0.8:'#0072B2',0.9:'#D55E00'}
fig,axs=plt.subplots(1,4,figsize=(16,4.1),sharey=True)
for k,(ax,Y0) in enumerate(zip(axs,[20,24,28,32])):
    for u in [0.8,0.9]:
        tg,Y,dY=traj(Y0,u)
        Ytot=np.maximum(0.0,Y-eps(tg)); dYtot=Y0-Ytot
        ax.plot(tg,dY,'-',color=UC[u],lw=2.4,zorder=5)
        ax.plot(tg,dYtot,'--',color=UC[u],lw=1.6,alpha=0.9,zorder=4)
    for frac in (0.25,0.35):
        ax.axhline(frac*Y0,color='0.5',lw=1.0,ls=':',zorder=2)
        ax.text(TMAX*0.985,frac*Y0+0.15,f'{int(frac*100)}%',fontsize=7.5,color='0.4',va='bottom',ha='right')
    for tv in (12,24): ax.axvline(tv,color='0.88',lw=0.8,zorder=1)
    ax.text(0.035,0.955,f'({chr(97+k)})  Y-BOCS$_0$={Y0}',transform=ax.transAxes,fontsize=11,va='top',ha='left',fontweight='bold')
    ax.set_xlabel('week'); ax.set_xlim(0,TMAX); ax.grid(alpha=.18)
axs[0].set_ylabel('drug-attributable  $Y_0-Y(t)$  (Y-BOCS pts)'); axs[0].set_ylim(-0.4,21)
lh=[Line2D([],[],color=UC[0.8],lw=2.4,label='$u=0.8$'),Line2D([],[],color=UC[0.9],lw=2.4,label='$u=0.9$'),
    Line2D([],[],color='0.35',lw=2.2,ls='-',label='drug only'),Line2D([],[],color='0.35',lw=1.6,ls='--',label='+ placebo')]
axs[0].legend(handles=lh,fontsize=8,loc='upper left',bbox_to_anchor=(0.02,0.88),framealpha=0.95)
plt.tight_layout(); plt.savefig('fig5_trajectories.pdf'); plt.savefig('fig5_trajectories.png',dpi=140)
print("wrote fig5_trajectories.pdf")
