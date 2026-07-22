"""Memantine proof-of-concept (deliverable): the two fold routes for the same OCD attractor vs severity.
SINK (serotonin, e_C-multiple) enters the serotonin-syndrome range for Y0>=27; SOURCE (memantine, %LTP-gain
reduction alpha-down) stays a bounded, serotonin-independent intervention. Source is accessible where sink is toxic."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json')); THM=9.15;BB=bf['b'];PS=GP.PHI_SAT;eC_h=M.eC_h
M.GAMMA_OBS=2.0;M.b_5HT1B=BB;M.setup(bf['kappa_des'],2.0);GP.KAPPA_DES=bf['kappa_des']
Gs=np.linspace(1e-3,3,4000)
def hasOCD(al,ec):
    f=al*np.array([B.Sshape(g,THM,PS) for g in Gs])-(1.0+BB*(ec-eC_h))*Gs
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        if g>0.25 and (al*B.Sshape(g+1e-3,THM,PS)-(1.0+BB*(ec-eC_h))*(g+1e-3))<(al*B.Sshape(g-1e-3,THM,PS)-(1.0+BB*(ec-eC_h))*(g-1e-3)): return True
    return False
Y0g=np.arange(22,34.6,0.5); fa=[]; fe=[]
for Y0 in Y0g:
    al,_,_=GP.alpha_for(Y0,THM)
    fa.append(100*next((m for m in np.arange(0,1.0,0.01) if not hasOCD(al*(1-m),eC_h)),np.nan))
    fe.append(next((ec for ec in np.arange(eC_h,320,0.3) if not hasOCD(al,ec)),np.nan)/eC_h)
fa=np.array(fa); fe=np.array(fe)
fig,axL=plt.subplots(figsize=(7.2,5.0)); axR=axL.twinx()
# sink toxicity bands on the eC (right) axis
axR.axhspan(10,60,color='#f39c12',alpha=0.10)
axR.axhline(10,color='#f39c12',lw=1,ls='--')
lR,=axR.plot(Y0g,fe,color='#1f3a93',lw=2.8,label='sink route: serotonin $e_C$ needed ($\\times$ baseline)')
lL,=axL.plot(Y0g,fa,color='#c0392b',lw=2.8,label='source route: memantine LTP reduction $\\alpha\\!\\downarrow$ (%)')
axL.axhspan(0,75,color='#2ecc71',alpha=0.06)
axL.set_xlabel('presenting Y-BOCS$_0$'); axL.set_ylabel('memantine LTP-gain reduction to fold  (%)',color='#c0392b')
axR.set_ylabel('serotonin $e_C$ to fold  ($\\times$ baseline)',color='#1f3a93')
axL.tick_params(axis='y',labelcolor='#c0392b'); axR.tick_params(axis='y',labelcolor='#1f3a93')
axL.set_ylim(0,100); axR.set_ylim(0,60); axL.set_xlim(22,34); axL.grid(alpha=.2)
axR.text(22.2,10.8,'supraphysiological serotonin\n(rising toxicity risk; illustrative)',fontsize=7.0,color='#b9770e',va='bottom')
yc=np.interp(10,fe,Y0g); axR.plot(yc,10,'o',ms=7,mfc='#1f3a93',mec='k',zorder=6)
axR.annotate(f'sink route enters\ntoxicity at Y$_0\\approx${yc:.0f}',xy=(yc,10),xytext=(yc-4.6,20),fontsize=8,color='#1f3a93',arrowprops=dict(arrowstyle='-|>',color='#1f3a93',lw=1))
axL.legend([lL,lR],[t.get_label() for t in (lL,lR)],fontsize=8.5,loc='upper center')
axL.set_title('Source vs sink: memantine reaches severe attractors serotonin cannot',fontsize=10.5)
plt.tight_layout(); plt.savefig('fig_source_vs_sink.pdf'); plt.savefig('fig_source_vs_sink.png',dpi=140)
print("fold at Y0=28: alpha_down=%.0f%%, eC=%.1fx ; sink enters 10x at Y0=%.1f"%(fa[np.argmin(abs(Y0g-28))],fe[np.argmin(abs(Y0g-28))],yc))
