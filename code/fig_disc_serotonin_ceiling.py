"""Discussion figure: the serotonin ceiling (drug-agnostic; e_C free parameter)."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json'))
THM=9.15;BB=bf['b'];PS=GP.PHI_SAT;eC_h=M.eC_h
M.GAMMA_OBS=2.0;M.b_5HT1B=BB;M.setup(bf['kappa_des'],2.0);GP.KAPPA_DES=bf['kappa_des']
Gs=np.linspace(1e-3,3.0,1600)
def RHS(G,al,ec): return al*B.Sshape(G,THM,PS)-(1.0+BB*(ec-eC_h))*G
def ocd_fp(al,ec):
    f=al*np.array([B.Sshape(g,THM,PS) for g in Gs])-(1.0+BB*(ec-eC_h))*Gs; best=None
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        if g>0.25 and (RHS(g+1e-3,al,ec)-RHS(g-1e-3,al,ec))<0: best=g
    return best
_S={}
def Ssh(g):
    if g not in _S: _S[g]=B.Sshape(g,THM,PS)
    return _S[g]
def foldec(Y0):  # bisection: has-attractor is monotone-decreasing in eC
    al,_,_=GP.alpha_for(Y0,THM); lo,hi=eC_h,240.
    if ocd_fp(al,lo) is None: return lo,al
    for _ in range(34):
        mid=0.5*(lo+hi)
        if ocd_fp(al,mid) is None: hi=mid
        else: lo=mid
    return hi,al
def foldalpha(Y0):  # min LTP-gain reduction fraction (source route, memantine) that folds; monotone in m
    al,_,_=GP.alpha_for(Y0,THM); lo,hi=0.0,0.98
    if ocd_fp(al*(1-lo),eC_h) is None: return lo
    for _ in range(30):
        mid=0.5*(lo+hi)
        if ocd_fp(al*(1-mid),eC_h) is None: hi=mid
        else: lo=mid
    return hi
fig,(axA,axB)=plt.subplots(1,2,figsize=(12.5,4.6))
Y0=28; ecf,al=foldec(Y0); Gg=np.linspace(0,2.7,600)
ecs=[eC_h,8,14,20,26,ecf]; cmap=cm.viridis(np.linspace(0.05,0.8,len(ecs)))
axA.axhline(0,color='0.6',lw=0.8,ls=':')
for ec,c in zip(ecs,cmap):
    isf=abs(ec-ecf)<1e-6
    axA.plot(Gg,[RHS(g,al,ec) for g in Gg],color=('#c0392b' if isf else c),lw=(3.0 if isf else 2.0),
             zorder=(6 if isf else 3),label=f'{ec/eC_h:.0f}×'+('  (fold)' if isf else ''))
    g=ocd_fp(al,ec)
    if g: axA.plot(g,0,'o',ms=8,mfc='#6a3d9a',mec='#3a2060',zorder=7)
axA.plot(0,0,'o',ms=8,mfc='k',mec='k',zorder=7)
axA.set_xlim(-0.03,2.7); axA.set_ylim(-0.22,0.12); axA.grid(alpha=.2)
axA.set_xlabel('$G_{CS}$  (corticostriatal weight)'); axA.set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$')
axA.set_title(f'(a)  Y-BOCS$_0$=28: attractor annihilates only at $e_C$={ecf/eC_h:.0f}× baseline',fontsize=10)
axA.legend(title='$e_C/e_C^{\\,h}$',fontsize=8,loc='upper right',framealpha=0.95)
Y0g=np.arange(18,34.6,1.0); fe=np.array([foldec(y)[0]/eC_h for y in Y0g]); fa=np.array([100*foldalpha(y) for y in Y0g])
axB.axhspan(10,64,color='#f39c12',alpha=0.11)
axB.axhline(10,color='#f39c12',lw=1,ls='--')
axB.text(18.2,10.8,'supraphysiological serotonin\n(rising toxicity risk; illustrative)',fontsize=7.0,color='#b9770e',va='bottom')
lS,=axB.plot(Y0g,fe,'-',color='#1f3a93',lw=2.6,zorder=5)
axB.axhline(15.6/eC_h,color='0.4',lw=1.1,ls=':'); axB.text(34.4,15.6/eC_h+0.5,'max SSRI',fontsize=7.3,color='0.35',ha='right',va='bottom')
yc=np.interp(10,fe,Y0g); axB.plot(yc,10,'o',ms=7,mfc='#c0392b',mec='k',zorder=6)
axB2=axB.twinx(); lM,=axB2.plot(Y0g,fa,'-',color='#c0392b',lw=2.6,zorder=5)
axB2.set_ylim(0,100); axB2.set_ylabel('memantine: LTP-gain reduction to fold (%)',color='#c0392b'); axB2.tick_params(axis='y',labelcolor='#c0392b')
axB.set_xlim(18,34); axB.set_ylim(0,64); axB.grid(alpha=.2)
axB.set_xlabel('baseline Y-BOCS$_0$'); axB.set_ylabel('serotonin: $e_C$ to fold ($\\times$ baseline)',color='#1f3a93'); axB.tick_params(axis='y',labelcolor='#1f3a93')
axB.legend([lS,lM],['serotonin (sink) — toxicity-capped','memantine (source) — bounded'],fontsize=7.8,loc='upper center',framealpha=0.95)
axB.set_title('(b)  source vs sink: memantine reaches attractors serotonin cannot',fontsize=10)
plt.tight_layout(); plt.savefig('fig_disc_serotonin_ceiling.pdf'); plt.savefig('fig_disc_serotonin_ceiling.png',dpi=140)
print(f"Y0=28 fold eC={ecf:.1f} ({ecf/eC_h:.1f}x); 10x-crossing at Y0={yc:.1f}")
