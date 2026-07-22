"""Discussion: three complementary mechanisms fold a Y-BOCS0=32 attractor that none folds alone.
SSRI (sink, via eC(u)) + AP (circuit, via aAP -> phi_d1 down) + memantine (source, via LTP-gain alpha down)."""
import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # px_ap_pk_model.py is bundled in this repo
import px_ap_pk_model as PK
from var_circuit_validation import idx, S, M, ext
import build_fit_gcs_slow as Mod, gamma_profile_calib as GP, gcs_bistable_population as B
C=GP.C
bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; PS=GP.PHI_SAT; eC_h=Mod.eC_h; phih=Mod.phid1_h  # canonical b=0.0119 (best_fit.json), matches fig4/fig5
Mod.GAMMA_OBS=2.0; Mod.b_5HT1B=BB; Mod.setup(bf['kappa_des'],HMAX); GP.KAPPA_DES=bf['kappa_des']
ie=idx['e']; id1=idx['d1']; id2=idx['d2']; RHO=0.7
def M_g(g): Mm=M.copy(); Mm[id1,ie]+=g; Mm[id2,ie]+=RHO*g; return Mm
# AP circuit grid: phe, phd1 vs (G, aAP)
Gg=np.linspace(0.02,3.2,60); aAPg=np.linspace(0,20,26); PHE=np.zeros((60,26)); PHD1=np.zeros((60,26))
for i,G in enumerate(Gg):
    seed=np.array([12,12,40,20,60,30,30,15,28.]); Mm=M_g(G)
    for j,a in enumerate(aAPg):
        def res(phi): V=(Mm@phi+ext).copy(); V[id2]+=a; return phi-S(V)
        seed=fsolve(res,seed,xtol=1e-10); PHE[i,j]=seed[ie]; PHD1[i,j]=seed[id1]
def pe_pd(G,a):
    i=np.searchsorted(Gg,G); i=min(max(i,1),len(Gg)-1); w=(G-Gg[i-1])/(Gg[i]-Gg[i-1])
    pe=(1-w)*np.interp(a,aAPg,PHE[i-1])+w*np.interp(a,aAPg,PHE[i])
    pd=(1-w)*np.interp(a,aAPg,PHD1[i-1])+w*np.interp(a,aAPg,PHD1[i]); return pe,pd
def wovr(pd): return np.clip((pd-phih)/(PS-phih),0,1)   # memantine state-dependent weight
def eC_ss(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); return float(C.ec_series(lambda t:D,D50,tg)[-1])
def RHS(G,al,u,aAP,m):
    pe,pd=pe_pd(G,aAP); src=pe*(pd-phih)*(pd-THM)*(PS-pd)
    return al*(1-m)*src-(1.0+BB*(eC_ss(u)-eC_h))*G   # memantine = uniform LTP-gain reduction (matches SI source-vs-sink)
def folds(al,u,aAP,m):  # True if no diseased attractor (annihilated)
    f=np.array([RHS(g,al,u,aAP,m) for g in Gg])
    for k in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gg[k]-f[k]*(Gg[k+1]-Gg[k])/(f[k+1]-f[k])
        if g>0.25 and RHS(g+1e-2,al,u,aAP,m)<RHS(g-1e-2,al,u,aAP,m): return False
    return True
Y0=32; al,G0,_=GP.alpha_for(Y0,THM)
uS=0.8; aAP_D50=8.0*0.50  # AP at D50: 50% occ -> eff.block 0.50 -> aAP=C*eff=8*0.5=4.0
print(f"Y-BOCS0={Y0}: alpha={al:.4f}, baseline attractor G0={G0:.2f}, eC(u=0.8)={eC_ss(uS):.1f} nM")
print(f"SSRI u=0.8 alone folds? {folds(al,uS,0,0)}")
print(f"AP D50 (aAP={aAP_D50}) alone folds? {folds(al,0,aAP_D50,0)}")
def m_to_fold(u,aAP):
    lo,hi=0.0,0.99
    if folds(al,u,aAP,lo): return 0.0
    if not folds(al,u,aAP,hi): return np.nan
    for _ in range(28):
        mid=.5*(lo+hi)
        if folds(al,u,aAP,mid): hi=mid
        else: lo=mid
    return hi
mM=m_to_fold(0,0); mSA=m_to_fold(uS,aAP_D50); mSSRI=m_to_fold(uS,0)
print(f"memantine WITH SSRI(0.8), no AP, to fold: {100*mSSRI:.0f}%")
print(f"memantine ALONE to fold Y0=32: {100*mM:.0f}% LTP-gain reduction")
print(f"memantine WITH SSRI(0.8)+AP(D50) to fold: {100*mSA:.0f}%  (SSRI+AP residual)")
print(f"SSRI+AP+memantine(30%) folds? {folds(al,uS,aAP_D50,0.30)}")
# split into two figures: MAIN = big single-panel cumulative tri-therapy; SI = monotherapies
mF=0.46  # memantine level (above triple threshold 41%, below mono 64%)
Gp=np.linspace(0,2.9,600)
# ---- standardized fixed-point marker convention (matches Figs 2b, 4) ----
PURPLE='#6a3d9a'; SADDLE_EC='#c0392b'  # OCD attractor = purple filled; saddle = open red edge; health = black filled
def mark_fp(ax,u,a,m):
    f=np.array([RHS(g,al,u,a,m) for g in Gg])
    for k in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gg[k]-f[k]*(Gg[k+1]-Gg[k])/(f[k+1]-f[k])
        if g<0.05: continue
        stable=RHS(g+1e-2,al,u,a,m)<RHS(g-1e-2,al,u,a,m)
        if stable and g>0.25:                       # surviving OCD attractor
            ax.plot(g,0,'o',ms=9,mfc=PURPLE,mec='#3a2060',mew=1.1,zorder=9)
        elif not stable:                            # saddle (commitment threshold)
            ax.plot(g,0,'o',ms=9,mfc='white',mec=SADDLE_EC,mew=1.6,zorder=9)
def draw(ax,combos,hi,lfs,ylim=(-0.30,0.18),curveleg=False,clloc='upper left'):
    ax.axhline(0,color='0.6',lw=0.8,ls=':')
    curvehandles=[]
    for lab,u,a,m,c,ls in combos:
        lw=3.4 if (hi and hi in lab) else 2.2; z=6 if (hi and hi in lab) else 3
        h,=ax.plot(Gp,[RHS(g,al,u,a,m) for g in Gp],color=c,lw=lw,ls=ls,label=lab,zorder=z)
        curvehandles.append(h)
        mark_fp(ax,u,a,m)
    ax.plot(0,0,'o',ms=8.5,mfc='k',mec='k',zorder=8)          # health (stable)
    ax.set_xlim(-0.03,2.9); ax.set_ylim(*ylim); ax.grid(alpha=.2)
    ax.set_xlabel('$G_{CS}$  (corticostriatal weight)'); ax.set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$')
    # on-figure fixed-point key (marker convention), separate from the curve legend
    fpkey=[Line2D([],[],ls='',marker='o',mfc='k',mec='k',ms=8,label='health (stable)'),
           Line2D([],[],ls='',marker='o',mfc='white',mec=SADDLE_EC,mew=1.5,ms=8,label='saddle (threshold)'),
           Line2D([],[],ls='',marker='o',mfc=PURPLE,mec='#3a2060',ms=8,label='OCD attractor')]
    leg1=ax.legend(handles=fpkey,fontsize=8.2,loc='upper right',framealpha=0.95,title='fixed points')
    leg1.get_title().set_fontsize(8.2); ax.add_artist(leg1)
    # on-figure curve legend (upper-left gap opened by the expanded ordinate)
    if curveleg:
        leg2=ax.legend(handles=curvehandles,fontsize=7.2,loc=clloc,framealpha=0.95,
                       title='treatment arm',borderpad=0.5,labelspacing=0.32,handlelength=1.9)
        leg2.get_title().set_fontsize(7.2); ax.add_artist(leg2)
# palette harmonized to manuscript semantics (Okabe-Ito colorblind-safe): SSRI blue, AP green, memantine vermillion, triple reddish-purple
SSRI_C='#0072B2'; AP_C='#009E73'; MEM_C='#D55E00'; TRIP_C='#CC79A7'; REF_C='#555555'; MIX_C='#E69F00'
mono=[('untreated',0,0,0,REF_C,'-'),
      (f'SSRI only ($u$=0.8)',uS,0,0,SSRI_C,'-'),
      (f'AP only ($D_{{50}}$)',0,aAP_D50,0,AP_C,'-'),
      (f'memantine only ({100*mF:.0f}%)',0,0,mF,MEM_C,'-')]
cum =[('no drug',0,0,0,REF_C,'-'),
      (f'SSRI ($u$=0.8)',uS,0,0,SSRI_C,'-'),
      (f'SSRI + AP ($D_{{50}}$)',uS,aAP_D50,0,AP_C,'-'),
      (f'SSRI + mem {100*mF:.0f}% (no AP)',uS,0,mF,MIX_C,'--'),
      (f'SSRI + mem {100*mSSRI:.0f}% (no AP)',uS,0,mSSRI,MIX_C,':'),
      (f'SSRI + AP + mem {100*mF:.0f}%',uS,aAP_D50,mF,TRIP_C,'-')]
# MAIN figure: cumulative tri-therapy, large single panel
figM,axM=plt.subplots(figsize=(8.6,6.0)); draw(axM,cum,'SSRI + AP + mem',9.8,ylim=(-0.4,0.4),curveleg=True,clloc='upper left')
# annotate the folded (curative) triple-therapy curve: attractor gone -> drives to remission
axM.annotate('no OCD attractor:\nstate flows to health (remission)',xy=(1.35,RHS(1.35,al,uS,aAP_D50,mF)),
             xytext=(1.55,0.10),fontsize=9,color=TRIP_C,ha='left',va='center',fontweight='bold',
             arrowprops=dict(arrowstyle='-|>',color=TRIP_C,lw=1.4,mutation_scale=12))
axM.set_title('Refractory patient ($Y_{\\mathrm{model},0}$=32): stacking mechanisms annihilates the attractor',fontsize=12)
figM.tight_layout(); figM.savefig('fig_combo_fold.pdf'); figM.savefig('fig_combo_fold.png',dpi=140)
# SI figure: monotherapies each sub-threshold
figS,axS=plt.subplots(figsize=(7.2,5.0)); draw(axS,mono,None,9.0)
axS.set_title('Each mechanism alone leaves the $Y_{\\mathrm{model},0}$=32 attractor intact',fontsize=11)
figS.tight_layout(); figS.savefig('fig_combo_mono.pdf'); figS.savefig('fig_combo_mono.png',dpi=140)
print(f"saved fig_combo_fold.pdf (memantine level {100*mF:.0f}%)")
