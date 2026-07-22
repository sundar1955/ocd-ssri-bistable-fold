"""Bifurcation diagram of the reduced 1-D G_CS plasticity dynamics under SSRI occupancy.
(a) For a representative patient (Y-BOCS0=20): equilibrium corticostriatal weight G_CS* vs SERT occupancy u,
    with the stable OCD branch, the unstable saddle branch, and the stable health branch continued explicitly,
    and the saddle-node fold marked where the OCD and saddle branches collide and annihilate.
(b) The fold occupancy u_fold vs baseline severity Y-BOCS0 (the severity-gating curve), with the maximal-SSRI
    occupancy line -- shows which baselines can be folded by an SSRI alone.
This is the continuation/branch object whose bifurcation is analyzed; the phase-line figure (Fig. 4) is its
vertical-slice complement."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
_uc={}
def eC_ss(u):
    if u<=1e-6: return eC_h
    if u in _uc: return _uc[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1]); _uc[u]=v; return v
def cdrug(u): return 1.0+BB*(eC_ss(u)-eC_h)
_GG=np.linspace(1e-4,2.9,4000)
def RHS(G,al,u): return al*B.Sshape(G,THM,PS)-cdrug(u)*G
def fixed_points(al,u):
    f=al*np.array([B.Sshape(g,THM,PS) for g in _GG])-cdrug(u)*_GG; out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=_GG[i]-f[i]*(_GG[i+1]-_GG[i])/(f[i+1]-f[i])
        st=(RHS(g+1e-3,al,u)-RHS(g-1e-3,al,u))<0; out.append((g,st))
    # health at G~0 is always a stable FP (RHS(0)=0, baseline-referenced source)
    if not any(g<0.02 for g,_ in out): out=[(0.0,True)]+out
    return out
def u_fold(al):
    lo,hi=0.0,0.999
    # OCD attractor exists at u=lo; find u where highest non-health stable FP disappears
    def has_ocd(u):
        return any((g>0.25 and st) for g,st in fixed_points(al,u))
    if not has_ocd(lo): return 0.0
    if has_ocd(hi): return np.nan
    for _ in range(40):
        m=0.5*(lo+hi)
        if has_ocd(m): lo=m
        else: hi=m
    return 0.5*(lo+hi)

# ---------- panel (a): branches for Y0=20 ----------
Y0=20.0; al,G0,ok=GP.alpha_for(Y0,THM)
us=np.linspace(0,0.985,220)
ocd=[]; sad=[]; hea=[]
for u in us:
    fps=fixed_points(al,u); stab=sorted([g for g,s in fps if s]); uns=sorted([g for g,s in fps if not s])
    hi=[g for g in stab if g>0.25]; ocd.append(hi[-1] if hi else np.nan)
    sad.append(uns[-1] if uns else np.nan)
    lo=[g for g in stab if g<0.25]; hea.append(lo[0] if lo else 0.0)
ocd=np.array(ocd); sad=np.array(sad); hea=np.array(hea)
uf=u_fold(al)

plt.rcParams.update({'font.size':12})
fig,(axA,axB)=plt.subplots(1,2,figsize=(13.2,5.2))
def Yof(G): return M.Yread(G)
# G-axis branches
axA.plot(us,ocd,color='#6a3d9a',lw=3.0,zorder=4,label='OCD attractor (stable)')
axA.plot(us,sad,color='#c0392b',lw=2.6,ls='--',zorder=4,label='saddle (unstable)')
axA.plot(us,hea,color='#1f7a1f',lw=3.0,zorder=4,label='healthy state (stable)')
# fold marker
if np.isfinite(uf):
    gf=np.interp(uf,us,np.nan_to_num(ocd,nan=np.interp(uf,us,sad)))
    # value of colliding branches at fold ~ where ocd and sad meet
    gfold=np.nanmin([np.interp(uf,us,ocd), np.interp(uf,us,sad)]) if np.isfinite(np.interp(uf,us,ocd)) else np.interp(uf,us,sad)
    # robust: take saddle value near uf
    gfold=np.interp(uf,us[np.isfinite(sad)],sad[np.isfinite(sad)])
    axA.plot(uf,gfold,'D',ms=12,color='#c0392b',mec='k',mew=1.2,zorder=6)
    axA.annotate(f'saddle-node fold\n$u_{{\\rm fold}}={uf:.2f}$',xy=(uf,gfold),xytext=(uf-0.30,gfold+0.55),
                 fontsize=11,ha='left',arrowprops=dict(arrowstyle='-|>',color='0.4',lw=1.3))
    axA.axvspan(uf,1.0,color='#1f7a1f',alpha=0.06)
    axA.text(min(uf+0.02,0.9),0.10,'past fold:\nhealth only',fontsize=9.5,color='#1f7a1f',va='bottom')
axA.set_xlabel('SERT occupancy  $u=D/(D+D_{50})$',fontsize=13)
axA.set_ylabel('equilibrium weight  $G_{CS}^{*}$',fontsize=13)
axA.set_title('(a)  Bifurcation diagram, Y-BOCS$_0=20$',fontsize=13)
axA.set_xlim(0,1.0); axA.set_ylim(-0.05,max(2.0,np.nanmax(ocd)*1.08)); axA.grid(alpha=.22)
axA.legend(fontsize=10.5,loc='lower center',framealpha=0.95)

# ---------- panel (b): fold occupancy vs baseline severity ----------
Y0g=np.arange(16,29.1,0.5); ufs=[]
for y in Y0g:
    a,_,_=GP.alpha_for(y,THM); ufs.append(u_fold(a))
ufs=np.array(ufs)
axB.plot(Y0g,ufs,'-o',color='#6a3d9a',lw=2.6,ms=4,zorder=5)
axB.axhline(0.95,ls=':',color='#c0392b',lw=1.4); axB.text(16.2,0.955,'max SSRI ($u\\approx0.95$)',fontsize=10,color='#c0392b',va='bottom')
# shade unreachable (u_fold>0.95 or nan)
reach=np.where(np.isfinite(ufs)&(ufs<=0.95),ufs,np.nan)
axB.fill_between(Y0g,0.95,1.0,color='0.85',alpha=0.5)
# mark last reachable baseline
finite=Y0g[np.isfinite(ufs)&(ufs<=0.95)]
if len(finite): axB.axvline(finite[-1],ls='--',color='0.5',lw=1.0)
axB.set_xlabel('baseline Y-BOCS$_0$',fontsize=13); axB.set_ylabel('fold occupancy  $u_{\\rm fold}$',fontsize=13)
axB.set_title('(b)  Which baselines an SSRI can fold',fontsize=13)
axB.set_xlim(16,29); axB.set_ylim(0.3,1.0); axB.grid(alpha=.22)
plt.tight_layout(); plt.savefig('fig_bifurcation.pdf'); plt.savefig('fig_bifurcation.png',dpi=140)
print(f"Y0=20 u_fold={uf:.3f}; branches OCD/saddle/health continued over u in [0,0.985]")
print("u_fold vs Y0:", [(float(y),round(float(v),3)) for y,v in zip(Y0g,ufs) if y in (16,18,20,22,24,26,28)])
