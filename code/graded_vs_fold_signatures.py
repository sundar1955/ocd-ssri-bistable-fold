"""
STAGE 3 (CORE / WP1 / R5 crux): the discriminating observables.
FOLD (canonical bistable) vs GRADED (IDR null) -- both matched on the trial mean (chi2 6.27 vs 6.50, each with its own drug coupling b; difference within noise).
Compute the signatures a graded model CANNOT produce:
  (A) per-patient equilibrium Y vs steady occupancy u   -> DISCONTINUITY (saddle-node jump) vs smooth decline
  (B) hysteresis / durable off-drug remission            -> one-way transition (up-fold, health persists) vs reversible
  (C) endpoint Y-BOCS distribution at a clinical dose    -> mode + remission-ATOM (BC>.55) vs unimodal shift (BC<.55)
                                                            + Variability Ratio VR (Munkholm blindness test)
  (D) critical slowing: relaxation rate lambda(u)        -> ->0 at the fold (EWS) vs bounded (no EWS)
  (E) population remission fraction vs dose               -> saturating step w/ resistant plateau vs smooth
Outputs: fig_graded_vs_fold.pdf/png + signatures_summary.json + console table.
"""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
TAU_FOLD=12.0; TAU_GRADED=18.0
REMIT=12.0   # Y-BOCS remission threshold (endpoint)

# ---- shared drive: steady-state eC at occupancy u ----
_ucache={}
def eC_ss(u):
    if u<=1e-6: return eC_h
    if u in _ucache: return _ucache[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1])
    _ucache[u]=v; return v
def cdrug(u): return 1.0+BB*(eC_ss(u)-eC_h)   # drug sink coefficient

# ---- FOLD model ----
_GGRID=np.linspace(1e-4,2.9,9000)
def RHS_fold(G,al,u): return al*B.Sshape(G,THM,PS)-cdrug(u)*G
def fold_fps(al,u):
    f=al*np.array([B.Sshape(g,THM,PS) for g in _GGRID])-cdrug(u)*_GGRID; out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=_GGRID[i]-f[i]*(_GGRID[i+1]-_GGRID[i])/(f[i+1]-f[i])
        st=(RHS_fold(g+1e-3,al,u)-RHS_fold(g-1e-3,al,u))<0; out.append((g,st))
    return out
# cache S(G) on the grid once per alpha (S independent of u); attractor = highest stable FP reached from OCD start
_Scache={}
def _Sgrid(al):
    if al not in _Scache: _Scache[al]=al*np.array([B.Sshape(g,THM,PS) for g in _GGRID])
    return _Scache[al]
def fold_equil_from_OCD(al,G_start,u):
    """attractor from an OCD (high-G) initial condition at steady occupancy u (t->inf), by root-finding.
       From high G one settles onto the HIGHEST stable fixed point; when the OCD branch is annihilated by the
       saddle-node, that highest stable point IS health (G->0). Exact and ~1000x faster than integration."""
    f=_Sgrid(al)-cdrug(u)*_GGRID; sc=np.where(np.diff(np.sign(f))!=0)[0]
    stables=[]
    for i in sc:
        g=_GGRID[i]-f[i]*(_GGRID[i+1]-_GGRID[i])/(f[i+1]-f[i])
        if (RHS_fold(g+1e-3,al,u)-RHS_fold(g-1e-3,al,u))<0: stables.append(g)
    if not stables: return 1e-4                 # only health (G->0) is attracting
    hi=max(stables)
    # health boundary attractor: if RHS<0 all the way down to the grid floor, the low state is ~0
    return hi
def relax_rate_fold(al,u,G):   # -dRHS/dG at attractor, /tau  (1/time); ->0 at saddle-node
    d=1e-4; return -(RHS_fold(G+d,al,u)-RHS_fold(G-d,al,u))/(2*d)/TAU_FOLD

# ---- GRADED (IDR) null: G* = G0/(1+b dEc); relaxation rate = cdrug/tau (constant, bounded) ----
def graded_equil(G0,u): return G0/cdrug(u)
def relax_rate_graded(u): return cdrug(u)/TAU_GRADED

# =========================================================================
#  (A) single-patient equilibrium Y vs occupancy  (Y0 = 20)
# =========================================================================
Y0A=20.0; alA,G0A,okA=GP.alpha_for(Y0A,THM)
def find_u_fold(al,G0,lo=0.0,hi=0.985,thr=0.1,iters=70):
    """last occupancy at which the OCD attractor still exists; the saddle-node lies just above (bisection)."""
    if fold_equil_from_OCD(al,G0,hi)>=thr: return hi
    for _ in range(iters):
        mid=0.5*(lo+hi)
        if fold_equil_from_OCD(al,G0,mid)>=thr: lo=mid
        else: hi=mid
    return lo
uf_A=find_u_fold(alA,G0A)
# geometric refinement toward the saddle-node so critical slowing (lambda ~ sqrt(uf-u) -> 0) is actually
# resolved rather than floored by coarse sampling; densest sample sits ~3e-5 below the fold
us=np.unique(np.clip(np.concatenate([np.linspace(0,0.985,120), uf_A-np.geomspace(3e-5,0.06,60)]),0,0.985))
print(f"(A) u_fold(Y0=20)~{uf_A:.5f}; refined grid du_min={np.min(np.diff(np.sort(us))):.1e}")
Yfold_A=[]; Ygrad_A=[]; lam_fold=[]; lam_grad=[]
for u in us:
    Gf=fold_equil_from_OCD(alA,G0A,u); Yfold_A.append(M.Yread(Gf)); lam_fold.append(relax_rate_fold(alA,u,Gf))
    Gg=graded_equil(G0A,u); Ygrad_A.append(M.Yread(Gg)); lam_grad.append(relax_rate_graded(u))
Yfold_A=np.array(Yfold_A); Ygrad_A=np.array(Ygrad_A)
u_fold_A=us[np.argmax(np.abs(np.diff(Yfold_A))>3.0)] if np.any(np.abs(np.diff(Yfold_A))>3.0) else np.nan
jump_A=float(np.max(np.abs(np.diff(Yfold_A))))
print(f"(A) Y0=20: fold jump dY={jump_A:.1f} at u~{u_fold_A:.3f}; graded max step dY={np.max(np.abs(np.diff(Ygrad_A))):.2f}")

# =========================================================================
#  (B) hysteresis: up-ramp (from OCD) vs down-ramp (from health), Y0=20
# =========================================================================
def continue_branch(al,u_seq,G_init):
    Gs=[]; G=G_init; dt=0.02
    for u in u_seq:
        for _ in range(int(300/dt)): G+=dt*RHS_fold(G,al,u)/TAU_FOLD; G=min(max(G,1e-4),3.4)
        Gs.append(G)
    return np.array(Gs)
up_seq=us; dn_seq=us[::-1]
Gup=continue_branch(alA,up_seq,G0A)              # start on OCD branch, raise dose
Gdn=continue_branch(alA,dn_seq,Gup[-1])          # start on final (health) state, lower dose
Yup=np.array([M.Yread(g) for g in Gup]); Ydn=np.array([M.Yread(g) for g in Gdn])
# durable off-drug remission: after reaching health at high u, set u=0 and check state persists
G_offdrug=continue_branch(alA,[0.0],Gup[-1])[0]; Y_offdrug=M.Yread(G_offdrug)
graded_offdrug=M.Yread(graded_equil(G0A,0.0))
print(f"(B) fold: at u=0 after remission Y={Y_offdrug:.2f} (durable if ~0); graded off-drug Y={graded_offdrug:.2f} (=baseline {Y0A})")

# =========================================================================
#  (C) endpoint distribution at a clinical dose (u_clin ~ 40 mg SSRI)
# =========================================================================
def occ(mg,D50): return mg/(mg+D50)
u_clin=occ(40.,10.)   # ~0.80 at 'typical' potency
rng=np.random.default_rng(7)
N=4000; Yb=np.clip(rng.normal(24.,5.5,N),10.,39.5)
Yend_fold=[]; Yend_grad=[]; Yb_keep=[]; n_excl=0
for yb in Yb:
    al,G0,ok=GP.alpha_for(yb,THM)
    if al is None:            # below fold -> subclinical, essentially already near/at health floor
        n_excl+=1; continue
    Gf=fold_equil_from_OCD(al,G0,u_clin); Yend_fold.append(M.Yread(Gf))
    Yend_grad.append(M.Yread(graded_equil(G0,u_clin))); Yb_keep.append(yb)
Yend_fold=np.array(Yend_fold); Yend_grad=np.array(Yend_grad); Yb_keep=np.array(Yb_keep)
def bimod_coef(x):
    x=np.asarray(x,float); n=len(x); s=((x-x.mean())/x.std());
    g1=np.mean(s**3); k=np.mean(s**4)-3
    return (g1**2+1)/(k+3*(n-1)**2/((n-2)*(n-3)))
def VR(treated,base): return np.std(treated,ddof=1)/np.std(base,ddof=1)
BC_fold=bimod_coef(Yend_fold); BC_grad=bimod_coef(Yend_grad)
VR_fold=VR(Yend_fold,Yb_keep); VR_grad=VR(Yend_grad,Yb_keep)
atom_fold=100*np.mean(Yend_fold<REMIT); atom_grad=100*np.mean(Yend_grad<REMIT)
print(f"(C) u_clin={u_clin:.2f}, N_placed={len(Yb_keep)} (excl {n_excl} subclinical):")
print(f"    remission%   fold={atom_fold:.1f}  graded={atom_grad:.1f}")
print(f"    BC (>.55=bimodal)  fold={BC_fold:.3f}  graded={BC_grad:.3f}")
print(f"    VR (SD_end/SD_base) fold={VR_fold:.3f}  graded={VR_grad:.3f}   (Munkholm null VR~1)")

# =========================================================================
#  (E) population remission fraction vs dose
# =========================================================================
uu=np.linspace(0,0.985,60); rem_fold=[]; rem_grad=[]
sub=rng.choice(len(Yb_keep),800,replace=False); Ybs=Yb_keep[sub]
als=[GP.alpha_for(y,THM) for y in Ybs]
for u in uu:
    ff=[]; gg=[]
    for (al,G0,ok),y in zip(als,Ybs):
        ff.append(M.Yread(fold_equil_from_OCD(al,G0,u))<REMIT)
        gg.append(M.Yread(graded_equil(G0,u))<REMIT)
    rem_fold.append(100*np.mean(ff)); rem_grad.append(100*np.mean(gg))
rem_fold=np.array(rem_fold); rem_grad=np.array(rem_grad)

# save numbers
_gchi2=json.load(open('graded_freeb_fit.json'))['chi2']   # free-b matched-null (each model fits its own coupling b)
summary=dict(fold_chi2=6.269, graded_chi2=round(_gchi2,3),
             A_fold_jump=jump_A, A_u_fold=float(u_fold_A),
             B_fold_offdrug_Y=float(Y_offdrug), B_graded_offdrug_Y=float(graded_offdrug),
             C_u_clin=float(u_clin), C_remit_fold=float(atom_fold), C_remit_grad=float(atom_grad),
             C_BC_fold=float(BC_fold), C_BC_grad=float(BC_grad),
             C_VR_fold=float(VR_fold), C_VR_grad=float(VR_grad), C_n_excl=int(n_excl))
json.dump(summary,open('signatures_summary.json','w'),indent=2)
print("\nmeans indistinguishable: fold chi2={:.2f} vs graded chi2={:.2f} (within noise; case rests on signatures, not fit)".format(summary['fold_chi2'],summary['graded_chi2']))

# =========================================================================
#  FIGURE  (2x2)
# =========================================================================
BLUE='#1f3a93'; RED='#D1495B'; ORANGE='#E69F00'; GREY='0.45'
fig,ax=plt.subplots(2,2,figsize=(11,8.4))

# (A) equilibrium Y vs u
a=ax[0,0]
a.plot(us,Yfold_A,color=RED,lw=2.6,label='bistable fold')
a.plot(us,Ygrad_A,color=BLUE,lw=2.2,ls='--',label='graded (indirect-response) null')
a.axvline(u_fold_A,color=GREY,lw=0.9,ls=':')
a.annotate('saddle-node\njump',xy=(u_fold_A,10),xytext=(u_fold_A-0.36,13.5),
           arrowprops=dict(arrowstyle='-|>',color=GREY,lw=1.1),fontsize=9,color=GREY)
a.set_xlabel('SERT occupancy $u$'); a.set_ylabel('endpoint $Y_{\\mathrm{model}}$ (attractor)')
a.set_title('(A) one patient ($Y_{\\mathrm{model},0}$=20): discontinuous vs smooth',fontsize=10.5)
a.legend(fontsize=9); a.grid(alpha=.2); a.set_ylim(-1,22)

# (B) hysteresis
b=ax[0,1]
b.plot(up_seq,Yup,color=RED,lw=2.6,label='raise dose (from OCD)')
b.plot(dn_seq,Ydn,color=ORANGE,lw=2.2,ls='-',label='lower dose (from remission)')
b.plot(us,Ygrad_A,color=BLUE,lw=2.0,ls='--',label='graded null (reversible)')
b.scatter([0],[Y_offdrug],s=70,color=RED,zorder=5,edgecolor='k',lw=.6)
b.annotate('durable off-drug\nremission',xy=(0,Y_offdrug),xytext=(0.12,6.5),fontsize=9,color=RED,
           arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.0))
b.set_xlabel('SERT occupancy $u$'); b.set_ylabel('endpoint $Y_{\\mathrm{model}}$')
b.set_title('(B) hysteresis: one-way transition vs reversible',fontsize=10.5)
b.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(0.01,0.56)); b.grid(alpha=.2); b.set_ylim(-1,22)

# (C) endpoint distributions
c=ax[1,0]
bins=np.linspace(0,40,33)
c.hist(Yend_grad,bins=bins,alpha=0.55,color=BLUE,label=f'graded  (BC={BC_grad:.2f}, VR={VR_grad:.2f})',density=True)
c.hist(Yend_fold,bins=bins,alpha=0.55,color=RED,label=f'fold  (BC={BC_fold:.2f}, VR={VR_fold:.2f})',density=True)
c.axvline(REMIT,color=GREY,ls=':',lw=1)
c.set_xlabel('endpoint $Y_{\\mathrm{model}}$'); c.set_ylabel('density')
c.set_title(f'(C) endpoint distribution at $u$={u_clin:.2f}: mode+atom vs shifted lump',fontsize=10.5)
c.legend(fontsize=8.5); c.grid(alpha=.2)
c.text(0.98,0.62,'BC = bimodality coefficient\n($>$0.55 = bimodal)\nVR = variance ratio\n(endpoint SD / baseline SD)',
       transform=c.transAxes,fontsize=7.2,ha='right',va='top',color='0.25',
       bbox=dict(boxstyle='round',fc='white',ec='0.8',alpha=0.9))

# (D) critical slowing (relaxation rate)
d=ax[1,1]
d.plot(us,np.array(lam_fold),color=RED,lw=2.4,label='fold: $\\lambda(u)\\to0$ at fold')
d.plot(us,np.array(lam_grad),color=BLUE,lw=2.0,ls='--',label='graded: bounded')
d.axvline(u_fold_A,color=GREY,lw=0.9,ls=':')
d.set_xlabel('SERT occupancy $u$'); d.set_ylabel('relaxation rate $\\lambda$  (1/wk)')
d.set_title('(D) critical slowing: early-warning signal present vs absent',fontsize=10.5)
d.legend(fontsize=9,loc='upper center'); d.grid(alpha=.2)

plt.tight_layout(); plt.savefig('fig_graded_vs_fold.pdf'); plt.savefig('fig_graded_vs_fold.png',dpi=150)
print("\nwrote fig_graded_vs_fold.pdf / .png and signatures_summary.json")
