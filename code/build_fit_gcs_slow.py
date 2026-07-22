"""
RESTRUCTURED SSRI calibration -- gamma_C = 0.  Drug acts ONLY through the SLOW G_CS plasticity.

Two slow timescales (both known biology):
  (1) 5-HT1A AUTORECEPTOR desensitization: theta relaxes over tau_desens -> eC rises over weeks.
  (2) G_CS PLASTICITY: serotonin-dependent LTD (5-HT1B/eCB) lowers the corticostriatal weight over tau_G.

Serotonin balance (fast QSS, gamma_C=0 so NO direct eC->cortex):
  F0*(1+w*phe(G,0)) = (k_SERT(mg,D50) + k_OCT3)*eC + theta*h_max0*eC/(K+eC)
    - k_SERT dose-dependent (SSRI); k_OCT3 = SSRI-INSENSITIVE uptake-2 floor (OCT3/PMAT); autoreceptor term theta*..
Slow G_CS ODE (var_slow_fold / project_gcs_ode_parameters):
  tau_G dG_CS/dt = alpha*u(G_CS) - (kappa_eCB + b*eC(t))*G_CS ,  b = alpha_5HT1B = 0.0177 wk^-1/nM (fixed, lit)
    u(G) = phe(G,0)*phd1(G,0) normalized (Hebbian co-activity, LTP source); alpha pinned by baseline balance.
Readout (gamma_C=0): Y = 40*((phd1(G,0)-phid1_h)/(Qd1-phid1_h))^gamma_obs ,  gamma_obs = 2 (CANONICAL).
SSRI raises eC -> sink up -> G_CS falls slowly -> caudate falls -> Y falls. Onset from tau_desens + tau_G.

CANONICAL FROZEN ENGINE (2026-07-12): gamma_obs=2, k_SERT0=7/s, k_rem(kOCT3)=0.15*k_SERT0=1.05/s, K=10nM,
betaSERT=1.0 (SERT fully blockable; D50=Meyer ED50), w=0, eC_h=2.8nM, tau_desens=2wk. b & kappa_des CALIBRATED
(see fit_b_kappades.py + best_fit.json); NOTE module-level b_5HT1B/GAMMA_OBS below are DEFAULTS overridden by
the fit. Plasticity source = baseline-referenced BCM (gcs_bistable_population.Sshape).
"""
import numpy as np, time, sys
from scipy.optimize import brentq, differential_evolution
import build_fit_ssri_panel as P   # reuse circuit grid: P.phe(G,s), P.phd1(G,s)

# k_SERT0 = EFFECTIVE clearance-rate scale (NOT biochem Vmax/Km ~28000/hr; model scale-invariant, h_max0 co-scales).
# betaSERT=1.0 (2026-07-12): SERT FULLY BLOCKABLE. Meyer2004 Hill-fit asymptotes 86-102%, animals 92-95%; the
#   ~80% is CLINICAL-DOSE occupancy, NOT a ceiling. Setting beta=1 makes D50 = Meyer ED50 (half-occupancy) exact.
# kOCT3 = k_rem = SSRI-insensitive removal FLOOR (=Best2010's k_rem: OCT3/uptake-2 + glia/blood/diffusion).
#   k_rem/kSERT0 = 0.15, triangulated: Best fluox-peak SSRI-inaccessible ~0.1, Mathews SERT-KO eC 6.4x ~0.18,
#   Baganz OCT3 ~0.2. POST-Best functional significance + augmentability: Horton2013 (PMID 23785165, \V,
#   D-22+sub-eff fluvoxamine->antidepressant, OCT3-KO-abolished). Magnitude=modeling estimate, degenerate w/ autoreceptor.
# eC_h=2.8 nM = striatal basal extracell 5-HT ZNF (Mathews2004 PMID 15589347, \V). b_5HT1B=FITTED.
# LOCKED: (kSERT0,kOCT3,F0) EXACT scaling degeneracy (F0 pinned); only ratio k_rem/kSERT0 + dose-shape (D50; Meyer) matter.
kSERT0=7.; kOCT3=1.05; K=10.; betaSERT=1.0; w=0.0; eC_h=2.8   # kOCT3=0.15*kSERT0
phid1_h=P.phid1_h; Qd1=P.Qd1; b_5HT1B=0.0119; GAMMA_OBS=2.0   # DEFAULTS (overridden by fit); canonical gamma=2, calibrated b

# LTP source shape u(G)=phe*phd1 at s=0, normalized to G->0
_Gg=np.linspace(0.0,3.5,141)
_u=np.array([P.phe(g,0.0)*P.phd1(g,0.0) for g in _Gg]); _u=_u/_u[0]
def u_of(G): return float(np.interp(np.clip(G,_Gg[0],_Gg[-1]),_Gg,_u))
def phd1_0(G): return P.phd1(G,0.0)
def Yread(G): p=max(0.0,(phd1_0(G)-phid1_h)/(Qd1-phid1_h)); return 40*p**GAMMA_OBS
def G_of_Y(Ytarget):
    return brentq(lambda g: Yread(g)-Ytarget, 0.05, 3.4, xtol=1e-5)

F0=None; hmax0=None; theta_h=None
def kSERT(mg,D50): return kSERT0*(1-betaSERT*mg/(mg+D50))
def setup(kappa_des,hm0):
    # FIX (2026-07-09, grounded Best2010/Artigas1996): autoreceptor gates RELEASE, not clearance.
    #   source(release) = F0 / (1 + theta*hmax0*eC/(K+eC));  K=10nM = 5-HT1B autoreceptor Kd (Hagan2012).
    #   Higher theta (sensitive) or eC -> more suppression of release; theta desensitizes (tau_desens) -> release
    #   recovers -> eC rises. Dropped the old w*phe cortex->release term (deferred to raphe-dynamics extension).
    global F0,hmax0,theta_h
    hmax0=hm0; theta_h=1.0   # OptionB: desensitization driven by (eC-eC_h); at baseline eC=eC_h -> theta=1
    # pin F0 so baseline (theta_h=1, mg=0) gives eC=eC_h:  F0/(1+theta_h*hmax0*eC_h/(K+eC_h)) = (kSERT(0)+kOCT3)*eC_h
    F0=(kSERT(0,1.)+kOCT3)*eC_h*(1+theta_h*hmax0*eC_h/(K+eC_h))
def eC_qss(theta,mg,D50,G=None):
    # G unused now (gamma_C=0 and w=0 -> eC decoupled from circuit; feedforward dose->eC->G_CS).
    def bal(e): return F0/(1+theta*hmax0*e/(K+e)) - (kSERT(mg,D50)+kOCT3)*e
    return brentq(bal,0.05,400.,xtol=1e-7)

def trajectory(dose_fn,D50,Gocd,tau_G,kappa_eCB,kappa_des,tau_d,weeks):
    """Return absolute Y at requested weeks. G starts at Gocd (baseline OCD balance)."""
    eC0=eC_qss(theta_h,0.,D50,Gocd)
    alpha=(kappa_eCB+b_5HT1B*eC0)*Gocd/u_of(Gocd)   # pin LTP gain to baseline balance
    dt=0.05; th=theta_h; G=Gocd; wk_set=set(np.round(weeks,3)); out={}
    tmax=weeks[-1]
    for t in np.arange(0,tmax+dt/2,dt):
        rt=round(t,3)
        if rt in wk_set: out[rt]=Yread(G)
        mg=dose_fn(t); e=eC_qss(th,mg,D50,G)
        th += dt*((1-th)-kappa_des*(e-eC_h)*th)/tau_d
        G  += dt*(alpha*u_of(G)-(kappa_eCB+b_5HT1B*e)*G)/tau_G
        G   = min(max(G,0.02),3.4)
    return np.array([out[round(wk,3)] for wk in weeks])

def objective(x, detail=False):
    tau_G,kappa_eCB,kappa_des,tau_d,hm0,D50f,D50c=x
    setup(kappa_des,hm0)
    parts={}
    Gocd=G_of_Y(P.YB_TC)
    # FLX Tollefson absolute
    s=0.0
    for d in (20,40,60):
        Ym=trajectory(P.const(d),D50f,Gocd,tau_G,kappa_eCB,kappa_des,tau_d,P.WK_T)+P.NS(P.WK_T)
        wt=P.TOLn[d]/P.TOLsd[d]**2; s+=np.sum(wt*(Ym-P.TOL[d])**2)/np.sum(wt)
    parts['FLX']=s
    # CIT Montgomery change-from-baseline
    Yb0=Yread(Gocd); s=0.0
    for d in (20,40,60):
        Ym=trajectory(P.const(d),D50c,Gocd,tau_G,kappa_eCB,kappa_des,tau_d,P.WK_M)+P.NS(P.WK_M)
        s+=0.5*np.mean(((Yb0-Ym)-P.MONc[d])**2)
    parts['CIT']=s
    tot=sum(parts.values())
    return (tot,parts) if detail else tot

#        tau_G,     kappa_eCB,  kappa_des,  tau_desens,  h_max0,     D50_flx, D50_cit
BOUNDS=[(0.5,40.),(0.01,0.6),(0.02,0.5),(1.,8.),(100.,3000.),(2.,80.),(2.,80.)]
NAMES=['tau_G','kappa_eCB','kappa_des','tau_desens','h_max0','D50_flx','D50_cit']

if __name__=='__main__':
    if '--test' in sys.argv:
        x0=[8.,0.09,0.15,4.,1500.,6.,12.]; t0=time.time()
        tot,parts=objective(x0,detail=True)
        print(f'test={tot:.3f} in {time.time()-t0:.2f}s'); [print(f'  {k:6s}{v:.3f}') for k,v in parts.items()]
        # show onset shape for FLX60
        setup(x0[2],x0[4]); Gocd=G_of_Y(P.YB_TC)
        Y=trajectory(P.const(60),x0[5],Gocd,x0[0],x0[1],x0[2],x0[3],P.WK_T)
        print('FLX60 Y(t):',[f'{y:.1f}' for y in Y],'(base',round(P.YB_TC,1),')')
    else:
        print("gamma_C=0 slow-G_CS fit; optimizing...",flush=True); t0=time.time()
        res=differential_evolution(objective,BOUNDS,maxiter=80,popsize=18,tol=1e-6,seed=1,polish=True,disp=True,workers=-1)
        tot,parts=objective(res.x,detail=True)
        print(f"\nDONE {(time.time()-t0)/60:.1f} min; SSE={res.fun:.4f}")
        with open('gcs_slow_result.txt','w') as f:
            f.write(f"gamma_C=0 (fixed); gamma_obs=1; b_5HT1B=0.0177\nSSE={res.fun}\n")
            for n,v,bd in zip(NAMES,res.x,BOUNDS):
                rail=''
                if abs(v-bd[0])<1e-3*(bd[1]-bd[0]): rail='  <-- RAILED (lower)'
                if abs(v-bd[1])<1e-3*(bd[1]-bd[0]): rail='  <-- RAILED (upper)'
                line=f"{n} = {v:.5f}{rail}"; f.write(line+"\n"); print(line)
            f.write("\nper-study SSE:\n")
            for k,v in parts.items(): f.write(f"  {k:6s}{v:.4f}\n"); print(f"  {k:6s}{v:.4f}")
        np.save('gcs_slow_bestx.npy',res.x)
        print("wrote gcs_slow_result.txt")
